## Based on https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/models/efficientnet.py#L10
from transformers import AutoConfig, EfficientNetForImageClassification
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import nn
from typing import List
from .base_model import BaseModel, has_required_inputs
import torch
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    Also supports SimCLR-style loss when labels=None and mask=None.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...], at least 3 dims required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # logits: [bsz*anchor_count, bsz*contrast_count]
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mean log-likelihood over positives
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class EfficentNet(nn.Module, BaseModel):
    def __init__(self, num_channels: int = 1, num_classes: int = None,
                 supcon_temperature: float = 0.07, supcon_contrast_mode: str = "all"):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(
            "google/efficientnet-b1",
            num_labels=self.num_classes,
            num_channels=self.num_channels,
            problem_type="multi_label_classification"
        )
        self.model = EfficientNetForImageClassification(config)

        # contrastive loss module
        self.supcon = SupConLoss(
            temperature=supcon_temperature,
            contrast_mode=supcon_contrast_mode,
            base_temperature=supcon_temperature
        )

    def _embed_from_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, C, H, W]
        returns embedding: [B, D] where D=1280 for efficientnet-b1
        """
        # Forward through EfficientNet backbone (no classifier head)
        out = self.model.efficientnet(pixel_values=pixel_values, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, 1280, H', W']
        pooled = F.adaptive_avg_pool2d(last_hidden, output_size=(1, 1))  # [B, 1280, 1, 1]
        emb = pooled.squeeze(-1).squeeze(-1)  # [B, 1280]
        return emb

    @has_required_inputs()
    def forward(self, labels, **kwrgs):
        """
        Keeps return type the same: dict with keys {"loss","logits"}.

        Behavior:
        - If audio_in is 4D [B,C,H,W] => original EfficientNetForImageClassification loss (cross-entropy / BCE).
        - If audio_in is 5D [B,V,C,H,W] => supervised/SimCLR-style contrastive loss on embeddings,
          and logits are produced from view 0 (for metrics/eval compatibility).
        """
        audio_in = kwrgs["audio_in"]

        # Case A: standard single-view classification path (keeps original behavior)
        if audio_in.ndim == 4:
            out = self.model(pixel_values=audio_in, labels=labels)
            return {"loss": out.loss, "logits": out.logits}

        # Case B: multi-view contrastive path
        if audio_in.ndim != 5:
            raise ValueError(f"Expected audio_in to be 4D or 5D, got shape {tuple(audio_in.shape)}")

        # audio_in: [B, V, C, H, W]
        bsz, n_views, C, H, W = audio_in.shape

        # Flatten views -> embed -> reshape back
        flat = audio_in.view(bsz * n_views, C, H, W)
        emb = self._embed_from_pixel_values(flat)              # [B*V, D]
        emb = F.normalize(emb, dim=1)                          # common for contrastive learning
        features = emb.view(bsz, n_views, -1)                  # [B, V, D]

        # Labels handling:
        # - If labels is None: SimCLR-style instance contrast (positives are other views of same sample)
        # - If labels is int class ids [B]: supervised contrastive positives share class
        # - If labels is multi-hot [B,num_classes]: we build a mask based on shared labels
        mask = None
        supcon_labels = None

        if labels is None:
            supcon_labels = None
            mask = None
        else:
            # labels may be float multi-hot for multi-label classification
            if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                # Build positive mask: samples i,j are positives if they share at least one label.
                # This is the least-assumption way to "supervise" in a multi-label setting.
                # (If you *only* want instance positives, set labels=None in the call path.)
                with torch.no_grad():
                    mask = (labels.float() @ labels.float().T) > 0.0
                mask = mask.float().to(features.device)
                supcon_labels = None
            else:
                # Assume class index style labels [B]
                supcon_labels = labels

        loss = self.supcon(features, labels=supcon_labels, mask=mask)

        # Produce logits from view 0 for compatibility with Trainer metrics/eval.
        view0 = audio_in[:, 0, ...]  # [B,C,H,W]
        logits_out = self.model(pixel_values=view0).logits

        return {"loss": loss, "logits": logits_out}

    def get_embedding(self, audio_batch: torch.Tensor):
        # Kept for compatibility; unchanged behavior.
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            audio_batch = audio_batch.to(device)
            emb = self._embed_from_pixel_values(audio_batch)
            return emb
         