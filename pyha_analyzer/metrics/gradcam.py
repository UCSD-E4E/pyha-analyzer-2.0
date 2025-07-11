import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    #capture behavior during inference and backprop
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None #gradient of the output with respect to the predicted class
                                # high gradient means very important for final decision
        self.activations = None #output of convolutional layer
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        # Register hooks that actually tracks/records the activations and gradients
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def generate(self, input_tensor, class_idx=None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        output = self.model(input_tensor)
        logits=output.logits
        if class_idx is None:
            #class_idx = output.argmax(dim=1).item()
            class_idx = logits.argmax(dim=1).item()

        #tells it to backpropagate to see how much pixel affected class score
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        #average each channel's gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        #weighted sum of all activation maps (tells us where classifier looked)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        #only keep positive (waht helped to classify not hurt)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy() # need to do #cpu() to prevent memory error issues
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize because spectogram will be normalized
        # self.plot_gradcam_histogram(cam)
        return cam


    def show_gradcam_overlay(self, item, cam_output, p):
        #print(pdfPath)
        spectrogram_tensor = torch.tensor(item["audio_in"], dtype=torch.float32)
        spectrogram_np = spectrogram_tensor.squeeze(0).numpy()  # shape: (H, W)

        # Resize spectrogram to match CAM output size
        cam_H, cam_W = cam_output.shape
        spectrogram_resized = cv2.resize(spectrogram_np, (cam_W, cam_H))

        # Normalize spectrogram
        spectrogram_resized = (spectrogram_resized - spectrogram_resized.min()) / \
                            (spectrogram_resized.max() - spectrogram_resized.min())

        label = item.get("labels", "Unknown")
        if ((label ==[ 0, 1]).all()):
            currLabel = "Non_Degraded_Reef"
        elif ((label == [1,0]).all()):
            currLabel = "Degraded_Reed"
        else:
            currLabel="Unknown"
        filepath = item.get("filepath", "Unknown")
        filepath = ("/").join(filepath.split("/")[-3:])

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Plot spectrogram (left)
        axs[0].imshow(spectrogram_np)
        axs[0].axis("off")

        # Plot Grad-CAM overlay (right)
        axs[1].imshow(spectrogram_resized, cmap='gray', aspect='auto', origin='lower')
        im = axs[1].imshow(cam_output, cmap='jet', alpha=0.7, extent=[0, cam_W, 0, cam_H])
        axs[1].axis("off")

        # Add shared colorbar
        fig.colorbar(im, ax=axs[1], label="CAM intensity", shrink=0.8)

        # Set a single title for the whole figure
        fig.suptitle(f"{filepath}", fontsize=12)

        # Adjust layout to prevent title overlap
        plt.subplots_adjust(top=0.85)

        # Save to PDF or show
        #fig.savefig(p, format='pdf')
        # plt.show()
        plt.close(fig)

    #plot histogram for one individual gradcam 
    def plot_gradcam_histogram(self, gradcam, title="Grad-CAM Histogram"):
        plt.figure(figsize=(6, 4))
        plt.hist(gradcam.flatten(), bins=50, range=(0, 1), color="blue", alpha=0.7)
        plt.title(title)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
