import os
from .logging import Logger


class WANDBLogging(Logger):
    def __init__(self, project_name):
        import wandb
        self.wandb = wandb
        self.wandb.login()
        
        self.project_name = project_name

        os.environ["WANDB_PROJECT"] = project_name

        ## don't upload models to wandb. Don't do that. We don't have space.
        ## If you are reading this and thinking,
        ## "himm wow i'd love to upload models to wandb. i should change the below enviroment variable"
        ## DO NOT. DO NOT CHANGE THE BELOW ENVIROMENT VARIABLE.

        # If you want to upload model checkpoints somewhere please reach out to Acoustic Species Project Leads.
        os.environ["WANDB_LOG_MODEL"] = "false"
        super().__init__()

    def __del__(self):
        self.wandb.finish()
