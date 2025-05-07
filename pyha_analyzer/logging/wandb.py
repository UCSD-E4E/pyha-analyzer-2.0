import os
import wandb
from transformers import TrainingArguments
from .logging import Logger

class WANDBLogging(Logger):
    def __init__(self, project_name):
        wandb.login()
        self.project_name = project_name

        os.environ["WANDB_PROJECT"]=project_name

        ## don't upload models to wandb. Don't do that. We don't have space.
        ## If you are reading this and thinking, 
        ## "himm wow i'd love to upload models to wandb. i should change the below enviroment variable" 
        ## DO NOT. DO NOT CHANGE THE BELOW ENVIROMENT VARIABLE.

        # If you want to upload model checkpoints somewhere please reach out to Acoustic Species Project Leads.
        os.environ["WANDB_LOG_MODEL"] = "false" 
        super().__init__()

    def modify_trainer(self, trainer):
        trainer.training_args

    def modify_training_args(working_dir, training_args=None):
        if training_args is None:
            return TrainingArguments(
                working_dir,
                #report_to="wandb"
            )
    
        else:
            # training_args.report_to = "wandb" 
            # according to documentation, default is "all", so it will send to wandb if it exists. 
            return training_args

    def __del__():    
        wandb.finish()

    


