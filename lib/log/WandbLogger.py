import os
import wandb
import torch.nn as nn

from lib.log.BaseLogger import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(
            self,
            wandb_project_name: str,
            api_key: str,
            config: dict,
            entity: str = None,
            models: [nn.Module] = [],
            run_name: str = None):
        super().__init__(config=config)

        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()

        self.run: wandb = wandb.init(project=wandb_project_name, entity=entity, name=run_name)
        self.graphs = wandb.watch(models, log_freq=100, log="all")
        wandb.config.update(self.config)

    def log(self, data: dict):
        wandb.log(data)

    def dispose(self):
        self.run.finish()


class WandbSweepLogger(BaseLogger):

    def __init__(
            self,
            config: dict):
        super().__init__(config=config)

    def log(self, data: dict):
        wandb.log(data)

    def dispose(self):
        pass