from abc import ABC, abstractmethod
import torch


class BaseTrainer(ABC):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(self.config.training.device)

    def load(self):
        self.load_dataset()
        self.build_model()
        self.build_optimizer()
        self.logger.save_config()

    def count_parameters(self):
        """ Count trainable parameters in model. """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def print_parameters(self):
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

    @abstractmethod
    def load_dataset(self):
        """Load dataset and dataloader."""

    @abstractmethod
    def build_model(self):
        """Build the model."""

    @abstractmethod
    def build_optimizer(self):
        """Load the optimizer."""

    @abstractmethod
    def train(self):
        """Run the training process."""
