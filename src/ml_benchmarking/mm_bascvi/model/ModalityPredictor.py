import torch
import torch.nn as nn

class ModalityPredictor(nn.Module):
    """
    A standalone classifier that maps the latent embedding 'z' to
    modality logits (e.g., 3 modalities) through multiple hidden layers
    with nonlinearities.
    """
    def __init__(self, latent_dim: int, num_modalities: int = 3, hidden_dims: list = [128, 64]):
        super().__init__()
        
        # Build layers list starting with input dim
        dims = [latent_dim] + hidden_dims + [num_modalities]
        layers = []
        
        # Create layers with ReLU activation between them
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # No activation after final layer
                layers.append(nn.ReLU())
                
        self.classifier = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            logits: [batch_size, num_modalities]
        """
        return self.classifier(z)
