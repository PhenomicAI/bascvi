import torch
import torch.nn as nn

class BatchDiscriminator(nn.Module):
    """
    A standalone classifier that maps the latent embedding 'z' to
    batch logits.
    """
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass with optional gradient reversal
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        reverse_grad : bool, default=False
            Whether to apply gradient reversal
        alpha : float, default=1.0
            Strength of gradient reversal
            
        Returns:
        --------
        torch.Tensor
            Batch classification logits
        """

        x = self.hidden_layers(x)
        return self.output_layer(x)
