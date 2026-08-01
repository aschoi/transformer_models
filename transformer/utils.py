import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Add and Norm
    """

    d_model: int
    dropout_rate: int

    def __init__(
        self, 
        d_model, 
        dropout_rate=0.1
    ) -> None:
        """
        Add & Norm Layer Constructor
        Residual Connection + Layer Normalization

        Args:
            d_model:    <int>    Model Dimension
            dropout:    <float>  Dropout rate
        """
        super(AddNorm, self).__init__()
        self.nn_layer_norm = nn.LayerNorm(d_model)
        self.nn_dropout = nn.Dropout(dropout_rate)


    def forward(
        self, 
        tnsr_X, 
        tnsr_sublayer_output
    ) -> None:
        """
        Args:
            X:                  <tensor>  Original input (residual connection)
            sublayer_output:    <tensor>  Output from sublayer (attention or FFN)
        Returns:
            <tensor>  Normalized output after Residual Connection 
        """
        # Add:  Residual Connection + Apply Dropout
        tnsr_output = tnsr_X + self.nn_dropout(tnsr_sublayer_output)

        # Norm: Apply Layer Normalization
        return self.nn_layer_norm(tnsr_output)

