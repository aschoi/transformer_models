import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    Class definition for Postion-wise Feed Forward Network
    """

    d_model: int
    d_ff: int
    dropout_rate: float
    activation: str

    def __init__(
        self, 
        d_model, 
        d_ff, 
        dropout=0.1, 
        activation='relu'
    ) -> None:
        """
        Position-wise FFN Constructor
        
        Args:
            d_model:        <int>     Model dimension
            d_ff:           <int>     Feed-forward dimension (typically: 4 * d_model)
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(PositionwiseFeedForward, self).__init__()
        self.nn_linear1 = nn.Linear(d_model, d_ff)
        self.nn_linear2 = nn.Linear(d_ff, d_model)
        self.nn_dropout = nn.Dropout(dropout)

        if activation.lower() == 'relu':
            self.nnf_activation = F.relu
        elif activation.lower() == 'gelu':
            self.nnf_activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()


    def forward(
        self, 
        tnsr_X: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tnsr_X:      <tensor>
        Return:
            tnsr_X:      <tensor>
        """
        tnsr_X = self.nn_linear1(tnsr_X)
        tnsr_X = self.nnf_activation(tnsr_X)
        tnsr_X = self.nn_dropout(tnsr_X)
        tnsr_X = self.nn_linear2(tnsr_X)

        return tnsr_X


    def _init_weights(self) -> None:
        """Initialize weights"""
        nn.init.xavier_uniform_(self.nn_linear1.weight)
        nn.init.xavier_uniform_(self.nn_linear2.weight)
        nn.init.constant_(self.nn_linear1.bias, 0)
        nn.init.constant_(self.nn_linear2.bias, 0)