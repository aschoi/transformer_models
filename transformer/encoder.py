import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .utils import AddNorm


class TransformerEncoderLayer(nn.Module):
    """
    class definition for a single layer (of possibly many) within the whole Transformer Encoder
    """

    d_model: int
    num_selfAttn_heads: int
    d_ff: int
    dropout_rate: float
    activation: str

    def __init__(
        self,
        d_model,
        num_selfAttn_heads,
        d_ff,
        dropout_rate=0.1,
        activation='relu'
    ) -> None:
        """      
        Transformer Encoder Layer Constructor

        Args:
            d_model:                 <int>     Model dimension
            num_selfAttn_heads:      <int>     Number of Heads
            d_ff
            dropout_rate:        <float>   Dropout rate
            activation:         <string>  Activation Function ('relu' or 'gelu')
        """

        super(TransformerEncoderLayer, self).__init__()
        self.selfAttn_sublayer = MultiHeadAttention(d_model, num_selfAttn_heads, dropout_rate)  # Multi-Head Self-Attention
        self.posWise_feedForward_sublayer = PositionwiseFeedForward(d_model, d_ff, dropout_rate, activation)  # Position-wise FFN
        self.add_norm1_sublayer = AddNorm(d_model, dropout_rate)  # Add & Norm
        self.add_norm2_sublayer = AddNorm(d_model, dropout_rate)  # Add & Norm


    def forward(
        self,
        tnsr_X: torch.Tensor,
        tnsr_mask: torch.Tensor | None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tnsr_X:      <tensor> 
            tnsr_mask:   <tensor>
        
        Return:
            tnsr_X                   <tensor>    shape: (batch_size, seq_len_q, d_model)
            tnsr_attention_weights   <tensor>    shape: (batch_size, num_heads, seq_len_q, seq_len_q)
        """
        # Multi-Head Self-Attention + Add & Norm
        tnsr_selfAttn_output, tnsr_selfAttn_weights = self.selfAttn_sublayer(tnsr_X, tnsr_X, tnsr_X, tnsr_mask)
        tnsr_X = self.add_norm1_sublayer(tnsr_X, tnsr_selfAttn_output)

        # Position-wise Feed-Forward + Add & Norm
        tnsr_ff_output = self.posWise_feedForward_sublayer(tnsr_X)
        tnsr_X = self.add_norm2_sublayer(tnsr_X, tnsr_ff_output)

        return tnsr_X, tnsr_selfAttn_weights


class TransformerEncoder(nn.Module):
    """
    Combined layer cake that is the Transformer Encoder
    """

    num_encoder_layers: int
    d_model: int
    num_attn_heads: int
    d_ff: int
    dropout_rate: float
    activation: str

    def __init__(
        self, 
        num_encoder_layers, 
        d_model, 
        num_attn_heads, 
        d_ff, 
        dropout_rate=0.1, 
        activation='relu'
    ) -> None:
        """
        <Stack> of Transformer Encoder Layers:      stack<TransformerEncoderLayer>

        Args:
            num_encoder_layers      <int>     Number of Transformer Encoder Layers
            d_model:             <int>     Model dimension
            num_attn_heads:      <int>     Number of Heads
            d_ff                    <int>    
            dropout_rate:        <float>   Dropout rate
            activation:          <string>  Activation Function ('relu' or 'gelu')
        """

        super(TransformerEncoder, self).__init__()

        self.stack_transformerEncoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_attn_heads, d_ff, dropout_rate, activation
            ) for encoderLayer in range(num_encoder_layers)
        ])

        self.num_encoder_layers = num_encoder_layers


    def forward(
        self, 
        tnsr_X: torch.Tensor, 
        tnsr_mask: torch.Tensor | None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Encoder <stack>

        Args:
            X:      <tensor>
            mask    <tensor>
        Return:
            X:                          <tensor>            shape: (batch_size, seq_len_q, d_model)
            attention_weights_stack     <list <tensor>>     element shape: (batch_size, num_heads?, seq_len_q, seq_len_q)
        """
        attn_weights_stack = []

        for encoderLayer in self.stack_transformerEncoder_layers:
            tnsr_X, tnsr_attn_weights = encoderLayer(tnsr_X, tnsr_mask)
            attn_weights_stack.append(tnsr_attn_weights)

        return tnsr_X, attn_weights_stack

