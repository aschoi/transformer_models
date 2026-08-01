import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .utils import AddNorm


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """

    d_model: int
    num_attn_heads: int
    d_ff: int
    dropout_rate: float
    activation: str

    def __init__(
        self, 
        d_model, 
        num_attn_heads, 
        d_ff, 
        dropout_rate=0.1, 
        activation='relu'
    ) -> None:
        """
        Transformer Decoder Layer Constructor (layer is equiv to head)

        Args:
            d_model:        <int>     Model dimension
            num_attn_heads       <int>     Number of Heads
            d_ff:           <int>    
            dropout:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')

        """
        super(TransformerDecoderLayer, self).__init__()

        # Masked Multi-Head Self-Attention + AddNrom
        self.sublayer_selfAttn = MultiHeadAttention(d_model, num_attn_heads, dropout_rate)
        self.sublayer_addNorm1 = AddNorm(d_model, dropout_rate)

        # Encoder-Decoder Multi-Head Attention + AddNorm
        self.sublayer_crossAttn = MultiHeadAttention(d_model, num_attn_heads, dropout_rate)
        self.sublayer_addNorm2 = AddNorm(d_model, dropout_rate)

        # Position-wise FFN + AddNorm
        self.sublayer_feedForward = PositionwiseFeedForward(d_model, d_ff, dropout_rate, activation)
        self.sublayer_addNorm3 = AddNorm(d_model, dropout_rate)    
        

    def forward(
        self, 
        tnsr_X: torch.Tensor, 
        tnsr_encoder_output: torch.Tensor, 
        tnsr_selfAttn_mask: torch.Tensor | None=None, 
        tnsr_crossAttn_mask: torch.Tensor | None=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            X:                      <tensor>    shape: (batch_size, seq_len_q,   d_model)
            encoder_output:         <tensor>    shape: (batch_size, seq_len_q,   d_model)
            self_attention_mask:    <tensor>    shape: (batch_size,         1, seq_len_q, seq_len_q)
            cross_attention_mask:   <tensor>    shape: (batch_size,         1,         1, seq_len_q)
        Return:
            final_output:           <tensor>    shape: (batch_size, seq_len_q, d_model)
            (self_attn_weights, cross_attn_weights)     tuple(<tensor>, <tensor>)
        """

        # 1) Masked Multi-head Self-Attention + Add & Norm
        tnsr_selfAttn_output, tnsr_selfAttn_weights = self.sublayer_selfAttn(tnsr_X, tnsr_X, tnsr_X, tnsr_selfAttn_mask)
        tnrs_mid_output = self.sublayer_addNorm1(tnsr_X, tnsr_selfAttn_output)

        # 2) Encoder-Decoder Multi-Head Attention + Add & Norm
        tnsr_crossAttn_output, tnsr_crossAttn_weights = self.sublayer_crossAttn(
            tnrs_mid_output, tnsr_encoder_output, tnsr_encoder_output, tnsr_crossAttn_mask
        )
        tnrs_mid_output = self.sublayer_addNorm2(tnrs_mid_output, tnsr_crossAttn_output)

        # 3) Position-wise FFN + Add & Norm
        tnsr_ff_output = self.sublayer_feedForward(tnrs_mid_output)
        tnsr_final_output = self.sublayer_addNorm3(tnrs_mid_output, tnsr_ff_output)

        return tnsr_final_output, (tnsr_selfAttn_weights, tnsr_crossAttn_weights)



class TransformerDecoder(nn.Module):
    """
    Transformer Decoder full cake layers
    """

    num_decoder_layers: int 
    d_model: int 
    num_attn_heads: int
    d_ff: int 
    dropout_rate: float
    activation: str

    def __init__(
        self, 
        num_decoder_layers, 
        d_model, 
        num_attn_heads, 
        d_ff, 
        dropout_rate=0.1, 
        activation='relu'
    ) -> None:
        """
        [stack] of Transformer Decoder Layers:    stack<TransformerDecoderLayer>

        Args:
            num_decoder_layers      <int>     Number of Transformer Encoder Layers
            d_model:        <int>     Model dimension
            num_attn_heads       <int>    Number of Heads
            d_ff:           <int>    
            dropout_rate:        <float>   Dropout rate
            activation:     <string>  Activation Function ('relu' or 'gelu')
        """
        super(TransformerDecoder, self).__init__()

        self.num_decoder_layers = num_decoder_layers
        self.stack_transformerDecoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_attn_heads, d_ff, dropout_rate, activation
            ) for decoderLayer in range(num_decoder_layers)
        ])


    def forward(
        self, 
        tnsr_X: torch.Tensor, 
        tnsr_encoder_output: torch.Tensor, 
        tnsr_selfAttn_mask: torch.Tensor | None=None, 
        tnsr_crossAttn_mask: torch.Tensor | None=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward propogation for Transformer decoder

        Args:
            tnsr_X:                      <tensor>    shape: (batch_size, seq_len_q,   d_model)
            tnsr_encoder_output:         <tensor>    shape: (batch_size, seq_len_q,   d_model)
            tnsr_selfAttn_mask:    <tensor>    shape: (batch_size,         1, seq_len_q, seq_len_q)
            tnsr_crossAttn_mask:   <tensor>    shape: (batch_size,         1,         1, seq_len_q)
        Return:
            tnsr_X:           <tensor>   shape: (batch_size, seq_len_q,   d_model)
            (all_self_attn_weights, all_cross_attn_weights)    tuple(list<tensor>, list<tensor>)    both of shape: (batch_size, num_heads?, seq_len_q, seq_len_q)
        """
        all_self_attn_weights = []
        all_cross_attn_weights = []

        for decoderLayer in self.stack_transformerDecoder_layers:
            tnsr_X, (self_attn_weights, cross_attn_weights) = decoderLayer(
                tnsr_X, tnsr_encoder_output, tnsr_selfAttn_mask, tnsr_crossAttn_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        return tnsr_X, (all_self_attn_weights, all_cross_attn_weights)
