import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    r"""
    Attention Module
    """

    d_model: int
    num_attn_heads: int
    dropout_rate: float
    weights_init: str

    def __init__(
        self,
        d_model,
        num_attn_heads,
        dropout_rate=0.1,
        weights_init='xavier'
    ) -> None:
        """
        Multi-Head Attention Module Constructor

        Args:
            d_model:         <int>    Model Dimension
            num_attn_heads:  <int>    Number of Attention Heads
            dropout:         <float>  Dropout rate
        """

        super(MultiHeadAttention, self).__init__()
        assert d_model % num_attn_heads == 0, "d_model must be divisible by num_attn_heads"

        self.d_model = d_model
        self.num_attn_heads = num_attn_heads
        self.d_k = d_model // num_attn_heads

        # Linear Projections for Query, Key, Value, and output
        self.nn_linearProj_w_Q = nn.Linear(d_model, d_model)
        self.nn_linearProj_w_K = nn.Linear(d_model, d_model)
        self.nn_linearProj_w_V = nn.Linear(d_model, d_model)
        self.nn_linearProj_w_output = nn.Linear(d_model, d_model)

        self.nn_dropout = nn.Dropout(dropout_rate)
        self._init_weights(weights_init)


    def _init_weights(
        self, 
        weights_init: str
    ) -> None:
        '''
        Initialize weights using Xavier Uniform Initialization 

        Args:
            weights_init:       
        '''

        if (weights_init == 'xavier'):
            for nn_linear_proj in [self.nn_linearProj_w_Q, self.nn_linearProj_w_K, self.nn_linearProj_w_V, self.nn_linearProj_w_output]:
                nn.init.xavier_uniform_(nn_linear_proj.weight)
                if nn_linear_proj.bias is not None:
                    nn.init.constant_(nn_linear_proj.bias, 0)

    
    def scaled_dotProd_attention(
        self,
        tnsr_q: torch.Tensor,
        tnsr_k: torch.Tensor,
        tnsr_v: torch.Tensor,
        tnsr_mask: torch.Tensor=None
    ) -> None:
        """
        Scaled Dot Product Attention for Multiple Heads.

        Args:
            tnsr_q: 
            tnsr_k:
            tnsr_v:
            tnsr_mask:
        Return:
            tnsr_output:
            tnsr_attention_weights: 
        """

        # 1) transpose k
        # 2) matmul
        # 3) elementwise division with sqrt(d_k).   sqrt(d_k) means scaled version of d_k
        tnsr_scores = torch.matmul(tnsr_q, tnsr_k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if tnsr_mask is not None:
            # mask == 0 creates a boolean tensor. True = 0 and False = non-zero
            # .masked_fill(condition, value) ==> replaces every score where condition = True with value
            # instead of using 0, -1e9 is essentially 0 
            # Thus "masking" the intended coordinates
            tnsr_scores = tnsr_scores.masked_fill(tnsr_mask==0, -1e9)

        tnsr_attention_weights = F.softmax(tnsr_scores, dim=-1)
        tnsr_attention_weights = self.nn_dropout(tnsr_attention_weights)

        tnsr_output = torch.matmul(tnsr_attention_weights, tnsr_v)
        
        return tnsr_output, tnsr_attention_weights
    

    def forward(
        self,
        tnsr_query: torch.Tensor,
        tnsr_key: torch.Tensor,
        tnsr_value: torch.Tensor,
        tnsr_mask: torch.Tensor | None=None
    ) -> None:
        """
        Forward Propogation for Multi-head Attention.

        Args:
            tnsr_query:      Shape: (batch_size, d_model)
            tnsr_key:        Shape: (batch_size, seq_len, d_model)
            tnsr_value:      Shape: (batch_size, seq_len, d_model)
        Return:
            tnsr_output:
            tnsr_att_weights:
        """

        batch_size = tnsr_query.size(0)
        seq_len_q = tnsr_query.size(1)
        seq_len_k = tnsr_key.size(1)     # seq_len_k and seq_len_v are ALWAYS meant to be the same.
        seq_len_v = tnsr_value.size(1)   # But for the sake of not cutting corners, I name it explicitly

        # Linear Projections & Reshape for multi-head attention
        tnsr_Q = self.nn_linearProj_w_Q(tnsr_query).view(batch_size, seq_len_q, self.num_attn_heads, self.d_k).transpose(1, 2)
        tnsr_K = self.nn_linearProj_w_K(tnsr_key).view(batch_size, seq_len_k, self.num_attn_heads, self.d_k).transpose(1, 2)
        tnsr_V = self.nn_linearProj_w_V(tnsr_value).view(batch_size, seq_len_v, self.num_attn_heads, self.d_k).transpose(1, 2)

        if tnsr_mask is not None:
            if tnsr_mask.dim() == 2:     # [B, K]
                tnsr_mask = tnsr_mask.unsqueeze(1).unsqueeze(1)
            elif tnsr_mask.dim() == 3:   # [B, Q, K] or [1, T, T]
                tnsr_mask = tnsr_mask.unsqueeze(1)

        # Apply Attention
        tnsr_attn_output, tnsr_attn_weights = self.scaled_dotProd_attention(tnsr_Q, tnsr_K, tnsr_V, tnsr_mask)

        # Concatenate heads and apply output Linear Projection
        tnsr_attn_output = tnsr_attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        tnsr_output = self.nn_linearProj_w_output(tnsr_attn_output)

        return tnsr_output, tnsr_attn_weights

