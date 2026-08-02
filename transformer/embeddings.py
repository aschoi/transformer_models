import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Class for Positional Encoding
    """

    d_model: int
    seq_max_len: int
    dropout_rate: float

    def __init__(
        self,
        d_model,
        seq_max_len=5000,
        dropout_rate=0.1
    ) -> None:
        """
        Sinusoidal Positional Encoding
        Args:
            d_model:
            seq_max_len:
            dropout_rate:
        """
        super(PositionalEncoding, self).__init__()
        self.nn_dropout = nn.Dropout(dropout_rate)

        # Create Positional Encoding Matrix
        # Bascially, think of it as: the tokenized sequence is a vector. 
        # And every token has d_model number of features associated with it.
        tnsr_pe = torch.zeros(seq_max_len, d_model)
        tnsr_position = torch.arange(0, seq_max_len, dtype=torch.float32).unsqueeze(1) 

        # Compute the div_term_tensor for Sinusoidal Pattern
        # div_term shape:  (ceil(d_model / 2), )
        tnsr_div_term = torch.exp(
            torch.arange(
                0, d_model, 2, dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        # IF dim_model is even, THEN pe[:, 1::2] has dim_model/2 cols.
        tnsr_pe[:, 0::2] = torch.sin(tnsr_position * tnsr_div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        # IF dim_model odd, THEN pe[:, 1::2] slice has (dim_model-1)/2 cols.
        # div_term has (d_model+1)/2 elements. So div_term[:-1] is used, which has (d_model-1)/2 elements.
        if d_model % 2 == 1:
            tnsr_pe[:, 1::2] = torch.cos(tnsr_position * tnsr_div_term[:-1])
        else:
            tnsr_pe[:, 1::2] = torch.cos(tnsr_position * tnsr_div_term)

        # Add a dimension for "batch" and register as buffer
        tnsr_pe = tnsr_pe.unsqueeze(0)
        # We are Registering it to buffer to MAKE SURE it isn't trainable!
        # Values are calculated once, then must remain fixed
        self.register_buffer('tnsr_pe', tnsr_pe)


    def forward(
        self, 
        X: torch.Tensor
    ) -> None:
        """
        Add Positional Encoding to input embedding

        Args:
            X
        Return:
            outputs:
        """
        seq_len = X.size(1)
        X = X + self.tnsr_pe[:, :seq_len]

        return self.nn_dropout(X)


class TokenEmbedding(nn.Module):
    """
    Token Embedding
    """

    vocab_size: int
    d_model: int
    padding_idx: int
    param_init: str

    def __init__(
        self,
        vocab_size,
        d_model,
        padding_idx,
        param_init='xavier'
    ) -> None:
        """
        Token Embedding Layer

        Args:
            vocab_size: 
            d_model:
            padding_idx:
            param_init:     
        """
        super(TokenEmbedding, self).__init__()
        self.nn_embedding = nn.Embedding(vocab_size, d_model, padding_idx)
        self.d_model = d_model
        self._init_params(param_init)


    def _init_params(self, param_init: str) -> None:
        if (param_init == 'xavier'):
            nn.init.xavier_uniform_(self.nn_embedding.weight)


    def forward(self, X: torch.Tensor) -> None:
        """
        Convert token indices to embeddings scaled by sqrt(d_model)
        Args:
            X:      <tensor>
        Return:
            output:     <tensor>
        """

        # Embed then scale by sqrt(d_model)
        return self.nn_embedding(X) * math.sqrt(self.d_model)