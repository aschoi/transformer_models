import torch
import torch.nn as nn
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.embeddings import TokenEmbedding, PositionalEncoding


class Transformer(nn.Module):
    r"""
    Complete Transformer Model
    """

    src_vocab_size: int
    tgt_vocab_size: int
    src_pad_id: int
    tgt_pad_id: int
    d_model: int
    num_attention_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    d_ff: int
    dropout: float
    activation: str
    max_seq_len: int
    param_init: str

    def __init__(
        self,
        src_vocab_size, 
        tgt_vocab_size,
        src_pad_id,
        tgt_pad_id,
        d_model,
        num_attn_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        dropout=0.1,
        activation='relu',
        max_seq_len=5000,
        param_init='xavier_normal'
    ) -> None:
        """
        Transformer Constructor
        
        Args:
            src_vocab_size:             
            tgt_vocab_size:             
            src_pad_id:                 
            tgt_pad_id:                 
            d_model,
            num_attention_heads:                
            num_encoder_layers:         
            num_decoder_layers:         
            d_ff:                   
            dropout:            
            activation:     
            max_seq_len:            
            param_init:             

        """

        super(Transformer, self).__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder & Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_attn_heads, d_ff, dropout, activation
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_attn_heads, d_ff, dropout, activation
        )

        # Output Projection aka Final Output
        self.nn_output_projection = nn.Linear(d_model, tgt_vocab_size)

        # init parameters aka WEIGHTS
        self._init_parameters(param_init)


    def _init_parameters(
        self, 
        param_init: str
    ) -> None:
        """
        Initial Model Parameters aka Weights

        Args:
            param_init:     Parameter initialization choice
        """
        if (param_init=='xavier_normal'):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
    


    def create_padding_mask(
        self,
        tnsr_token_ids: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Create an Atention Mask that prevents Attention form reading PAD tokens

        Args:
            tnsr_token_ids:     Integer token IDs.                  shape: (batch_size, seq_len)
            pad_token_id:       Integer ID used for PAD token.
        
        Return:
            tnsr_padding_mask:  Boolean Mask        Shape: (batch_size, 1, 1, seq_len)
                                Along final dim:    True  == this key pos contains real token
                                                    False == this key pos contains padding
        """

        # Create a Tensor of True/False elements. Essentially scanning through entire Tensor
        # Every tokenID == to pad_token_id gets a False. Everything else gets a True
        #       True  IF real token.
        #       False IF padding token.
        # Shape: (batch_size, seq_len)
        tnsr_isReal_token = tnsr_token_ids != pad_token_id

        # Add a dim for Attention Heads
        # Shape: (batch_size, 1, seq_len)
        tnsr_isReal_token = tnsr_isReal_token.unsqueeze(1)

        # Add a dim for Query positions
        # Shape: (batch_size, 1, 1, seq_len)
        tnsr_padding_mask = tnsr_isReal_token.unsqueeze(2)

        return tnsr_padding_mask


    def create_causal_mask(
        self,
        tnsr_tgt: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Prevent each decoder position from attending to future positions.


        Args:
            tnsr_tgt:     
        
        Return:
            Boolean mask    Shape: (1, 1, tgt_length, tgt_length)
        """
        tgt_seq_len = tnsr_tgt.size(1)

        tnsr_causal_mask = torch.tril(
            torch.ones(
                tgt_seq_len,
                tgt_seq_len,
                dtype=torch.bool,
                device=tnsr_tgt.device
            )
        )

        return tnsr_causal_mask.unsqueeze(0).unsqueeze(0)

        # Parsed out explanation of whats happening with create_cause_mask
        # tnsr_query_positions = torch.arange(tgt_length).unsqueeze(1)
        # tnsr_key_positions = torch.arange(tgt_length).unsqueeze(0)
        # # A Query may read a key when: key_position <= query_position
        # # At the curr Query position, the Query may attend to itself and all eariler positions
        # # BUT NOT to later positions.
        # #       True  == Attention is allowed
        # #       False == Attention is blocked
        # # Shape: (tgt_length, tgt_length)
        # tnsr_attention_is_allowed = tnsr_key_positions <= tnsr_query_positions

        # # Shape: (1, 1, tgt_length, tgt_length)
        # tnsr_causal_mask = tnsr_attention_is_allowed.unsqueeze(0).unsqueeze(0)


    def forward(
        self,
        tnsr_srcTokens_tokenizerIds: torch.Tensor,
        tnsr_intendedDecoderInput_tokenizerIds: torch.Tensor,
        tnsr_src_mask: torch.Tensor=None,
        tnsr_intendedDecoderInput_mask: torch.Tensor=None
    ) -> torch.Tensor: 
        """
        Forward pass through complete transformer

        Args:
            tnsr_srcTokens_tokenizerIds:                 
                            Shape: (batch_size, src_seq_length)
            tnsr_intendedDecoderInput_tokenizerIds:            
                            Intended input for the Decoder      Shape: (batch_size, tgt_seq_length)
                            SERVES TWO DIFFERENT PURPOSES       
                            *** During training  == Teacher Forcing aka getting fed the ground truth
                            *** During Inference == for loop: next token prediction -> append -> feed to decoder 
            src_mask:       
            tgt_mask:   
        Return:
            output:         final output
        """
        # Embed Source
        tnsr_src_embedded = self.positional_encoding(
            self.src_embedding(tnsr_srcTokens_tokenizerIds)
        )
        # Encode Source
        tnsr_encoder_output, tnsr_encoded_weights = self.encoder(tnsr_src_embedded, tnsr_src_mask)

        # Embed target
        tnsr_tgt_embedded = self.positional_encoding(
            self.tgt_embedding(tnsr_intendedDecoderInput_tokenizerIds)
        )
        # Decode target
        tnsr_decoder_output, tnsr_decoded_weights = self.decoder(
            tnsr_tgt_embedded, tnsr_encoder_output, tnsr_intendedDecoderInput_mask, tnsr_src_mask
        )

        # Project to final dimension size == Target Vocabulary size 
        tnsr_output = self.nn_output_projection(tnsr_decoder_output)

        return tnsr_output
