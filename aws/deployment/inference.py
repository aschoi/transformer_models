
from model_loader import ModelBundle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from datasets import load_dataset
import sacrebleu
# from dataclasses import dataclass

import time
import re
from pathlib import Path
from collections.abc import Iterable
from transformer.model import Transformer
import json


def greedy_decode_batch(
    model: nn.Module,
    source: torch.Tensor,
    src_pad_id: int,
    tgt_pad_id: int,
    tgt_bos_id: int,
    tgt_eos_id: int,
    max_output_length: int = 100
) -> torch.Tensor:
    """
    Translate a batch of source sequences using greedy decoding.

    Args:
        model:      Encoder-decoder Transformer.
        source:     Source token IDs shaped: [batch_size, source_length]
        src_pad_id: Padding-token ID used by the source tokenizer
        tgt_pad_id: Padding-token ID used by the target tokenizer.
        tgt_bos_id: Beginning-of-sequence ID used by the target tokenizer.
        tgt_eos_id: End-of-sequence ID used by the target tokenizer.
        max_output_length:Maximum generated sequence length, including BOS.

    Returns:
        Generated target IDs shaped: [batch_size, generated_length]
    """

    device = source.device

    if max_output_length < 2:
        raise ValueError("max_output_length must be at least 2 to allow BOS and one generated token.")

    model.eval()

    batch_size = source.size(0)

    # Every target sequence begins with the target-language BOS token.
    generated = torch.full(size=(batch_size, 1), fill_value=tgt_bos_id, dtype=torch.long, device=device)

    # Tracks which sequences have already generated EOS.
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # The source does not change during decoding, so create this once.
    src_mask = model.create_padding_mask(source, src_pad_id)

    with torch.inference_mode():
        # BOS already occupies one position.

        # The REPETITIVE re-feeding of the sequential generation of the model's predictions
        # The generative output gets built, one token at a time. 
        for _ in range(max_output_length - 1):

            # sending generated in as causal_mask parameter merely to use its size. 
            # sending genereated so that when mask is created, it creates a tensor specific to the
            # device being used, instead of having to pass in that info as an argument explicitly
            tgt_causal_mask = model.create_causal_mask(generated)
            tgt_padding_mask = model.create_padding_mask(generated, tgt_pad_id)

            # Both masks use:
            # True  = attention allowed
            # False = attention blocked
            tgt_mask = tgt_causal_mask & tgt_padding_mask
            logits = model(source, generated, src_mask, tgt_mask)

            # Expected logits shape:
            # [batch_size, target_length, target_vocab_size]
            if logits.ndim != 3:
                raise RuntimeError(f"Expected model output with 3 dimensions [batch, target_length, vocabulary_size], but received shape {tuple(logits.shape)}.")

            # Only use the prediction at the newest target position.
            next_token_logits = logits[:, -1, :]

            # Greedy decoding chooses the highest-logit token.
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Sequences that previously produced EOS receive PAD from
            # this point onward while the other sequences continue.
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, tgt_pad_id), next_token)

            generated = torch.cat([generated, next_token], dim=1)

            # Mark newly completed sequences.
            finished |= next_token.squeeze(1).eq(tgt_eos_id)

            if finished.all():
                break

    return generated

def clean_decoded_text(text: str) -> str:
    # Remove spaces before common punctuation.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def translate(
    bundle: ModelBundle,
    text: str,
    max_output_length: int,
) -> str:
    encoded = bundle.src_tokenizer.encode(text)

    source_ids = encoded.ids

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=bundle.device,
    )

    with torch.inference_mode():
        generated = greedy_decode_batch(
            model=bundle.model,
            source=source,
            src_pad_id=bundle.config["src_pad_id"],
            tgt_pad_id=bundle.config["tgt_pad_id"],
            tgt_bos_id=bundle.config["tgt_bos_id"],
            tgt_eos_id=bundle.config["tgt_eos_id"],
            max_output_length=max_output_length,
        )    

    generated_ids = generated[0].detach().cpu().tolist()

    return bundle.tgt_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


