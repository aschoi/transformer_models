from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch
from tokenizers import Tokenizer

from transformer.model import Transformer


@dataclass
class ModelBundle:
    model: Transformer
    src_tokenizer: Tokenizer
    tgt_tokenizer: Tokenizer
    config: dict[str, Any]
    device: torch.device


def load_model_bundle(model_dir: str | Path) -> ModelBundle:

    model_path = Path(model_dir)

    with open(model_path / "model_config.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    device = torch.device("cpu")

    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        src_pad_id=config["src_pad_id"],
        tgt_pad_id=config["tgt_pad_id"],
        d_model=config["d_model"],
        num_attn_heads=config["num_attn_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        activation=config["activation"],
    )

    state_dict = torch.load(model_path / "model_state_dict.pt", map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        src_tokenizer=Tokenizer.from_file(str(model_path / "english_bpe.json")),
        tgt_tokenizer=Tokenizer.from_file(str(model_path / "german_bpe.json")),
        config=config,
        device=device
    )

