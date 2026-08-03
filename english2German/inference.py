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
# from dataclasses import dataclass

import time
import re
from pathlib import Path
from collections.abc import Iterable
from .model import Transformer


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

    if max_output_length < 2:
        raise ValueError("max_output_length must be at least 2 to allow BOS and one generated token.")

    model.eval()

    batch_size = source.size(0)

    # Every target sequence begins with the target-language BOS token.
    generated = torch.full(size=(batch_size, 1), fill_value=tgt_bos_id, dtype=torch.long)

    # Tracks which sequences have already generated EOS.
    finished = torch.zeros(batch_size, dtype=torch.bool)

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


def main():

    # ------ functions --------
    def collate_fn(batch):
        """
        Custom collate function
        Args:

        Return:
        """
        src_seqs = []
        tgt_seqs = []

        englishTexts = []
        germanTexts = []

        for sample in batch:

            engSample = sample["en"]
            englishTexts.append(engSample)
            bos_engEncodings_eos = encode_source(engSample)
            bos_engEncodings_eos_tensor = torch.tensor(bos_engEncodings_eos, dtype=torch.long)
            src_seqs.append(bos_engEncodings_eos_tensor)

            germSample = sample["de"]
            germanTexts.append(germSample)
            bos_deEncodings_eos = encode_target(germSample)
            bos_deEncodings_eos_tensor = torch.tensor(bos_deEncodings_eos, dtype=torch.long)
            tgt_seqs.append(bos_deEncodings_eos_tensor)

        # takes the list of seqs (diff lengths) and pads the shorter ones to match the longest sequence. Returns one tensor
        t_bos_srcEncodings_eos_pads_asIds = pad_sequence(src_seqs, batch_first=True, padding_value=SRC_PAD_ID)
        t_bos_tgtEncodings_eos_pads_asIds = pad_sequence(tgt_seqs, batch_first=True, padding_value=TGT_PAD_ID)

        # [BOS, token1, token2, ..., tokenN]
        t_bos_tgtEncodings_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, :-1]

        # [token1, token2, ..., tokenN, EOS]
        t_tgtEncodings_eos_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, 1:]

        return {
            "bos_src_eos": t_bos_srcEncodings_eos_pads_asIds,
            "bos_tgt": t_bos_tgtEncodings_pads_asIds,
            "tgt_eos": t_tgtEncodings_eos_pads_asIds,
            "bos_tgt_eos": t_bos_tgtEncodings_eos_pads_asIds,
            "en": englishTexts,
            "de": germanTexts
        }


    
    def encode_source(text: str) -> list[int]:
        encoding = source_tokenizer.encode(text)

        return [SRC_BOS_ID, *encoding.ids, SRC_EOS_ID]


    def encode_target(text: str) -> list[int]:
        encoding = target_tokenizer.encode(text)

        return [TGT_BOS_ID, *encoding.ids, TGT_EOS_ID]



    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    unk_token = '[UNK]'

    # -------- main --------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tokenizer = Tokenizer.from_file("english2German/checkpoints/tokenizers/english_bpe.json")
    target_tokenizer = Tokenizer.from_file("english2German/checkpoints/tokenizers/german_bpe.json")

    SRC_PAD_ID = source_tokenizer.token_to_id("[PAD]")
    SRC_BOS_ID = source_tokenizer.token_to_id("[BOS]")
    SRC_EOS_ID = source_tokenizer.token_to_id("[EOS]")

    TGT_PAD_ID = target_tokenizer.token_to_id("[PAD]")
    TGT_BOS_ID = target_tokenizer.token_to_id("[BOS]")
    TGT_EOS_ID = target_tokenizer.token_to_id("[EOS]")

    source_vocab_size = source_tokenizer.get_vocab_size()
    target_vocab_size = target_tokenizer.get_vocab_size()


    # Create Model
    model = Transformer(
        src_vocab_size=source_vocab_size,
        tgt_vocab_size=target_vocab_size,
        src_pad_id=SRC_PAD_ID,
        tgt_pad_id=TGT_PAD_ID,
        d_model=256,
        num_attn_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        dropout=0.2,
        activation='gelu',
        max_seq_len=5000,
        param_init='xavier_normal'
    ).to(device)


    parameter_name = next(iter(model.state_dict()))
    before = model.state_dict()[parameter_name].clone()

    checkpoint_path = Path("english2German/checkpoints/english2German_transformer.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # print(f'Loaded Model detals: {model}')
    print("Loaded checkpoint successfully")
    print("Saved epoch:", checkpoint.get("epoch"))
    print("Validation loss:", checkpoint.get("validation_loss"))

    after = model.state_dict()[parameter_name]

    print("Parameter:", parameter_name)
    print("Weights changed:", not torch.equal(before, after))
    print()


    englishPrompt = input("Enter English sentence to be translated to German: ")

    predictions = []
    references = []
    gref = []

    with torch.inference_mode():

        src_seqs = []

        bos_engEncodings_eos = encode_source(englishPrompt)
        bos_engEncodings_eos_tensor = torch.tensor(bos_engEncodings_eos, dtype=torch.long)
        src_seqs.append(bos_engEncodings_eos_tensor)

        # takes the list of seqs (diff lengths) and pads the shorter ones to match the longest sequence. Returns one tensor
        t_bos_srcEncodings_eos_pads_asIds = pad_sequence(src_seqs, batch_first=True, padding_value=SRC_PAD_ID)


        # # [BOS, token1, token2, ..., tokenN]
        # t_bos_tgtEncodings_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, :-1]

        # # [token1, token2, ..., tokenN, EOS]
        # t_tgtEncodings_eos_pads_asIds = t_bos_tgtEncodings_eos_pads_asIds[:, 1:]


        generated = greedy_decode_batch(
            model=model,
            source=t_bos_srcEncodings_eos_pads_asIds,
            src_pad_id=SRC_PAD_ID,
            tgt_pad_id=TGT_PAD_ID,
            tgt_bos_id=TGT_BOS_ID,
            tgt_eos_id=TGT_EOS_ID,
            max_output_length=100,
        )
                
    
        i = 0
        for predicted_ids in generated.tolist():

            predicted_text = target_tokenizer.decode(predicted_ids, skip_special_tokens=True)
            predicted_text = clean_decoded_text(predicted_text)

            predictions.append(predicted_text)
            i += 1


        print("Generated IDs:", generated[-1].tolist())
        print("Translation:", predictions[-1])
        # print(f"German Text: {gref[-1]}")
        print()



if __name__ == "__main__":
    main()
    