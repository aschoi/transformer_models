import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

from .model import Transformer
from .train import TransformerTrainer
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from collections.abc import Iterable
from pathlib import Path
import json



def train_bpe_hf_tokenizer(
    texts: Iterable[str], 
    vocab_size: int,
    unk_token: str,
    special_tokens: list[str],
    pretokenizer: str,
    minFreq: int
) -> Tokenizer:
    """
    Hugging Face Byte Pair Encoding Tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if pretokenizer == 'whitespace':
        tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=minFreq, special_tokens=special_tokens)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer


def main():

    # collate_fn responsibilities
    # Takes several individual dataset samples and combines them into one batch suitable for the mode
    # a dataset has individual samples
    # the Dataloader gathers batch_size samples
    # collate_fn organizes them into batch tensors

    # Responsiblities:
    # 1) receive raw source-target pairs
    # 2) tokenize source sentences
    # 3) tokenize target sentences
    # 4) add bos and eos
    # 5) convert token IDs to tensors
    # 6) Pad sequences to teh batch's longest sequence
    # 7) Optionally create masks
    # 8) Return the structured batch
    # - optionally, also possible to perform target shift here
    def collate_fn(batch):
        """
        Custom collate function
        
        Batch:  a list of size batch.  [dict{}]
                more precisely, contains whatever my dataset's __getitem__() returns
        Return  a tensor of size batch
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

        return [SRC_BOS_ID, *encoding.ids, SRC_EOS_ID,]


    def encode_target(text: str) -> list[int]:
        encoding = target_tokenizer.encode(text)

        return [TGT_BOS_ID, *encoding.ids, TGT_EOS_ID,]


    # -------- main -------
    print("Running main.")


    # ======== DATA AND TOKENIZER PARAMETERS ======== #
    dataset_name = "bentrevett/multi30k"
    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    unk_token = '[UNK]'
    src_txt = 'en'
    src_vocab_size = 4000
    src_pre_tokenizer = 'whitespace'
    src_tokenizer_min_freq = 2 
    tgt_txt = 'de'
    tgt_vocab_size = 5000
    tgt_pre_tokenizer = 'whitespace'
    tgt_tokenizer_min_freq = 2
    tokenizerDir_savePath = "english2German/checkpoints/tokenizers"
    tokenizer_directory = Path(tokenizerDir_savePath)
    tokenizerSrc_savePath = str(tokenizer_directory / "english_bpe.json")
    tokenizerTgt_savePath = str(tokenizer_directory / "german_bpe.json")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = load_dataset(dataset_name)
    training_dataset = dataset['train']
    validation_dataset = dataset['validation']
    #  test_dataset = dataset['test']

    Tokenizer
    source_tokenizer = train_bpe_hf_tokenizer(
        texts=training_dataset[src_txt], 
        vocab_size=src_vocab_size,
        unk_token=unk_token,
        special_tokens=SPECIAL_TOKENS,
        pretokenizer=src_pre_tokenizer,
        minFreq=src_tokenizer_min_freq
    )

    target_tokenizer = train_bpe_hf_tokenizer(
        texts=training_dataset[tgt_txt], 
        vocab_size=tgt_vocab_size,
        unk_token=unk_token,
        special_tokens=SPECIAL_TOKENS,
        pretokenizer=tgt_pre_tokenizer,
        minFreq=tgt_tokenizer_min_freq
    )


    tokenizer_directory.mkdir(parents=True, exist_ok=True)
    source_tokenizer.save(tokenizerSrc_savePath)
    target_tokenizer.save(tokenizerTgt_savePath)
    
    SRC_PAD_ID = source_tokenizer.token_to_id(SPECIAL_TOKENS[0])
    SRC_UNK_ID = source_tokenizer.token_to_id(SPECIAL_TOKENS[1])
    SRC_BOS_ID = source_tokenizer.token_to_id(SPECIAL_TOKENS[2])
    SRC_EOS_ID = source_tokenizer.token_to_id(SPECIAL_TOKENS[3])

    TGT_PAD_ID = target_tokenizer.token_to_id(SPECIAL_TOKENS[0])
    TGT_UNK_ID = target_tokenizer.token_to_id(SPECIAL_TOKENS[1])
    TGT_BOS_ID = target_tokenizer.token_to_id(SPECIAL_TOKENS[2])
    TGT_EOS_ID = target_tokenizer.token_to_id(SPECIAL_TOKENS[3])

    source_vocab_size = source_tokenizer.get_vocab_size()
    target_vocab_size = target_tokenizer.get_vocab_size()


    # ======== MODEL AND TRAINING PARAMETERS ======== #
    d_model = 128
    num_attn_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    dropout = 0.2
    activate = 'gelu'
    max_seq_len = 5000
    param_init = 'xavier_normal'

    batch_size = 32
    shuffle = True
    cur_step_count = 0
    warmup_steps = 800
    epochs = 2

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    # Create Model
    model = Transformer(
        src_vocab_size=source_vocab_size,
        tgt_vocab_size=target_vocab_size,
        src_pad_id=SRC_PAD_ID,
        tgt_pad_id=TGT_PAD_ID,
        d_model=d_model,
        num_attn_heads=num_attn_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activate,
        max_seq_len=max_seq_len,
        param_init=param_init
    ).to(device)

    # Train Model
    trainer = TransformerTrainer(model, train_loader, validation_loader, TGT_PAD_ID, step_count=cur_step_count, warmup_steps=warmup_steps)

    # Parameters
    all_model_parameters = {
        "dataset_name": dataset_name,
        "SPECIAL_TOKENS": SPECIAL_TOKENS,
        "unk_token": unk_token,
        "src_txt": src_txt,
        "tgt_txt": tgt_txt,
        "src_vocab_size": src_vocab_size,
        "src_pre_tokenizer": src_pre_tokenizer,
        "src_tokenizer_min_freq": src_tokenizer_min_freq,
        "tgt_vocab_size": tgt_vocab_size,
        "tgt_pre_tokenizer": tgt_pre_tokenizer,
        "tgt_tokenizer_min_freq": tgt_tokenizer_min_freq,
        "tokenizerDir_relativeSavePath": tokenizerDir_savePath,
        "tokenizer_directory": str(tokenizer_directory),
        "tokenizerSrc_savePath": tokenizerSrc_savePath,
        "tokenizerTgt_savePath": tokenizerTgt_savePath,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "src_pad_id": SRC_PAD_ID,
        "tgt_pad_id": TGT_PAD_ID,
        "d_model": d_model,
        "num_attn_heads": num_attn_heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "activation": activate,
        "max_seq_len": max_seq_len,
        "param_init": param_init,
        "cur_step_count": cur_step_count,
        "warmup_steps": warmup_steps,
        "epochs": 0,
        "history": []
    }

    start = time.perf_counter()

    print(f"\n-----Training for {epochs} epochs-----")
    for epoch in range(1, epochs+1):
        training_loss, validation_loss = trainer.train_epoch(epoch, SRC_PAD_ID, TGT_PAD_ID, TGT_BOS_ID, TGT_EOS_ID, source_tokenizer, target_tokenizer, device, all_model_parameters)

        print(f"Epoch {epoch}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss}\n")

        all_model_parameters['epochs'] = epoch


    with open("english2German/all_parameters.json", "w") as file:
        json.dump(all_model_parameters, file, indent=4)


if __name__ == "__main__":
    main()


        


