import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from collections.abc import Iterable
from pathlib import Path
import time


def train_bpe_hf_tokenizer(
    texts: Iterable[str], 
    vocab_size: int,
    unk_token: str,
    special_tokens: list[str]
) -> Tokenizer:
    """
    Hugging Face Byte Pair Encoding Tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer


def main():

    '''
    Takes several individual dataset samples and combines them into one batch suitable for the mode
    a dataset has individual samples
    the Dataloader gathers batch_size samples
    collate_fn organizes them into batch tensors

    Responsiblities:
    1) receive raw source-target pairs
    2) tokenize source sentences
    3) tokenize target sentences
    4) add bos and eos
    5) convert token IDs to tensors
    6) Pad sequences to teh batch's longest sequence
    7) Optionally create masks
    8) Return the structured batch
    - optionally, also possible to perform target shift here
    '''
    def collate_fn(batch):
        """
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


    def encode_source(
        # source_tokenizer: Tokenizer, 
        text: str, 
        # src_bos_id: int, 
        # src_eos_id: int
    ) -> list[int]:
        
        encoding = source_tokenizer.encode(text)
        # src_pad_id = source_tokenizer.token_to_id("[PAD]")

        return [SRC_BOS_ID, *encoding.ids, SRC_EOS_ID]


    def encode_target(
        # target_tokenizer: Tokenizer, 
        text: str, 
        # tgt_bos_id: int, 
        # tgt_eos_id: int
    ) -> list[int]:
        
        encoding = target_tokenizer.encode(text)
        # tgt_pad_id = target_tokenizer.token_to_id("[PAD]")

        return [TGT_BOS_ID, *encoding.ids, TGT_EOS_ID]


    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    unk_token = '[UNK]'


    print("A Tokenizer Demo")

    dataset = load_dataset("bentrevett/multi30k")
    training_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    source_tokenizer = train_bpe_hf_tokenizer(
        texts=training_dataset["en"], 
        vocab_size=4000,
        unk_token=unk_token,
        special_tokens=SPECIAL_TOKENS
    )

    target_tokenizer = train_bpe_hf_tokenizer(
        texts=training_dataset["de"], 
        vocab_size=5000,
        unk_token=unk_token,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer_directory = Path("english2German/tokenizerDemo")
    tokenizer_directory.mkdir(parents=True, exist_ok=True)

    source_tokenizer.save(str(tokenizer_directory / "english_bpe.json"))
    target_tokenizer.save(str(tokenizer_directory / "german_bpe.json"))

    SRC_PAD_ID = source_tokenizer.token_to_id("[PAD]")
    SRC_UNK_ID = source_tokenizer.token_to_id("[UNK]")
    SRC_BOS_ID = source_tokenizer.token_to_id("[BOS]")
    SRC_EOS_ID = source_tokenizer.token_to_id("[EOS]")

    TGT_PAD_ID = target_tokenizer.token_to_id("[PAD]")
    TGT_UNK_ID = target_tokenizer.token_to_id("[UNK]")
    TGT_BOS_ID = target_tokenizer.token_to_id("[BOS]")
    TGT_EOS_ID = target_tokenizer.token_to_id("[EOS]")

    source_vocab_size = source_tokenizer.get_vocab_size()
    target_vocab_size = target_tokenizer.get_vocab_size()

    train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset,batch_size=64,shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    for iBatch, batch in enumerate(train_loader):

        bos_src_eos_batch = batch['bos_src_eos']
        bos_tgt_batch = batch['bos_tgt']
        tgt_eos_batch = batch['tgt_eos']
        bos_tgt_eos_batch = batch['bos_tgt_eos']
        english_text_batch = batch['en']
        german_text_batch = batch['de']
        
        for iSample in range(len(english_text_batch)):
            # for i in bos_src_eos.tolist():
            src_text = source_tokenizer.decode(bos_src_eos_batch[iSample].tolist(), skip_special_tokens=True)
            btgt = target_tokenizer.decode(bos_tgt_batch[iSample].tolist(), skip_special_tokens=True)
            tgte = target_tokenizer.decode(tgt_eos_batch[iSample].tolist(), skip_special_tokens=True)
            tgt_text = target_tokenizer.decode(bos_tgt_eos_batch[iSample].tolist(), skip_special_tokens=True)

            print(src_text)
            print(btgt)
            print(tgte)
            print(tgt_text)
            print(english_text_batch[iSample])
            print(german_text_batch[iSample])
            print()

        if iBatch >= 10:
            break

    

if __name__ == '__main__':
    main()