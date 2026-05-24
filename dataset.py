import torch
import spacy

from datasets import load_dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
UNK_IDX = 0

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

train_data = load_dataset(
    "bentrevett/multi30k",
    split="train"
)

valid_data = load_dataset(
    "bentrevett/multi30k",
    split="validation"
)

def tokenize_de(text):

    return [
        tok.text.lower()
        for tok in spacy_de.tokenizer(text)
    ]

def tokenize_en(text):

    return [
        tok.text.lower()
        for tok in spacy_en.tokenizer(text)
    ]

class Vocabulary:

    def __init__(self):

        self.token_to_idx = {
            "<unk>": UNK_IDX,
            "<pad>": PAD_IDX,
            "<bos>": SOS_IDX,
            "<eos>": EOS_IDX
        }

        self.idx_to_token = {
            UNK_IDX: "<unk>",
            PAD_IDX: "<pad>",
            SOS_IDX: "<bos>",
            EOS_IDX: "<eos>"
        }

    def build(self, counter):

        idx = 4

        for token in counter:

            if token not in self.token_to_idx:

                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token

                idx += 1

    def __getitem__(self, token):

        return self.token_to_idx.get(
            token,
            UNK_IDX
        )

    def lookup_token(self, idx):

        return self.idx_to_token.get(
            idx,
            "<unk>"
        )

    def __len__(self):

        return len(self.token_to_idx)

def build_vocab(data, tokenizer, language):

    counter = Counter()

    for sample in data:

        counter.update(
            tokenizer(
                sample[language]
            )
        )

    vocab = Vocabulary()

    vocab.build(counter)

    return vocab

src_vocab = build_vocab(
    train_data,
    tokenize_de,
    "de"
)

tgt_vocab = build_vocab(
    train_data,
    tokenize_en,
    "en"
)

def text_to_tensor(tokens, vocab_obj):

    ids = [SOS_IDX]

    ids += [
        vocab_obj[token]
        for token in tokens
    ]

    ids.append(EOS_IDX)

    return torch.tensor(ids)

def collate_fn(batch):

    src_batch = []
    tgt_batch = []

    for sample in batch:

        src = text_to_tensor(
            tokenize_de(sample["de"]),
            src_vocab
        )

        tgt = text_to_tensor(
            tokenize_en(sample["en"]),
            tgt_vocab
        )

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = pad_sequence(
        src_batch,
        padding_value=PAD_IDX,
        batch_first=True
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        padding_value=PAD_IDX,
        batch_first=True
    )

    return {
        "src": src_batch,
        "tgt": tgt_batch
    }