import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm  # 用于构建词表的进度条
from typing import Iterable, List
import spacy
import os

try:
    import datasets
except ImportError:
    print("Error: Hugging Face 'datasets' 库未安装。")
    print("请在 (transformer_env) 环境中运行: pip install datasets --proxy http://127.0.0.1:7897")
    exit(1)

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_IDX = SPECIAL_TOKENS.index(PAD_TOKEN)

try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
except OSError:
    print("Error: spaCy models 'en_core_web_sm' and/or 'de_core_news_sm' not found.")
    print("请运行: python -m spacy download en_core_web_sm --proxy http://127.0.0.1:7897")
    print("和: python -m spacy download de_core_news_sm --proxy http://127.0.0.1:7897")
    exit(1)


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


class Vocab:
    def __init__(self, token_to_idx, idx_to_token, unk_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.token_to_idx)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]


def build_native_vocab(data_iter: Iterable, tokenizer_func, lang: str, min_freq: int) -> Vocab:
    counter = Counter()

    for data_sample in tqdm(data_iter, desc=f"Building {lang} Vocab"):
        tokens = tokenizer_func(data_sample['translation'][lang])
        counter.update(tokens)

    tokens_and_freq = [
        (token, freq) for token, freq in counter.items() if freq >= min_freq
    ]

    token_to_idx = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    idx_to_token = {i: token for i, token in enumerate(SPECIAL_TOKENS)}
    next_idx = len(SPECIAL_TOKENS)

    for token, _ in sorted(tokens_and_freq, key=lambda x: x[1], reverse=True):
        if token not in token_to_idx:
            token_to_idx[token] = next_idx
            idx_to_token[next_idx] = token
            next_idx += 1

    return Vocab(token_to_idx, idx_to_token, unk_idx=SPECIAL_TOKENS.index(UNK_TOKEN))


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = sample['translation']['en']
        tgt_text = sample['translation']['de']

        src_tokens = [BOS_TOKEN] + self.src_tokenizer(src_text) + [EOS_TOKEN]
        tgt_tokens = [BOS_TOKEN] + self.tgt_tokenizer(tgt_text) + [EOS_TOKEN]

        src_indices = self.src_vocab(src_tokens)
        tgt_indices = self.tgt_vocab(tgt_tokens)

        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch, pad_idx=PAD_IDX):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)

    return src_batch, tgt_batch


def get_seq2seq_dataloaders(batch_size, max_len=64, min_freq=2, root_dir=".data"):
    print("Loading IWSLT2017 dataset using Hugging Face 'datasets'...")
    try:
        dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-de", cache_dir=root_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Hugging Face 'datasets' 加载失败: {e}. 请确保代理设置正确.")
        exit(1)

    full_train_iter = dataset['train']
    val_iter = dataset['validation']

    SUBSET_SIZE = 50000
    print(f"Sampling first {SUBSET_SIZE} examples for training...")
    train_data_subset = list(full_train_iter.select(range(SUBSET_SIZE)))

    print("Building vocabularies...")
    src_vocab = build_native_vocab(train_data_subset, tokenize_en, lang='en', min_freq=min_freq)
    tgt_vocab = build_native_vocab(train_data_subset, tokenize_de, lang='de', min_freq=min_freq)

    train_dataset = TranslationDataset(train_data_subset, src_vocab, tgt_vocab, tokenize_en, tokenize_de, max_len)
    val_dataset = TranslationDataset(val_iter, src_vocab, tgt_vocab, tokenize_en, tokenize_de, max_len)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX)
    )

    print(f"Source (EN) Vocab Size: {len(src_vocab)}")
    print(f"Target (DE) Vocab Size: {len(tgt_vocab)}")
    print(f"Padding Index: {PAD_IDX}")

    return train_dataloader, val_dataloader, src_vocab, tgt_vocab, PAD_IDX