import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
from typing import Iterable, List
import spacy
import os
import requests
import sys

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
    print("请确保已在 (transformer_env) 环境中运行: python -m spacy download ...")
    sys.exit(1)


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


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILE_PATH = "data/tiny_shakespeare.txt"
DATA_DIR = "data"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(FILE_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        try:
            proxies = {
                "http": os.environ.get('HTTP_PROXY'),
                "https": os.environ.get('HTTPS_PROXY'),
            }
            response = requests.get(DATA_URL, proxies=proxies)
            response.raise_for_status()
            with open(FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("请检查您的网络连接或 'HTTP_PROXY' / 'HTTPS_PROXY' 环境变量。")
            sys.exit(1)


def get_shakespeare_splits(file_path=FILE_PATH):
    download_data()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        sys.exit(1)

    i = int(len(text) * 0.9)
    train_text, rem_text = text[:i], text[i:]
    i = int(len(rem_text) * 0.5)
    val_text, test_text = rem_text[:i], rem_text[i:]

    chars = sorted(list(set(train_text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return (train_text, val_text, test_text,
            vocab_size, char_to_idx, idx_to_char)


class CharDataset(Dataset):
    def __init__(self, text, context_size, char_to_idx):
        self.text = text
        self.context_size = context_size
        self.char_to_idx = char_to_idx
        self.encoded_text = torch.tensor(
            [self.char_to_idx.get(ch, 0) for ch in self.text], dtype=torch.long
        )

    def __len__(self):
        return len(self.encoded_text) - self.context_size

    def __getitem__(self, idx):
        x = self.encoded_text[idx: idx + self.context_size]
        y = self.encoded_text[idx + 1: idx + 1 + self.context_size]
        return x, y