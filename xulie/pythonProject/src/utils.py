import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_causal_mask(seq_len, device):

    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    return mask.unsqueeze(0).unsqueeze(0)


def plot_curves(loss_data: dict, filename: str, title: str):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(12, 6))

    i = 0
    for name, data in loss_data.items():
        plt.plot(data['train'], label=f'{name} (Train)', linestyle='--')
        plt.plot(data['val'], label=f'{name} (Validation)', linestyle='-', linewidth=2)
        i += 1

    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Comparison curves saved to {filename}")