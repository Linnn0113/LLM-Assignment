import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import json  # 用于保存损失

# --- 修复 Windows DLL 和 OMP 错误 ---
import os

os.environ["BUILD_TORCHTEXT"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入所有自定义模块
from seq2seq_dataset import get_seq2seq_dataloaders, PAD_IDX
from seq2seq_model import Seq2SeqTransformer
from utils import set_seed, plot_curves


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        logits = model(src_tokens=src, tgt_tokens=tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Batch {i}/{len(loader)}, Loss: {loss.item():.4f}")

    end_time = time.time()
    avg_loss = total_loss / len(loader)
    print(f"Epoch Training Time: {end_time - start_time:.2f}s")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        logits = model(src_tokens=src, tgt_tokens=tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1)
        )
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def main(args):
    set_seed(args.seed)

    print("\n" + "=" * 50)
    print(f"--- 启动实验: {args.experiment_name} ---")
    print(f"--- 位置编码 (enable_pe) 是否开启: {args.enable_pe} ---")
    print("=" * 50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, src_vocab, tgt_vocab, pad_idx = get_seq2seq_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=2
    )
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        pad_idx=pad_idx,
        dropout=args.dropout,
        enable_pe=args.enable_pe
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        end_time = time.time()

        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f"models/best_seq2seq_{args.experiment_name}_seed{args.seed}.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")

    results = {'train': train_losses, 'val': val_losses}
    loss_save_path = f"results/{args.experiment_name}_losses.json"
    with open(loss_save_path, 'w') as f:
        json.dump(results, f)
    print(f"Loss history saved to {loss_save_path}")
    print("Training complete.")


if __name__ == "__main__":

    args = argparse.Namespace(
        seed=42,
        epochs=13,
        batch_size=32,
        lr=3e-4,
        grad_clip=1.0,
        d_model=256,
        n_heads=8,
        n_layers=3,
        d_ff=1024,
        max_len=64,
        dropout=0.1,

        experiment_name='baseline',
        enable_pe=True
    )

    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    main(args)