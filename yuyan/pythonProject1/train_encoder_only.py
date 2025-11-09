import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.data_loaders import get_shakespeare_splits, CharDataset
    from src.models import EncoderOnlyLM
    from src.utils import set_seed, plot_curves
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 'train_encoder_only.py' 位于项目根目录 (与 'src' 文件夹同级),")
    print("并且 'src' 文件夹中包含所有必需的模型和数据加载器文件。")
    sys.exit(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

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

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def main(args):
    set_seed(args.seed)

    print("\n" + "=" * 50)
    print(f"--- 启动实验 (Encoder-Only): {args.experiment_name} ---")
    print(f"--- 位置编码 (enable_pe) 是否开启: {args.enable_pe} ---")
    print("=" * 50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    from src.data_loaders import CharDataset, get_shakespeare_splits
    (train_text, val_text, _test_text,
     vocab_size, char_to_idx, idx_to_char) = get_shakespeare_splits()

    train_data = CharDataset(train_text, args.context_size, char_to_idx)
    val_data = CharDataset(val_text, args.context_size, char_to_idx)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Vocab Size: {vocab_size}")
    print(f"Train dataset size: {len(train_data)} | Val dataset size: {len(val_data)}")

    model = EncoderOnlyLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.n_layers,
        num_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.context_size,
        dropout=args.dropout,
        enable_pe=args.enable_pe
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    criterion = nn.CrossEntropyLoss()

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
            model_save_path = f"models/best_encoder_only_{args.experiment_name}_seed{args.seed}.pt"
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
        epochs=10,
        batch_size=32,
        lr=3e-4,
        grad_clip=1.0,

        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_len=64,
        context_size=64,
        dropout=0.1,


        experiment_name='baseline_enc',
        enable_pe=True

    )

    main(args)