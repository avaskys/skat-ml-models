#!/usr/bin/env python3
"""
Train GameEvaluator (dense) or GameEvaluatorTransformer (attention-based) models.

These models evaluate hand+skat+game-type combinations for discard and game selection.

Usage:
    # Train transformer model (recommended)
    python scripts/train_game_eval.py --sgf data/games.sgf --model transformer

    # Train dense model
    python scripts/train_game_eval.py --sgf data/games.sgf --model dense

    # With torch.compile (CUDA only)
    python scripts/train_game_eval.py --sgf data/games.sgf --model transformer --compile
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from skat_ml.constants import GAME_EVALUATOR_INPUT_DIM, GAME_TYPES
from skat_ml.datasets import StreamingGameEvalDataset, game_eval_collate_fn
from skat_ml.features import extract_game_eval_features
from skat_ml.models import BinaryFocalLoss, GameEvaluator, GameEvaluatorTransformer
from skat_ml.training_utils import (
    add_common_args,
    count_lines,
    create_cosine_scheduler,
    create_warmup_scheduler,
    get_checkpoint_config,
    get_device,
    load_checkpoint,
    maybe_compile_model,
    restore_training_config,
    resume_from_checkpoint,
    save_best_model,
    save_checkpoint,
    save_config,
    setup_mixed_precision,
    setup_run_directory,
)


def train_model(args, defaults: dict = None):
    device = get_device()

    # Setup run directory and check for existing checkpoint FIRST
    output_dir = Path(args.output)
    run_dir, run_name = setup_run_directory(output_dir, args.run_name, f"game_eval_{args.model}")
    latest_path = run_dir / "game_eval_latest.pt"

    # Check for existing checkpoint to restore config (before creating DataLoaders)
    checkpoint = load_checkpoint(latest_path, device)
    if checkpoint:
        saved_config = get_checkpoint_config(checkpoint)
        restore_training_config(args, saved_config, defaults or {})

    # Estimate dataset size (after config restore so batch_size is correct)
    print("Counting games in SGF file...")
    total_games = count_lines(Path(args.sgf))
    est_samples = int(total_games * 0.9)
    est_batches = est_samples // args.batch_size
    print(f"Total games: {total_games:,}. Estimated training batches: {est_batches:,}")

    # Setup datasets (after config restore so batch_size is correct)
    train_dataset = StreamingGameEvalDataset(Path(args.sgf), split="train")
    val_dataset = StreamingGameEvalDataset(Path(args.sgf), split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=game_eval_collate_fn,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        collate_fn=game_eval_collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Initialize model
    if args.model == "dense":
        model = GameEvaluator(
            input_dim=GAME_EVALUATOR_INPUT_DIM,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        ).to(device)
    else:
        model = GameEvaluatorTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optionally compile
    model = maybe_compile_model(model, args.compile, device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler: cosine for dense, warmup for transformer
    if args.model == "dense":
        scheduler = create_cosine_scheduler(optimizer, args.epochs)
        scheduler_steps_per_epoch = True
    else:
        scheduler = create_warmup_scheduler(optimizer, warmup_steps=1000)
        scheduler_steps_per_epoch = False

    # Loss function
    if args.focal:
        criterion = BinaryFocalLoss(gamma=args.gamma, reduction="mean")
        print(f"Using BinaryFocalLoss with gamma={args.gamma}")
    else:
        criterion = nn.BCELoss()
        print("Using BCELoss")

    # Mixed precision
    use_amp, scaler = setup_mixed_precision(device, args.amp)

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config
    config = {
        "model_type": args.model,
        "d_model": args.d_model if args.model == "transformer" else None,
        "nhead": args.nhead if args.model == "transformer" else None,
        "num_layers": args.num_layers if args.model == "transformer" else None,
        "hidden_dim": args.hidden_dim if args.model == "dense" else None,
        "num_blocks": args.num_blocks if args.model == "dense" else None,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sgf": args.sgf,
        "focal": args.focal,
        "gamma": args.gamma if args.focal else None,
    }
    save_config(run_dir / "config.json", config)

    # TensorBoard
    writer = SummaryWriter(log_dir=run_dir)

    # Training state
    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    # Resume if checkpoint exists
    if checkpoint:
        start_epoch, best_val_loss, global_step = resume_from_checkpoint(
            checkpoint, model, optimizer, scheduler, latest_path
        )

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model: {args.model}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", total=est_batches)
        batch_idx = -1

        for batch_idx, batch in enumerate(pbar):
            hand_cards = batch["hand_cards"].to(device)
            skat_cards = batch["skat_cards"].to(device)
            skat_len = batch["skat_len"].to(device)
            game_type = batch["game_type"].to(device)
            position = batch["position"].to(device)
            is_hand = batch["is_hand"].to(device)
            bid = batch["bid"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if args.model == "transformer":
                    win_prob = model(hand_cards, skat_cards, skat_len, game_type, position, is_hand, bid)
                else:
                    # Dense model needs feature extraction
                    batch_size = hand_cards.size(0)
                    features = torch.zeros(batch_size, GAME_EVALUATOR_INPUT_DIM, device=device)
                    for i in range(batch_size):
                        cards = hand_cards[i].cpu().numpy().tolist()
                        pos = position[i].item()
                        gt = GAME_TYPES[game_type[i].item()]
                        is_hg = bool(is_hand[i].item())
                        bid_val = int(bid[i].item() * 264)  # Denormalize
                        skat = [c for c in skat_cards[i].cpu().numpy().tolist() if c < 32]
                        feat = extract_game_eval_features(cards, pos, gt, is_hg, bid_val, skat)
                        features[i] = torch.from_numpy(feat)
                    win_prob = model(features)

                loss = criterion(win_prob, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if not scheduler_steps_per_epoch:
                scheduler.step()

            train_loss += loss.item()
            predictions = (win_prob >= 0.5).float()
            train_correct += (predictions == label).sum().item()
            train_total += label.size(0)
            global_step += 1

            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{train_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100 * train_correct / train_total:.2f}%",
                    "lr": f"{current_lr:.6f}",
                })

            if global_step % 500 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/Accuracy", train_correct / train_total, global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

        pbar.close()
        avg_train_loss = train_loss / max(batch_idx + 1, 1)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation
        print(f"Validating Epoch {epoch}...")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                hand_cards = batch["hand_cards"].to(device)
                skat_cards = batch["skat_cards"].to(device)
                skat_len = batch["skat_len"].to(device)
                game_type = batch["game_type"].to(device)
                position = batch["position"].to(device)
                is_hand = batch["is_hand"].to(device)
                bid = batch["bid"].to(device)
                label = batch["label"].to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if args.model == "transformer":
                        win_prob = model(hand_cards, skat_cards, skat_len, game_type, position, is_hand, bid)
                    else:
                        batch_size = hand_cards.size(0)
                        features = torch.zeros(batch_size, GAME_EVALUATOR_INPUT_DIM, device=device)
                        for i in range(batch_size):
                            cards = hand_cards[i].cpu().numpy().tolist()
                            pos = position[i].item()
                            gt = GAME_TYPES[game_type[i].item()]
                            is_hg = bool(is_hand[i].item())
                            bid_val = int(bid[i].item() * 264)
                            skat = [c for c in skat_cards[i].cpu().numpy().tolist() if c < 32]
                            feat = extract_game_eval_features(cards, pos, gt, is_hg, bid_val, skat)
                            features[i] = torch.from_numpy(feat)
                        win_prob = model(features)

                    loss = criterion(win_prob, label)

                val_loss += loss.item()
                predictions = (win_prob >= 0.5).float()
                val_correct += (predictions == label).sum().item()
                val_total += label.size(0)
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {100*val_acc:.2f}%")

        writer.add_scalars("Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        if scheduler_steps_per_epoch:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  -> New Best Model! ({best_val_loss:.4f})")
            save_best_model(run_dir / "game_eval_best.pt", model, config)

        save_checkpoint(
            run_dir / "game_eval_latest.pt",
            model, optimizer, scheduler, epoch, best_val_loss, global_step, config
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GameEvaluator models")

    # Model selection
    parser.add_argument("--model", type=str, choices=["dense", "transformer"], default="transformer", help="Model type")

    # Data
    parser.add_argument("--sgf", type=str, required=True, help="Path to SGF file")

    # Transformer architecture
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension (transformer)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (transformer)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Dense architecture
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension (dense)")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual blocks (dense)")

    # Loss function
    parser.add_argument("--focal", action="store_true", help="Use BinaryFocalLoss instead of BCELoss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")

    # Common args
    add_common_args(parser)

    # Get defaults before parsing (to detect explicit overrides during resume)
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args()
    train_model(args, defaults)


if __name__ == "__main__":
    main()
