#!/usr/bin/env python3
"""
Train CardPlayTransformer model for trick-by-trick card selection.

This model uses attention to process game state as a sequence of moves.

Usage:
    python scripts/train_card_play.py --sgf data/games.sgf
    python scripts/train_card_play.py --sgf data/games.sgf --run-name my_run --epochs 10
    python scripts/train_card_play.py --sgf data/games.sgf --compile  # Faster on CUDA
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from skat_ml.constants import EST_CARD_PLAY_SAMPLES_PER_GAME
from skat_ml.datasets import FastCardPlayDataset, card_play_collate_fn
from skat_ml.models import CardPlayTransformer, FocalLoss
from skat_ml.training_utils import (
    add_common_args,
    count_lines,
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
    run_dir, run_name = setup_run_directory(output_dir, args.run_name, "card_play_transformer")
    latest_path = run_dir / "card_play_latest.pt"

    # Check for existing checkpoint to restore config (before creating DataLoaders)
    checkpoint = load_checkpoint(latest_path, device)
    if checkpoint:
        saved_config = get_checkpoint_config(checkpoint)
        restore_training_config(args, saved_config, defaults or {})

    # Estimate dataset size (after config restore so batch_size is correct)
    print("Counting games in SGF file...")
    total_games = count_lines(Path(args.sgf))
    est_samples = total_games * 0.9 * EST_CARD_PLAY_SAMPLES_PER_GAME  # 90% train split
    est_batches = int(est_samples / args.batch_size)
    print(f"Total games: {total_games:,}. Estimated training batches: {est_batches:,}")

    # Setup datasets (after config restore so batch_size is correct)
    train_dataset = FastCardPlayDataset(Path(args.sgf), winner_only=True, split="train")
    val_dataset = FastCardPlayDataset(Path(args.sgf), winner_only=True, split="val", shuffle_buffer=0)  # No shuffle needed for val

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=card_play_collate_fn,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=card_play_collate_fn,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Initialize model
    model = CardPlayTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and scheduler (before compile, so optimizer sees raw params)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = create_warmup_scheduler(optimizer, warmup_steps=1000)

    # Loss and mixed precision
    criterion = FocalLoss(gamma=args.gamma, reduction="mean")
    print(f"Using FocalLoss with gamma={args.gamma}")
    use_amp, scaler = setup_mixed_precision(device, args.amp)

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "model_type": "transformer",
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "sgf": args.sgf,
    }
    save_config(run_dir / "config.json", config)

    # TensorBoard
    writer = SummaryWriter(log_dir=run_dir)
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'runs'}")

    # Training state
    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    # Resume if checkpoint exists
    if checkpoint:
        start_epoch, best_val_loss, global_step = resume_from_checkpoint(
            checkpoint, model, optimizer, scheduler, latest_path, new_lr=args.lr
        )

    # Compile after loading checkpoint (compiled model has different state_dict keys)
    model = maybe_compile_model(model, args.compile, device)

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_total_all = 0
        train_correct_all = 0
        steps = 0

        pbar = tqdm(train_loader, total=est_batches, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in pbar:
            num_legal = batch["num_legal"].to(device)

            # Track all samples (forced plays are 100% correct with 0 loss)
            batch_size = num_legal.size(0)
            num_forced = (num_legal == 1).sum().item()
            train_total_all += batch_size
            train_correct_all += num_forced  # Forced plays always correct

            # Filter to multi-choice samples only (skip forced plays)
            mask = num_legal > 1
            if mask.sum() == 0:
                continue  # Skip batch if all forced plays

            game_type = batch["game_type"].to(device)[mask]
            declarer = batch["declarer"].to(device)[mask]
            is_ouvert = batch["is_ouvert"].to(device)[mask]
            hand = batch["hand"].to(device)[mask]
            hand_len = batch["hand_len"].to(device)[mask]
            ouvert_hand = batch["ouvert_hand"].to(device)[mask]
            ouvert_hand_len = batch["ouvert_hand_len"].to(device)[mask]
            history = batch["history"].to(device)[mask]
            history_len = batch["history_len"].to(device)[mask]
            trick = batch["trick"].to(device)[mask]
            trick_len = batch["trick_len"].to(device)[mask]
            legal_mask = batch["legal_mask"].to(device)[mask]
            targets = batch["target"].to(device)[mask]

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    game_type=game_type,
                    declarer=declarer,
                    is_ouvert=is_ouvert,
                    hand=hand,
                    hand_len=hand_len,
                    ouvert_hand=ouvert_hand,
                    ouvert_hand_len=ouvert_hand_len,
                    history=history,
                    history_len=history_len,
                    trick=trick,
                    trick_len=trick_len,
                    legal_mask=legal_mask,
                )
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            _, predicted = outputs.max(1)
            num_correct = predicted.eq(targets).sum().item()
            num_filtered = targets.size(0)
            train_loss_sum += loss.item() * num_filtered  # Sum of losses (criterion uses mean)
            train_total += num_filtered
            train_correct += num_correct
            train_correct_all += num_correct  # Add filtered correct to all
            steps += 1

            if steps % 100 == 0:
                train_loss = train_loss_sum / train_total if train_total > 0 else 0
                train_loss_all = train_loss_sum / train_total_all if train_total_all > 0 else 0
                acc = 100.0 * train_correct / train_total if train_total > 0 else 0
                acc_all = 100.0 * train_correct_all / train_total_all if train_total_all > 0 else 0
                pbar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "loss_all": f"{train_loss_all:.4f}",
                    "acc": f"{acc:.2f}%",
                    "acc_all": f"{acc_all:.2f}%",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                })

            if steps % 500 == 0:
                writer.add_scalar("Live/loss", train_loss_sum / train_total, global_step)
                writer.add_scalar("Live/loss_all", train_loss_sum / train_total_all, global_step)
                writer.add_scalar("Live/acc", 100.0 * train_correct / train_total, global_step)
                writer.add_scalar("Live/acc_all", 100.0 * train_correct_all / train_total_all, global_step)
                writer.add_scalar("Live/lr", scheduler.get_last_lr()[0], global_step)

            if args.max_steps > 0 and steps >= args.max_steps:
                break

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_total_all = 0
        val_correct_all = 0
        val_steps = 0

        print(f"\nValidating Epoch {epoch}...")

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                num_legal = batch["num_legal"].to(device)

                # Track all samples (forced plays are 100% correct with 0 loss)
                batch_size = num_legal.size(0)
                num_forced = (num_legal == 1).sum().item()
                val_total_all += batch_size
                val_correct_all += num_forced  # Forced plays always correct

                # Filter to multi-choice samples only
                mask = num_legal > 1
                if mask.sum() == 0:
                    continue  # Skip batch if all forced plays

                game_type = batch["game_type"].to(device)[mask]
                declarer = batch["declarer"].to(device)[mask]
                is_ouvert = batch["is_ouvert"].to(device)[mask]
                hand = batch["hand"].to(device)[mask]
                hand_len = batch["hand_len"].to(device)[mask]
                ouvert_hand = batch["ouvert_hand"].to(device)[mask]
                ouvert_hand_len = batch["ouvert_hand_len"].to(device)[mask]
                history = batch["history"].to(device)[mask]
                history_len = batch["history_len"].to(device)[mask]
                trick = batch["trick"].to(device)[mask]
                trick_len = batch["trick_len"].to(device)[mask]
                legal_mask = batch["legal_mask"].to(device)[mask]
                targets = batch["target"].to(device)[mask]

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(
                        game_type=game_type,
                        declarer=declarer,
                        is_ouvert=is_ouvert,
                        hand=hand,
                        hand_len=hand_len,
                        ouvert_hand=ouvert_hand,
                        ouvert_hand_len=ouvert_hand_len,
                        history=history,
                        history_len=history_len,
                        trick=trick,
                        trick_len=trick_len,
                        legal_mask=legal_mask,
                    )
                    loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                num_correct = predicted.eq(targets).sum().item()
                num_filtered = targets.size(0)

                val_loss_sum += loss.item() * num_filtered
                val_total += num_filtered
                val_correct += num_correct
                val_correct_all += num_correct

                val_steps += 1

                if val_steps % 50 == 0:
                    acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                    acc_all = 100.0 * val_correct_all / val_total_all if val_total_all > 0 else 0
                    val_pbar.set_postfix({
                        "acc": f"{acc:.2f}%",
                        "acc_all": f"{acc_all:.2f}%",
                    })

                if args.max_steps > 0 and val_steps >= (args.max_steps // 10):
                    break

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Filtered metrics (primary - multi-choice only)
        avg_val_loss = val_loss_sum / val_total if val_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

        # All samples metrics (for comparison to old runs)
        avg_val_loss_all = val_loss_sum / val_total_all if val_total_all > 0 else 0
        val_acc_all = 100.0 * val_correct_all / val_total_all if val_total_all > 0 else 0

        avg_train_loss = train_loss_sum / train_total if train_total > 0 else 0
        avg_train_loss_all = train_loss_sum / train_total_all if train_total_all > 0 else 0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        train_acc_all = 100.0 * train_correct_all / train_total_all if train_total_all > 0 else 0

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% (filtered)")
        print(f"  Train All:  {avg_train_loss_all:.4f} | Train Acc: {train_acc_all:.2f}% (all samples)")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}% (filtered)")
        print(f"  Val All:    {avg_val_loss_all:.4f} | Val Acc:   {val_acc_all:.2f}% (all samples)")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_all", avg_train_loss_all, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_all", avg_val_loss_all, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/train_all", train_acc_all, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Accuracy/val_all", val_acc_all, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        if avg_val_loss < best_val_loss:
            print(f"  -> New Best Model! ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
            save_best_model(run_dir / "card_play_best.pt", model, config)

        save_checkpoint(
            run_dir / "card_play_latest.pt",
            model, optimizer, scheduler, epoch, best_val_loss, global_step, config
        )

    writer.close()
    print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"Models and logs saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train CardPlayTransformer")

    # Data
    parser.add_argument("--sgf", type=str, required=True, help="Path to SGF data")

    # Architecture
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Loss
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")

    # Training limits
    parser.add_argument("--max-steps", type=int, default=0, help="Limit steps per epoch (0 = full)")

    # Common args
    add_common_args(parser)

    # Get defaults before parsing (to detect explicit overrides during resume)
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args()
    train_model(args, defaults)


if __name__ == "__main__":
    main()
