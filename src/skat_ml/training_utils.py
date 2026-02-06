"""
Shared utilities for model training scripts.

This module provides common functionality used across all training scripts
to reduce code duplication and ensure consistent behavior.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim


def get_device() -> torch.device:
    """Select the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def count_lines(path: Path) -> int:
    """Fast line count for estimating dataset size."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove torch.compile() prefix from checkpoint keys.

    When a model is saved after torch.compile(), the state dict keys
    are prefixed with '_orig_mod.'. This strips that prefix for loading
    into a non-compiled model.
    """
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  Stripping _orig_mod. prefix from compiled checkpoint")
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def setup_run_directory(
    output_dir: Path,
    run_name: str | None,
    default_prefix: str,
) -> tuple[Path, str]:
    """Create and return the run directory path.

    Args:
        output_dir: Base output directory (e.g., models/)
        run_name: Optional custom run name
        default_prefix: Prefix for auto-generated names (e.g., 'bidding_transformer')

    Returns:
        Tuple of (run_dir, run_name)
    """
    if run_name:
        name = run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{default_prefix}_{timestamp}"

    run_dir = output_dir / "runs" / name
    return run_dir, name


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict | None:
    """Load a checkpoint if it exists.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to

    Returns:
        Checkpoint dict if file exists, None otherwise
    """
    if not checkpoint_path.exists():
        return None

    print(f"Found existing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def get_checkpoint_config(checkpoint: dict) -> dict:
    """Extract config from a checkpoint."""
    return checkpoint.get("config", {})


def restore_training_config(args, saved_config: dict, defaults: dict) -> None:
    """
    Restore training hyperparameters from checkpoint config.

    Only restores values that weren't explicitly set on command line
    (i.e., values that still match their parser defaults).

    Args:
        args: Parsed command-line arguments
        saved_config: Config dict from checkpoint
        defaults: Dict of parser default values (to detect explicit overrides)
    """
    restored = []
    overridden = []

    def maybe_restore(key: str, attr_name: str = None):
        """Restore key from saved_config if not explicitly overridden."""
        attr = attr_name or key.replace("-", "_")
        if key not in saved_config or saved_config[key] is None:
            return

        current_val = getattr(args, attr, None)
        default_val = defaults.get(attr)
        saved_val = saved_config[key]

        if current_val == default_val:
            # User didn't override, restore from checkpoint
            setattr(args, attr, saved_val)
            restored.append(f"{key}={saved_val}")
        elif current_val != saved_val:
            # User explicitly set a different value
            overridden.append(f"{key}={current_val} (checkpoint had {saved_val})")

    # Architecture params
    for key in ["d_model", "nhead", "num_layers", "hidden_dim", "num_blocks", "dropout"]:
        maybe_restore(key)

    # Training params
    maybe_restore("batch_size")
    maybe_restore("lr")
    maybe_restore("gamma")

    # Boolean flags need special handling
    if "focal" in saved_config:
        if not args.focal and saved_config["focal"]:
            # User didn't set --focal but checkpoint used it
            args.focal = True
            restored.append("focal=True")
        elif args.focal and not saved_config["focal"]:
            overridden.append("focal=True (checkpoint had False)")

    # For bidding script
    if "use_weights" in saved_config:
        checkpoint_no_weights = not saved_config["use_weights"]
        if not args.no_weights and checkpoint_no_weights:
            args.no_weights = True
            restored.append("no_weights=True")

    if restored:
        print(f"  Restored from checkpoint: {', '.join(restored)}")
    if overridden:
        print(f"  Overridden by command line: {', '.join(overridden)}")


def load_model_state(model: torch.nn.Module, checkpoint: dict) -> None:
    """Load model weights from checkpoint, handling compiled models."""
    state_dict = strip_compiled_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)


def load_optimizer_state(optimizer: optim.Optimizer, checkpoint: dict) -> None:
    """Load optimizer state from checkpoint."""
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_scheduler_state(
    scheduler: optim.lr_scheduler.LRScheduler,
    checkpoint: dict,
) -> bool:
    """Load scheduler state from checkpoint.

    Returns:
        True if loaded successfully, False if failed
    """
    try:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return True
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not load scheduler state ({e}), using fresh scheduler")
        return False


def resume_from_checkpoint(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    checkpoint_path: Path,
    new_lr: float | None = None,
) -> tuple[int, float, int]:
    """Resume training state from a checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to checkpoint (for logging)
        new_lr: If provided, override the optimizer's learning rate

    Returns:
        Tuple of (start_epoch, best_val_loss, global_step)
    """
    print(f"Resuming from {checkpoint_path}")

    load_model_state(model, checkpoint)
    load_optimizer_state(optimizer, checkpoint)
    load_scheduler_state(scheduler, checkpoint)

    # Check if LR is being overridden
    old_lr = optimizer.param_groups[0]["lr"]
    lr_changed = new_lr is not None and abs(old_lr - new_lr) > 1e-9

    if lr_changed:
        # Override LR in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr
        # Update scheduler base_lrs so get_last_lr() returns correct value
        scheduler.base_lrs = [new_lr for _ in scheduler.base_lrs]
        print(f"  LR overridden: {old_lr} -> {new_lr}")

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    global_step = checkpoint.get("global_step", 0)

    print(f"  Resumed at epoch {start_epoch}, global_step {global_step}, best_val_loss {best_val_loss:.4f}")

    return start_epoch, best_val_loss, global_step


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    global_step: int,
    config: dict,
) -> None:
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "config": config,
    }, path)


def save_best_model(
    path: Path,
    model: torch.nn.Module,
    config: dict,
) -> None:
    """Save just the model weights and config (for inference)."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, path)


def save_config(path: Path, config: dict) -> None:
    """Save config to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def maybe_compile_model(
    model: torch.nn.Module,
    compile_enabled: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Optionally compile the model with torch.compile().

    Args:
        model: The model to potentially compile
        compile_enabled: Whether --compile flag was passed
        device: The device (compile only works well on CUDA)

    Returns:
        The model (compiled or not)
    """
    if not compile_enabled:
        return model

    if device.type != "cuda":
        print("Warning: --compile is most effective on CUDA, skipping on", device.type)
        return model

    print("Compiling model with torch.compile()...")
    return torch.compile(model)


def create_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int = 1000,
) -> optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_cosine_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
) -> optim.lr_scheduler.CosineAnnealingLR:
    """Create a cosine annealing learning rate scheduler."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def setup_mixed_precision(
    device: torch.device,
    amp_enabled: bool,
) -> tuple[bool, torch.amp.GradScaler]:
    """Setup automatic mixed precision training.

    Args:
        device: Training device
        amp_enabled: Whether --amp flag was passed

    Returns:
        Tuple of (use_amp, scaler)
    """
    use_amp = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if use_amp:
        print("Using automatic mixed precision (AMP)")

    return use_amp, scaler


def add_common_args(parser) -> None:
    """Add common training arguments to an argument parser."""
    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Output
    parser.add_argument("--output", type=str, default="models/", help="Output directory")
    parser.add_argument("--run-name", type=str, default="", help="Custom run name")

    # Optimizations
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() (CUDA only)")
