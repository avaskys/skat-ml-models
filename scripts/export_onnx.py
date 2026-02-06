#!/usr/bin/env python3
"""
Export trained models to ONNX for Java inference.

Supports all model types:
- BiddingEvaluator / BiddingTransformer
- GameEvaluator / GameEvaluatorTransformer
- CardPlayTransformer

Usage:
    # Export a specific model
    python scripts/export_onnx.py --model bidding --checkpoint models/runs/my_run/bidding_best.pt

    # Export all models from a directory
    python scripts/export_onnx.py --all --models-dir models/

    # Export from a specific run
    python scripts/export_onnx.py --model card_play --run-name my_card_play_run
"""

import argparse
import json
from pathlib import Path

import torch

from skat_ml.constants import (
    BIDDING_EVALUATOR_INPUT_DIM,
    CARD_PAD_IDX,
    GAME_EVALUATOR_INPUT_DIM,
    MAX_HAND,
    MAX_HISTORY,
    MAX_OUVERT,
    MAX_SKAT,
    MAX_TRICK,
)
import torch.nn as nn

from skat_ml.models import (
    BiddingEvaluator,
    BiddingTransformer,
    CardPlayPolicy,
    CardPlayTransformer,
    GameEvaluator,
    GameEvaluatorTransformer,
)
from skat_ml.training_utils import strip_compiled_prefix


# Wrapper classes for ONNX export that apply sigmoid to convert logits to probabilities
class BiddingEvaluatorExport(nn.Module):
    """Wrapper that applies sigmoid for ONNX export."""
    def __init__(self, model: BiddingEvaluator):
        super().__init__()
        self.model = model

    def forward(self, x):
        pickup_logits, hand_logits = self.model(x)
        return torch.sigmoid(pickup_logits), torch.sigmoid(hand_logits)


class BiddingTransformerExport(nn.Module):
    """Wrapper that applies sigmoid for ONNX export."""
    def __init__(self, model: BiddingTransformer):
        super().__init__()
        self.model = model

    def forward(self, hand_cards, position):
        pickup_logits, hand_logits = self.model(hand_cards, position)
        return torch.sigmoid(pickup_logits), torch.sigmoid(hand_logits)


class GameEvaluatorExport(nn.Module):
    """Wrapper that applies sigmoid for ONNX export."""
    def __init__(self, model: GameEvaluator):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)


class GameEvaluatorTransformerExport(nn.Module):
    """Wrapper that applies sigmoid for ONNX export."""
    def __init__(self, model: GameEvaluatorTransformer):
        super().__init__()
        self.model = model

    def forward(self, hand_cards, skat_cards, skat_len, game_type, position, is_hand, bid):
        logits = self.model(hand_cards, skat_cards, skat_len, game_type, position, is_hand, bid)
        return torch.sigmoid(logits)


def load_checkpoint_and_config(checkpoint_path: Path) -> tuple[dict, dict]:
    """Load checkpoint and config, checking both checkpoint and config.json file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state_dict and config from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        # Raw state_dict without wrapper
        state_dict = checkpoint
        config = {}

    # If no config in checkpoint, try loading from config.json in same directory
    if not config:
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            print(f"Loading config from {config_path}")
            with open(config_path) as f:
                config = json.load(f)

    state_dict = strip_compiled_prefix(state_dict)
    return state_dict, config


def export_bidding(checkpoint_path: Path, output_dir: Path):
    """Export BiddingEvaluator or BiddingTransformer to ONNX."""
    print(f"Loading bidding model from {checkpoint_path}...")

    state_dict, config = load_checkpoint_and_config(checkpoint_path)
    model_type = config.get("model_type", "transformer")

    if model_type == "dense":
        model = BiddingEvaluator(
            input_dim=BIDDING_EVALUATOR_INPUT_DIM,
            hidden_dim=config.get("hidden_dim", 512),
            num_blocks=config.get("num_blocks", 4),
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Wrap with sigmoid for export (model outputs logits)
        export_model = BiddingEvaluatorExport(model)
        export_model.eval()

        dummy_input = torch.randn(1, BIDDING_EVALUATOR_INPUT_DIM)
        onnx_path = output_dir / "bidding_dense.onnx"

        torch.onnx.export(
            export_model,
            dummy_input,
            str(onnx_path),
            input_names=["features"],
            output_names=["pickup_probs", "hand_probs"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=17,
        )
    else:  # transformer
        model = BiddingTransformer(
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Wrap with sigmoid for export (model outputs logits)
        export_model = BiddingTransformerExport(model)
        export_model.eval()

        dummy_hand = torch.zeros(1, 10, dtype=torch.long)
        dummy_position = torch.zeros(1, dtype=torch.long)
        onnx_path = output_dir / "bidding_transformer.onnx"

        torch.onnx.export(
            export_model,
            (dummy_hand, dummy_position),
            str(onnx_path),
            input_names=["hand_cards", "position"],
            output_names=["pickup_probs", "hand_probs"],
            dynamic_axes={
                "hand_cards": {0: "batch"},
                "position": {0: "batch"},
            },
            opset_version=17,
        )

    print(f"Exported to {onnx_path}")
    _verify_onnx(onnx_path)


def export_game_eval(checkpoint_path: Path, output_dir: Path):
    """Export GameEvaluator or GameEvaluatorTransformer to ONNX."""
    print(f"Loading game eval model from {checkpoint_path}...")

    state_dict, config = load_checkpoint_and_config(checkpoint_path)
    model_type = config.get("model_type", "transformer")

    if model_type == "dense":
        model = GameEvaluator(
            input_dim=GAME_EVALUATOR_INPUT_DIM,
            hidden_dim=config.get("hidden_dim", 512),
            num_blocks=config.get("num_blocks", 4),
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Wrap with sigmoid for export (model outputs logits)
        export_model = GameEvaluatorExport(model)
        export_model.eval()

        dummy_input = torch.randn(1, GAME_EVALUATOR_INPUT_DIM)
        onnx_path = output_dir / "game_eval_dense.onnx"

        torch.onnx.export(
            export_model,
            dummy_input,
            str(onnx_path),
            input_names=["features"],
            output_names=["win_prob"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=17,
        )
    else:  # transformer
        model = GameEvaluatorTransformer(
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Wrap with sigmoid for export (model outputs logits)
        export_model = GameEvaluatorTransformerExport(model)
        export_model.eval()

        dummy_hand = torch.zeros(1, MAX_HAND, dtype=torch.long)
        dummy_skat = torch.full((1, MAX_SKAT), CARD_PAD_IDX, dtype=torch.long)
        dummy_skat_len = torch.zeros(1, dtype=torch.long)
        dummy_game_type = torch.zeros(1, dtype=torch.long)
        dummy_position = torch.zeros(1, dtype=torch.long)
        dummy_is_hand = torch.zeros(1, dtype=torch.long)
        dummy_bid = torch.zeros(1, dtype=torch.float32)

        onnx_path = output_dir / "game_eval_transformer.onnx"

        torch.onnx.export(
            export_model,
            (dummy_hand, dummy_skat, dummy_skat_len, dummy_game_type, dummy_position, dummy_is_hand, dummy_bid),
            str(onnx_path),
            input_names=["hand_cards", "skat_cards", "skat_len", "game_type", "position", "is_hand", "bid"],
            output_names=["win_prob"],
            dynamic_axes={
                "hand_cards": {0: "batch"},
                "skat_cards": {0: "batch"},
                "skat_len": {0: "batch"},
                "game_type": {0: "batch"},
                "position": {0: "batch"},
                "is_hand": {0: "batch"},
                "bid": {0: "batch"},
                "win_prob": {0: "batch"},
            },
            opset_version=17,
        )

    print(f"Exported to {onnx_path}")
    _verify_onnx(onnx_path)


def export_card_play(checkpoint_path: Path, output_dir: Path):
    """Export CardPlayPolicy or CardPlayTransformer to ONNX."""
    print(f"Loading card play model from {checkpoint_path}...")

    state_dict, config = load_checkpoint_and_config(checkpoint_path)
    model_type = config.get("model_type", "transformer")

    if model_type == "dense":
        model = CardPlayPolicy(
            hidden_dim=config.get("hidden_dim", 512),
            num_blocks=config.get("num_blocks", 4),
        )
        model.load_state_dict(state_dict)
        model.eval()

        from skat_ml.constants import CARD_PLAY_POLICY_INPUT_DIM
        dummy_input = torch.randn(1, CARD_PLAY_POLICY_INPUT_DIM)
        onnx_path = output_dir / "card_play_dense.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=17,
        )
    else:  # transformer
        model = CardPlayTransformer(
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 4),
            dropout=0.0,
        )
        model.load_state_dict(state_dict)
        model.eval()

        dummy_game_type = torch.zeros(1, dtype=torch.long)
        dummy_declarer = torch.zeros(1, dtype=torch.long)
        dummy_is_ouvert = torch.zeros(1, dtype=torch.long)
        dummy_hand = torch.zeros(1, MAX_HAND, dtype=torch.long)
        dummy_hand_len = torch.tensor([10], dtype=torch.long)
        dummy_ouvert_hand = torch.zeros(1, MAX_OUVERT, dtype=torch.long)
        dummy_ouvert_hand_len = torch.tensor([0], dtype=torch.long)
        dummy_history = torch.zeros(1, MAX_HISTORY, 2, dtype=torch.long)
        dummy_history_len = torch.tensor([0], dtype=torch.long)
        dummy_trick = torch.zeros(1, MAX_TRICK, 2, dtype=torch.long)
        dummy_trick_len = torch.tensor([0], dtype=torch.long)
        dummy_legal_mask = torch.ones(1, 32, dtype=torch.bool)

        dummy_inputs = (
            dummy_game_type,
            dummy_declarer,
            dummy_is_ouvert,
            dummy_hand,
            dummy_hand_len,
            dummy_ouvert_hand,
            dummy_ouvert_hand_len,
            dummy_history,
            dummy_history_len,
            dummy_trick,
            dummy_trick_len,
            dummy_legal_mask,
        )

        onnx_path = output_dir / "card_play_transformer.onnx"

        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_path),
            input_names=[
                "game_type",
                "declarer",
                "is_ouvert",
                "hand",
                "hand_len",
                "ouvert_hand",
                "ouvert_hand_len",
                "history",
                "history_len",
                "trick",
                "trick_len",
                "legal_mask",
            ],
            output_names=["logits"],
            dynamic_axes={
                "game_type": {0: "batch"},
                "declarer": {0: "batch"},
                "is_ouvert": {0: "batch"},
                "hand": {0: "batch"},
                "hand_len": {0: "batch"},
                "ouvert_hand": {0: "batch"},
                "ouvert_hand_len": {0: "batch"},
                "history": {0: "batch"},
                "history_len": {0: "batch"},
                "trick": {0: "batch"},
                "trick_len": {0: "batch"},
                "legal_mask": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

    print(f"Exported to {onnx_path}")
    _verify_onnx(onnx_path)


def _verify_onnx(onnx_path: Path):
    """Verify ONNX export with onnxruntime."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        print("  Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape}")
        print("  Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape}")
        print("  Verification successful!")
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")


def find_checkpoint(models_dir: Path, model_name: str, run_name: str = "") -> Path | None:
    """Find checkpoint file for a model."""
    candidates = []

    if run_name:
        run_dir = models_dir / "runs" / run_name
        candidates.extend([
            run_dir / f"{model_name}_best.pt",
            run_dir / f"{model_name}_latest.pt",
        ])

    # Check root models directory
    candidates.extend([
        models_dir / f"{model_name}_best.pt",
        models_dir / f"{model_name}_latest.pt",
    ])

    # Also check runs directories
    runs_dir = models_dir / "runs"
    if runs_dir.exists():
        for run in runs_dir.iterdir():
            if run.is_dir():
                candidates.append(run / f"{model_name}_best.pt")

    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Export trained models to ONNX")

    parser.add_argument(
        "--model",
        type=str,
        choices=["bidding", "game_eval", "card_play"],
        help="Model type to export",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to specific checkpoint file")
    parser.add_argument("--run-name", type=str, default="", help="Run name to export from")
    parser.add_argument("--models-dir", type=str, default="models/", help="Models directory")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for ONNX files")
    parser.add_argument("--all", action="store_true", help="Export all found models")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return

        if not args.model:
            print("Error: --model is required when using --checkpoint")
            return

        if args.model == "bidding":
            export_bidding(checkpoint_path, output_dir)
        elif args.model == "game_eval":
            export_game_eval(checkpoint_path, output_dir)
        elif args.model == "card_play":
            export_card_play(checkpoint_path, output_dir)

    elif args.all:
        print("Searching for models to export...")

        # Bidding
        path = find_checkpoint(models_dir, "bidding", args.run_name)
        if path:
            export_bidding(path, output_dir)
        else:
            print("No bidding model found, skipping")

        # Game eval
        path = find_checkpoint(models_dir, "game_eval", args.run_name)
        if path:
            export_game_eval(path, output_dir)
        else:
            print("No game_eval model found, skipping")

        # Card play
        path = find_checkpoint(models_dir, "card_play", args.run_name)
        if path:
            export_card_play(path, output_dir)
        else:
            print("No card_play model found, skipping")

    elif args.model:
        path = find_checkpoint(models_dir, args.model, args.run_name)
        if not path:
            print(f"Error: No checkpoint found for {args.model}")
            return

        if args.model == "bidding":
            export_bidding(path, output_dir)
        elif args.model == "game_eval":
            export_game_eval(path, output_dir)
        elif args.model == "card_play":
            export_card_play(path, output_dir)

    else:
        print("Error: Specify --model, --checkpoint, or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
