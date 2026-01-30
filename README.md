# Skat ML Models

Neural network models for playing the German card game Skat, trained on millions of games from the International Skat Server (ISS).

## Model Architecture

### Gameplay Phases

Skat has three decision phases, each with its own model:

```
BIDDING PHASE
├── BiddingEvaluator (dense)         ─┐
│   or                                ├── "Can I win at bid level X?"
└── BiddingTransformer (attention)   ─┘

GAME SELECTION PHASE (after winning bid)
├── GameEvaluator (dense)            ─┐
│   or                                ├── "Which discards + game type maximize win probability?"
└── GameEvaluatorTransformer         ─┘

CARD PLAY PHASE
└── CardPlayTransformer (attention)  ─── "Which card should I play?"
```

### Model Details

| Model | Input | Output | Architecture |
|-------|-------|--------|--------------|
| BiddingEvaluator | 35 features (10 cards + position) | 63 × 2 win probs (pickup/hand) | Dense ResNet |
| BiddingTransformer | 10 card indices + position | 63 × 2 win probs (pickup/hand) | Transformer |
| GameEvaluator | 75 features (hand + skat + context) | 1 win probability | Dense ResNet |
| GameEvaluatorTransformer | Card indices + context | 1 win probability | Transformer |
| CardPlayTransformer | Sequence of moves | 32 card logits | Transformer |

## Installation

```bash
# Clone the repository
git clone https://github.com/avaskys/skat-ml-models.git
cd skat-ml-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .
```

## Training

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Download Training Data

Download game data from the International Skat Server (ISS):

```bash
python scripts/download_data.py
```

This downloads ~9 million games in SGF format to `data/iss-games.sgf`.

### Training Commands

**Bidding models** (dense or transformer):
```bash
# Train BiddingTransformer (recommended)
python scripts/train_bidding.py --sgf data/iss-games.sgf --model transformer --epochs 20

# Train dense BiddingEvaluator
python scripts/train_bidding.py --sgf data/iss-games.sgf --model dense --epochs 20
```

**Game evaluation models** (dense or transformer):
```bash
# Train GameEvaluatorTransformer (recommended)
python scripts/train_game_eval.py --sgf data/iss-games.sgf --model transformer --epochs 20

# Train dense GameEvaluator
python scripts/train_game_eval.py --sgf data/iss-games.sgf --model dense --epochs 20
```

**Card play model** (transformer only):
```bash
python scripts/train_card_play.py --sgf data/iss-games.sgf --epochs 5
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sgf` | Path to SGF training data | Required |
| `--model` | `dense` or `transformer` (bidding/game_eval only) | `transformer` |
| `--epochs` | Number of training epochs | Varies |
| `--batch-size` | Batch size | Varies |
| `--lr` | Learning rate | Varies |
| `--output` | Output directory | `models/` |
| `--run-name` | Custom run name | Auto-generated |
| `--num-workers` | DataLoader workers | 4 |
| `--amp` | Use mixed precision (CUDA only) | False |
| `--compile` | Use torch.compile (CUDA only) | False |
| `--focal` | Use focal loss (bidding/game_eval) | False |
| `--gamma` | Focal loss gamma | 2.0 |

**Memory-constrained training:**
```bash
# Reduce memory usage on laptops/small GPUs
python scripts/train_bidding.py --sgf data/iss-games.sgf --model transformer \
    --batch-size 256 --num-workers 0
```

### Resuming Training

Training automatically resumes if a checkpoint exists. Use the same `--run-name`:

```bash
# Start training with specific settings
python scripts/train_bidding.py --sgf data/iss-games.sgf --model transformer \
    --run-name my_run --epochs 20 --batch-size 1024 --focal

# Later, resume (settings restored from checkpoint)
python scripts/train_bidding.py --sgf data/iss-games.sgf --run-name my_run --epochs 30
```

The script detects the existing checkpoint and restores all settings:
```
Found existing checkpoint: models/runs/my_run/bidding_latest.pt
  Restored from checkpoint: d_model=256, nhead=8, num_layers=4, batch_size=1024, focal=True
Resuming from models/runs/my_run/bidding_latest.pt
  Resumed at epoch 21, global_step 123456, best_val_loss 0.4532
```

**Overriding checkpoint settings:** Command-line arguments take precedence over checkpoint values:

```bash
# Resume but use smaller batch size (checkpoint had 1024)
python scripts/train_bidding.py --sgf data/iss-games.sgf --run-name my_run --batch-size 256
# Output: Overridden by command line: batch_size=256 (checkpoint had 1024)
```

### Monitoring

Training logs are written to TensorBoard:

```bash
tensorboard --logdir models/runs/
```

## Exporting to ONNX

Export trained models for Java inference:

```bash
# Export a specific model type
python scripts/export_onnx.py --model bidding --run-name my_run

# Export from a specific checkpoint
python scripts/export_onnx.py --model card_play --checkpoint models/runs/my_run/card_play_best.pt

# Export all available models
python scripts/export_onnx.py --all
```

Exported models are saved to `models/` with names like:
- `bidding_transformer.onnx`
- `game_evaluator_transformer.onnx`
- `card_play_transformer.onnx`

## SGF Data Format

Training data uses the ISS SGF format. Each line contains one complete game:

```
(;GM[Skat]P0[alice]R0[1523]P1[bob]R1[1487]P2[charlie]R2[1512]
DT[2024-07-15]ID[12345]
MV[w CJ.SA.HT.D9.C7.SJ.HA.DT.S9.H7|CA.ST.HK.DQ.C9.DJ.DA.SK.HQ.D7|CT.CK.CQ.C8.SQ.S8.S7.H9.H8.DK|HJ.D8
0 18 1 20 0 22 1 p 2 p 0 G.HJ.D8 ...]
R[d:0 win bidok m:2 p:78 v:48 s:0 z:0])
```

## Project Structure

```
skat-ml-models/
├── src/skat_ml/
│   ├── constants.py       # Card definitions, bid values
│   ├── features.py        # Feature extraction
│   ├── models/
│   │   ├── bidding.py     # BiddingEvaluator, BiddingTransformer
│   │   ├── game_eval.py   # GameEvaluator, GameEvaluatorTransformer
│   │   ├── card_play.py   # CardPlayTransformer
│   │   └── losses.py      # FocalLoss, BinaryFocalLoss
│   └── datasets/
│       ├── sgf_parser.py  # ISS SGF file parsing
│       ├── bidding.py     # StreamingBiddingDataset
│       ├── game_eval.py   # StreamingGameEvalDataset
│       └── card_play.py   # FastCardPlayDataset
├── scripts/
│   ├── download_data.py   # Download ISS game data
│   ├── train_bidding.py   # Train bidding models
│   ├── train_game_eval.py # Train game evaluation models
│   ├── train_card_play.py # Train card play model
│   └── export_onnx.py     # Export to ONNX format
└── models/                # Output (checkpoints + ONNX)
```

## Using Trained Models

### In Python

```python
import torch
from skat_ml.models import BiddingTransformer

# Load model
checkpoint = torch.load("models/runs/my_run/bidding_best.pt")
model = BiddingTransformer(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
hand_cards = torch.tensor([[0, 4, 11, 18, 25, 5, 12, 19, 26, 6]])  # 10 card indices
position = torch.tensor([0])  # Forehand
pickup_probs, hand_probs = model(hand_cards, position)
```

### In Java (via ONNX Runtime)

Use ONNX Runtime to load the exported models for inference.

## License

MIT
