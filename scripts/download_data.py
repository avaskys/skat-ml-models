#!/usr/bin/env python3
"""
Download ISS (International Skat Server) game dataset for training.

The dataset contains millions of games played on skatgame.net in SGF format.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output data/games.sgf
    python scripts/download_data.py --url https://skatgame.net/iss/iss-games-07-2024.sgf.bz2
"""

import argparse
import bz2
import hashlib
import sys
import urllib.request
from pathlib import Path

# Known ISS dataset releases
DATASETS = {
    "2024-07": {
        "url": "https://skatgame.net/iss/iss-games-07-2024.sgf.bz2",
        "description": "July 2024 snapshot (~9M games)",
    },
}

DEFAULT_DATASET = "2024-07"


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress indicator."""
    print(f"Downloading: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "skat-ml-models/1.0"})

    with urllib.request.urlopen(request) as response:
        total_size = response.headers.get("Content-Length")
        if total_size:
            total_size = int(total_size)
            print(f"Size: {total_size / 1024 / 1024:.1f} MB")

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    pct = downloaded / total_size * 100
                    mb = downloaded / 1024 / 1024
                    print(f"\r  {mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    mb = downloaded / 1024 / 1024
                    print(f"\r  {mb:.1f} MB", end="", flush=True)

        print()  # Newline after progress


def decompress_bz2(src: Path, dest: Path) -> None:
    """Decompress a bz2 file with progress."""
    print(f"Decompressing: {src.name} -> {dest.name}")

    src_size = src.stat().st_size
    processed = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    with bz2.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            # Note: Can't easily track bz2 read progress, so we show output size
            processed += len(chunk)
            mb = processed / 1024 / 1024
            print(f"\r  Wrote {mb:.1f} MB", end="", flush=True)

    print()


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download ISS game dataset for Skat ML training"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/iss-games.sgf",
        help="Output path for the SGF file (default: data/iss-games.sgf)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=list(DATASETS.keys()),
        help=f"Dataset version to download (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to download (overrides --dataset)",
    )
    parser.add_argument(
        "--keep-compressed",
        action="store_true",
        help="Keep the compressed .bz2 file after extraction",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']}")
            print(f"         {info['url']}")
        return

    # Determine URL
    if args.url:
        url = args.url
    else:
        url = DATASETS[args.dataset]["url"]

    # Setup paths
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if output_path.exists():
        num_games = count_lines(output_path)
        print(f"Dataset already exists: {output_path}")
        print(f"  {num_games:,} games")
        print("Use --output to specify a different path, or delete the existing file.")
        return

    # Download
    compressed_path = output_path.with_suffix(".sgf.bz2")

    if compressed_path.exists():
        print(f"Using existing compressed file: {compressed_path}")
    else:
        download_with_progress(url, compressed_path)

    # Decompress
    decompress_bz2(compressed_path, output_path)

    # Cleanup
    if not args.keep_compressed:
        compressed_path.unlink()
        print(f"Removed: {compressed_path}")

    # Stats
    num_games = count_lines(output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024

    print()
    print(f"Dataset ready: {output_path}")
    print(f"  {num_games:,} games")
    print(f"  {size_mb:.1f} MB")
    print()
    print("Example training commands:")
    print(f"  python scripts/train_bidding.py --sgf {output_path} --model transformer")
    print(f"  python scripts/train_game_eval.py --sgf {output_path} --model transformer")
    print(f"  python scripts/train_card_play.py --sgf {output_path}")


if __name__ == "__main__":
    main()
