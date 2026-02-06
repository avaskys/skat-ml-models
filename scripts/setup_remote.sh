#!/bin/bash
#
# Setup script for training Skat ML models on a remote machine.
# Syncs code from local machine (for testing unpushed changes).
#
# Usage:
#   # Basic setup (syncs to ~/skat-ml-models on remote)
#   ./scripts/setup_remote.sh user@gpu-server
#
#   # Custom remote directory
#   ./scripts/setup_remote.sh user@gpu-server --dir /data/skat-ml
#
#   # Sync only (skip environment setup)
#   ./scripts/setup_remote.sh user@gpu-server --sync-only
#

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(dirname "${SCRIPT_DIR}")"

# Default configuration
REMOTE_DIR="skat-ml-models"
PYTHON_CMD="python3"
SYNC_ONLY=false
SKIP_DATA=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 REMOTE_HOST [OPTIONS]

Setup Skat ML training environment on a remote machine by syncing from local.

Arguments:
    REMOTE_HOST         SSH destination (e.g., user@gpu-server)

Options:
    -d, --dir DIR       Remote directory name (default: skat-ml-models)
                        Will be created in remote home directory
    -p, --python CMD    Python command on remote (default: python3)
    --sync-only         Only sync code, skip environment setup
    --skip-data         Skip downloading training data
    -h, --help          Show this help message

Examples:
    # Basic setup
    $0 user@gpu-server

    # Custom remote directory
    $0 user@gpu-server --dir /data/skat-ml

    # Just sync code (after initial setup)
    $0 user@gpu-server --sync-only

    # Full setup but copy data manually later
    $0 user@gpu-server --skip-data
EOF
}

# Check for remote host argument
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

REMOTE_HOST="$1"
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --sync-only)
            SYNC_ONLY=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Skat ML Models - Remote Setup"
echo "=============================================="
echo ""
log_info "Local repository: ${LOCAL_REPO}"
log_info "Remote host: ${REMOTE_HOST}"
log_info "Remote directory: ${REMOTE_DIR}"
echo ""

# Verify local repository
if [[ ! -f "${LOCAL_REPO}/pyproject.toml" ]]; then
    log_error "Not a valid skat-ml-models repository: ${LOCAL_REPO}"
    exit 1
fi

# Test SSH connection
log_step "Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "echo 'SSH connection OK'" 2>/dev/null; then
    log_error "Cannot connect to ${REMOTE_HOST}"
    exit 1
fi

# Sync code to remote
log_step "Syncing code to remote..."
rsync -avz --delete \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude 'models/runs/' \
    --exclude 'models/*.onnx' \
    --exclude 'models/*.onnx.data' \
    --exclude '.DS_Store' \
    --exclude '*.egg-info/' \
    "${LOCAL_REPO}/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/"

log_info "Code synced successfully"

if [[ "${SYNC_ONLY}" == "true" ]]; then
    echo ""
    log_info "Sync complete (--sync-only mode)"
    echo ""
    echo "To train on remote:"
    echo "  ssh ${REMOTE_HOST}"
    echo "  cd ${REMOTE_DIR}"
    echo "  source venv/bin/activate"
    echo "  python scripts/train_bidding.py --sgf data/iss-games.sgf --model transformer"
    exit 0
fi

# Run setup on remote
log_step "Setting up environment on remote..."

ssh "${REMOTE_HOST}" bash << REMOTE_SCRIPT
set -e

cd "${REMOTE_DIR}"

echo ""
echo "=== Remote: Checking prerequisites ==="

# Check Python
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    echo "ERROR: ${PYTHON_CMD} not found"
    exit 1
fi

PYTHON_VERSION=\$(${PYTHON_CMD} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: \${PYTHON_VERSION}"

# Check version >= 3.10
MAJOR=\$(echo \$PYTHON_VERSION | cut -d. -f1)
MINOR=\$(echo \$PYTHON_VERSION | cut -d. -f2)
if [[ \$MAJOR -lt 3 ]] || [[ \$MAJOR -eq 3 && \$MINOR -lt 10 ]]; then
    echo "ERROR: Python 3.10+ required, found \${PYTHON_VERSION}"
    exit 1
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Detected ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    HAS_CUDA=true
else
    echo "WARNING: No NVIDIA GPU detected - training will use CPU"
    HAS_CUDA=false
fi

echo ""
echo "=== Remote: Setting up virtual environment ==="

if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    ${PYTHON_CMD} -m venv venv
else
    echo "Virtual environment already exists"
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing package..."
pip install -e . -q

echo ""
echo "=== Remote: Downloading training data ==="

if [[ -f "data/iss-games.sgf" ]]; then
    GAME_COUNT=\$(wc -l < data/iss-games.sgf | tr -d ' ')
    echo "Training data already exists: \${GAME_COUNT} games"
elif [[ "${SKIP_DATA}" == "true" ]]; then
    echo "Skipping data download (--skip-data)"
else
    echo "Downloading from ISS server..."
    python scripts/download_data.py
fi

echo ""
echo "=== Setup Complete ==="
REMOTE_SCRIPT

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo ""
echo "  ssh ${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  source venv/bin/activate"
echo ""
echo "  # Train bidding model"
echo "  python scripts/train_bidding.py --sgf data/iss-games.sgf --model transformer --epochs 20 --amp"
echo ""
echo "  # Train card play model"
echo "  python scripts/train_card_play.py --sgf data/iss-games.sgf --epochs 10 --amp"
echo ""
echo "To sync code changes later:"
echo "  ./scripts/setup_remote.sh ${REMOTE_HOST} --sync-only"
echo ""
echo "To copy trained models back:"
echo "  rsync -avz ${REMOTE_HOST}:${REMOTE_DIR}/models/runs/ ./models/runs/"
echo ""
