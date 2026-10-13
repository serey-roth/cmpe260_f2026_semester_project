#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WEBSHOP_DIR="third_party/webshop"

cd "$PROJECT_ROOT"

if [ ! -d "$WEBSHOP_DIR/.git" ]; then
    mkdir -p "$(dirname "$WEBSHOP_DIR")"
    git clone https://github.com/princeton-nlp/webshop.git "$WEBSHOP_DIR"
else
    echo "WebShop repository already exists."
fi

cd "$WEBSHOP_DIR"

if [ "$(uname -m)" = "arm64" ]; then
    bash setup_arm.sh -d small
else
    bash setup.sh -d small
fi

if [ ! -f "data/items_shuffle_1000.json" ]; then
    echo "WebShop setup did not download data/items_shuffle_1000.json." >&2
    echo "The upstream Google Drive data link may be unavailable or access-limited." >&2
    exit 1
fi