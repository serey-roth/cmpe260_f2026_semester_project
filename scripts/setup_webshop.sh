#!/usr/bin/env bash
set -e

WEBSHOP_DIR="third_party/webshop"

if [ ! -d "$WEBSHOP_DIR/.git" ]; then
    mkdir -p third_party
    git clone https://github.com/princeton-nlp/webshop.git "$WEBSHOP_DIR"
else
    echo "WebShop repository already exists."
fi

cd "$WEBSHOP_DIR"
bash setup.sh -d small