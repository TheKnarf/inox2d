#!/usr/bin/env bash
# Helper script to run Inox2D examples with sample models
set -e

BASE_URL="https://github.com/Inochi2D/example-models/raw/refs/heads/main"
MODELS_DIR="examples/example-models"

usage() {
    echo "Usage: $0 <example-crate> [ModelName] [-- <extra cargo args>]"
    echo "  example-crate: render-opengl | render-wgpu | render-bevy | render-webgl"
    echo "  ModelName:    Aka | Midori (default: Aka)"
    exit 1
}

[ -z "$1" ] && usage
EXAMPLE="$1"; shift
MODEL="${1:-Aka}"
[ $# -gt 0 ] && shift

if [ ! -d "$MODELS_DIR" ]; then
    mkdir -p "$MODELS_DIR"
fi

MODEL_PATH="$MODELS_DIR/${MODEL}.inx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ${MODEL}.inx..."
    if ! curl -Lf "${BASE_URL}/${MODEL}.inx" -o "$MODEL_PATH"; then
        echo "Failed to download ${MODEL}.inx" >&2
        rm -f "$MODEL_PATH"
        exit 1
    fi
fi

case "$EXAMPLE" in
    render-webgl)
        # WebGL example runs with trunk
        (cd examples/render-webgl && trunk serve "$@")
        ;;
    *)
        cargo run -p "$EXAMPLE" -- "$MODEL_PATH" "$@"
        ;;
esac

