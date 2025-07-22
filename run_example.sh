#!/usr/bin/env bash
# Helper script to run Inox2D examples with sample models
set -e

REPO_URL="https://github.com/Inochi2D/example-models"
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
    echo "Cloning example models..."
    git clone "$REPO_URL" "$MODELS_DIR"
    (cd "$MODELS_DIR" && git lfs pull)
fi

MODEL_PATH="$MODELS_DIR/${MODEL}.inx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model $MODEL_PATH not found. Available models:"
    ls "$MODELS_DIR"/*.inx | xargs -n1 basename | sed 's/\.inx$//'
    exit 1
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

