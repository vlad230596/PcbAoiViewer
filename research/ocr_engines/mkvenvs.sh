#!/bin/sh
# Creates one venv per engine under research/ocr_engines/venvs (never touches the project venv/).
# Caches on D: (C: is nearly full): pip -> cache/pip, models -> cache/hf etc.
cd "$(dirname "$0")"
export PIP_CACHE_DIR="$PWD/cache/pip"
base="opencv-python-headless numpy"
mk() {  # name, pip args...
  n=$1; shift
  [ -x venvs/$n/Scripts/python.exe ] || py -3.13 -m venv venvs/$n
  venvs/$n/Scripts/python.exe -m pip install -q --disable-pip-version-check $base "$@" > logs/install_$n.log 2>&1
  echo "$n exit $?"
}
case "$1" in
  tess)    mk tess ;;
  rapid3)  mk rapid3 rapidocr onnxruntime ;;
  paddle)  mk paddle paddlepaddle paddleocr ;;
  easyocr) mk easyocr --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision easyocr ;;
  doctr)   mk doctr --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision "python-doctr[torch]" ;;
  hf)      mk hf --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision transformers accelerate timm einops sentencepiece protobuf ;;
esac
