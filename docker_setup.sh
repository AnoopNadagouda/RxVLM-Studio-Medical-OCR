#!/bin/bash
set -e
pip install --ignore-installed PyYAML paddleocr 2>&1 | tail -10
python3 -c "from paddleocr import PaddleOCR; print('paddleocr import OK')"