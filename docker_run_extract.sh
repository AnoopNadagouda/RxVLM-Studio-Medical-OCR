#!/bin/bash
set -e
pip install --ignore-installed PyYAML paddleocr > /dev/null 2>&1
pip install opencv-contrib-python > /dev/null 2>&1
python3 /data/extract_letters_v2.py --dataset avi-kai