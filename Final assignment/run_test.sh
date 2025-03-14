#!/bin/bash

# Activate your virtual environment (if needed)
# source venv/bin/activate  # Uncomment if using a virtual environment

# Run the Python script with minimal settings
python train.py \
  --data-dir "./data/cityscapes" \
  --batch-size 2 \
  --epochs 2 \
  --lr 0.0001 \
  --num-workers 0 \
  --seed 42 \
  --experiment-id "test-run"