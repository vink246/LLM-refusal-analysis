#!/bin/bash

# Script to train sparse autoencoders and discover circuits
# Based on the methodology from the sparse feature circuits paper

CONFIG_PATH=$1

echo "Starting SAE training and circuit discovery..."
echo "Using config: $CONFIG_PATH"

# Train SAEs and discover circuits
python circuit_discovery_with_saes.py \
    --config $CONFIG_PATH \
    --train_saes

echo "SAE training and circuit discovery complete!"
# [file content end]