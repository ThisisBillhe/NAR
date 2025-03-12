#!/bin/bash

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# Set the dataset paths, modify the source paths to your own dataset paths!

# UCF101
ln -s /root/autodl-tmp/UCF-101/UCF-101 ./data/ucf101

# Kinetics-600
ln -s /root/autodl-tmp/kinetics-dataset/k600 ./data/k600

