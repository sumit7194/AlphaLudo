#!/bin/bash
# Package AlphaLudo source code for Google Colab

# Create a clean directory structure for the zip
cp -r src colab_package/
cp setup.py colab_package/
cp train_mastery.py colab_package/
cp checkpoints_mastery/mastery_no6_v1/model_latest.pt colab_package/model_latest.pt
cp requirements.txt colab_package/ 2>/dev/null || true # If exists

# Create the zip file
zip -r alphaludo_src.zip colab_package

# Clean up
rm -rf colab_package

echo "Created alphaludo_src.zip"
