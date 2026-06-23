# ScAle

ScAle trains lightweight last-token scaling adapters for SpatialEval VQA.

The base vision-language model is frozen. Only small scaling parameters are trained using PyTorch hooks.

## Files

- `create_train_test_split.py`: creates train/test split JSON files
- `ft_vlm_spatial.py`: trains the ScAle adapter

## Install

```bash
pip install torch transformers datasets accelerate pillow numpy qwen-vl-utils
