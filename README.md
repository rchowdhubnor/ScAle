# ScAle

ScAle trains lightweight last-token scaling adapters for SpatialEval VQA.

The base vision-language model is frozen. Only small scaling parameters are trained using PyTorch hooks.
## Dataset

This code uses the SpatialEval VQA dataset.

- Official SpatialEval repository: https://github.com/jiayuww/SpatialEval
- Hugging Face dataset: https://huggingface.co/datasets/MilaWang/SpatialEval
## Files

- `create_train_test_split.py`: creates train/test split JSON files
- `ft_vlm_spatial.py`: trains the ScAle adapter

## Install

```bash
pip install torch transformers datasets accelerate pillow numpy qwen-vl-utils
