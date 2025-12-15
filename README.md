# Transformer XL

This is an implementation of [Transformer-XL](https://arxiv.org/abs/1901.02860)
paper on top of GPT-2 implementation of nanoGPT.

## Instructions

```bash
# Install dependencies
uv sync

# Train the model
uv run src/train.py <training_config>

# Sample from the model
uv run src/sample.py <sample_config>
```
