# Rule based reward models
Rule-based reward models that can be loaded with `AutoModelForSequenceClassification`.

## Development
```
pip install -r requirement-dev.txt
black .
pytest -xsvv .
```

## Usage
```
# Save a reward model for specified tokenizer and dataset
python main.py \
  --tokenizer-type=openai-community/gpt2 \
  --dataset-type=gsm8k \
  --save-dir=/path/to/save/dir
# Use it from AutoModelForSequenceClassification.from_pretrained
```
