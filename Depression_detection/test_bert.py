from transformers import BertTokenizer, BertModel
import torch

try:
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Loading model...")
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
