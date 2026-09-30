import torch
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Artificial intelligence is changing the way we understand language."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states

print("Number of layers:", len(hidden_states))

for i, layer in enumerate(hidden_states):
    print("Layer", i, "shape:", layer.shape)