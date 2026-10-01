import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sentences for the experiment
sentences = [
    "The football team won the match.",
    "The player scored a goal.",
    "The cat is sleeping on the sofa.",
    "Dogs are common household animals.",
    "Artificial intelligence is changing technology.",
    "Machine learning is used in many applications."
]

# Tokenize sentences
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Extract hidden states
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states

print("Number of layers:", len(hidden_states))

# Analyze every layer
for layer_number, layer in enumerate(hidden_states):

    # Average token representations to obtain one vector per sentence
    sentence_vectors = layer.mean(dim=1).numpy()

    # Calculate pairwise cosine similarity
    similarity = cosine_similarity(sentence_vectors)

    print("\nLayer", layer_number)
    print("Average similarity:", similarity.mean())