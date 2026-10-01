import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sentences grouped by category
sentences = [
    "The football team won the match.",
    "The player scored a goal.",

    "The cat is sleeping on the sofa.",
    "Dogs are common household animals.",

    "Artificial intelligence is changing technology.",
    "Machine learning is used in many applications."
]

categories = [
    "sports",
    "sports",
    "animals",
    "animals",
    "technology",
    "technology"
]

# Tokenize
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

    # Attention-mask weighted mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1)

    masked_layer = layer * mask

    sentence_vectors = (
        masked_layer.sum(dim=1) /
        mask.sum(dim=1)
    ).numpy()

    # Pairwise cosine similarity
    similarity = cosine_similarity(sentence_vectors)

    within_scores = []
    between_scores = []

    # Compare every pair
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):

            if categories[i] == categories[j]:
                within_scores.append(similarity[i][j])
            else:
                between_scores.append(similarity[i][j])

    within_average = sum(within_scores) / len(within_scores)
    between_average = sum(between_scores) / len(between_scores)

    separation = within_average - between_average

    print("\nLayer", layer_number)
    print("Within-category similarity:", within_average)
    print("Between-category similarity:", between_average)
    print("Separation:", separation)