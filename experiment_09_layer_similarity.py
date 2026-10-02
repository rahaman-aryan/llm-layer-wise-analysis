import torch
from transformers import AutoTokenizer, AutoModel
from dataset_500 import sentences, categories

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

all_layer_representations = []

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        outputs = model(**inputs, output_hidden_states=True)

        sentence_layers = []

        for hidden_state in outputs.hidden_states:
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_state * mask
            mean_representation = masked_hidden.sum(dim=1) / mask.sum(dim=1)

            sentence_layers.append(mean_representation.squeeze(0))

        all_layer_representations.append(sentence_layers)

num_layers = len(all_layer_representations[0])

print("Number of sentences:", len(sentences))
print("Number of layers:", num_layers)

similarity_matrix = torch.zeros(num_layers, num_layers)

for i in range(num_layers):
    for j in range(num_layers):
        similarities = []

        for sentence_index in range(len(sentences)):
            vector_i = all_layer_representations[sentence_index][i]
            vector_j = all_layer_representations[sentence_index][j]

            similarity = torch.nn.functional.cosine_similarity(
                vector_i.unsqueeze(0),
                vector_j.unsqueeze(0)
            )

            similarities.append(similarity.item())

        similarity_matrix[i, j] = sum(similarities) / len(similarities)

print("\nLayer-wise Representation Similarity:")
print(similarity_matrix)

with open(
    "results/experiment_09_layer_similarity.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("Experiment 09: Layer-wise Representation Similarity\n\n")
    file.write(f"Number of sentences: {len(sentences)}\n")
    file.write(f"Number of layers: {num_layers}\n\n")

    for i in range(num_layers):
        values = []

        for j in range(num_layers):
            values.append(f"{similarity_matrix[i, j].item():.6f}")

        file.write(
            f"Layer {i}: " + ", ".join(values) + "\n"
        )

print("\nResults saved to results/experiment_09_layer_similarity.txt")
