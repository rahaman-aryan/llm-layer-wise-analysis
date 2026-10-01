import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Same 500-sentence dataset used in Experiment 4
exec(open("experiment_04_harder_dataset.py", encoding="utf-8").read())

categories = (
    ["sports"] * 100 +
    ["animals"] * 100 +
    ["technology"] * 100 +
    ["food"] * 100 +
    ["travel"] * 100
)

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states

print("Number of sentences:", len(sentences))
print("Number of layers:", len(hidden_states))

seeds = [42, 123, 456, 789, 2026]

results = []

for layer_number, layer in enumerate(hidden_states):

    mask = inputs["attention_mask"].unsqueeze(-1)

    sentence_vectors = (
        (layer * mask).sum(dim=1) /
        mask.sum(dim=1)
    ).numpy()

    accuracies = []

    for seed in seeds:

        train_indices, test_indices = train_test_split(
            range(len(sentences)),
            test_size=0.2,
            random_state=seed,
            stratify=categories
        )

        X_train = sentence_vectors[train_indices]
        X_test = sentence_vectors[test_indices]

        y_train = [categories[i] for i in train_indices]
        y_test = [categories[i] for i in test_indices]

        classifier = LogisticRegression(max_iter=1000)

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    results.append((mean_accuracy, std_accuracy))

    print("\nLayer", layer_number)
    print("Accuracies:", accuracies)
    print("Mean accuracy:", mean_accuracy)
    print("Standard deviation:", std_accuracy)