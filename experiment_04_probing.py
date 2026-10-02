import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

from dataset_500 import sentences, categories

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

train_indices, test_indices = train_test_split(
    range(len(sentences)),
    test_size=0.2,
    random_state=42,
    stratify=categories
)

for layer_number, layer in enumerate(hidden_states):

    mask = inputs["attention_mask"].unsqueeze(-1)

    sentence_vectors = (
        (layer * mask).sum(dim=1) /
        mask.sum(dim=1)
    ).numpy()

    X_train = sentence_vectors[train_indices]
    X_test = sentence_vectors[test_indices]

    y_train = [categories[i] for i in train_indices]
    y_test = [categories[i] for i in test_indices]

    classifier = LogisticRegression(max_iter=1000)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Layer", layer_number)
    print("Accuracy:", accuracy)

# Experiment 4 uses a controlled dataset.
# A harder dataset will be tested separately to reduce lexical cues.