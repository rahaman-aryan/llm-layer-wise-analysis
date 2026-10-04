
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch

from dataset_500 import sentences, categories

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("Number of sentences:", len(sentences))
print("Number of categories:", len(set(categories)))


word_lengths = [len(sentence.split()) for sentence in sentences]

print("\nSentence Length Statistics:")
print("Minimum words:", min(word_lengths))
print("Maximum words:", max(word_lengths))
print("Average words:", np.mean(word_lengths))

print("\nAverage Length by Category:")

for category in sorted(set(categories)):
    lengths = [
        len(sentence.split())
        for sentence, label in zip(sentences, categories)
        if label == category
    ]

    print(f"{category}: {np.mean(lengths):.2f} words")


    from collections import defaultdict

length_groups = defaultdict(lambda: defaultdict(list))

for sentence, category in zip(sentences, categories):
    length = len(sentence.split())
    length_groups[length][category].append(sentence)

matched_sentences = []
matched_categories = []

for length in sorted(length_groups):
    category_counts = [
        len(length_groups[length][category])
        for category in sorted(set(categories))
    ]

    if min(category_counts) > 0:
        count = min(category_counts)

        for category in sorted(set(categories)):
            selected = length_groups[length][category][:count]

            matched_sentences.extend(selected)
            matched_categories.extend([category] * count)

print("\nLength-Matched Dataset:")
print("Number of sentences:", len(matched_sentences))

print("\nSentences per category:")

for category in sorted(set(matched_categories)):
    count = matched_categories.count(category)
    print(f"{category}: {count}")

matched_lengths = [len(sentence.split()) for sentence in matched_sentences]

print("\nMatched Length Statistics:")
print("Minimum words:", min(matched_lengths))
print("Maximum words:", max(matched_lengths))
print("Average words:", np.mean(matched_lengths))


all_layer_representations = []

with torch.no_grad():
    for sentence in matched_sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        outputs = model(
            **inputs,
            output_hidden_states=True
        )

        sentence_layers = []

        for hidden_state in outputs.hidden_states:
            mask = inputs["attention_mask"].unsqueeze(-1)

            masked_hidden = hidden_state * mask

            mean_representation = (
                masked_hidden.sum(dim=1) / mask.sum(dim=1)
            )

            sentence_layers.append(
                mean_representation.squeeze(0)
            )

        all_layer_representations.append(sentence_layers)

num_layers = len(all_layer_representations[0])

print("\nNumber of matched sentences:", len(matched_sentences))
print("Number of layers:", num_layers)


print("\nLength-Controlled Probing Results:")

for layer in range(num_layers):

    X = torch.stack([
        all_layer_representations[i][layer]
        for i in range(len(matched_sentences))
    ]).numpy()

    y = np.array(matched_categories)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    classifier = LogisticRegression(
        max_iter=1000
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Layer {layer} Accuracy: {accuracy:.4f}"
    )


with open(
    "results/experiment_10_length_control.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("Experiment 10: Sentence-Length-Controlled Probing\n\n")
    file.write(f"Original sentences: {len(sentences)}\n")
    file.write(f"Matched sentences: {len(matched_sentences)}\n")
    file.write("Sentences per category: 53\n")
    file.write("Matched length range: 5-9 words\n")
    file.write(
        f"Matched average length: {np.mean(matched_lengths):.2f} words\n\n"
    )

    file.write("Layer-wise Accuracy:\n")

    for layer in range(num_layers):

        X = torch.stack([
            all_layer_representations[i][layer]
            for i in range(len(matched_sentences))
        ]).numpy()

        y = np.array(matched_categories)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        file.write(
            f"Layer {layer}: {accuracy:.4f}\n"
        )

print("\nResults saved to results/experiment_10_length_control.txt")