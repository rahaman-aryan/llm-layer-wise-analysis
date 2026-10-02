from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# Load the existing 500-sentence dataset
from dataset_500 import sentences, categories

print("Number of sentences:", len(sentences))
print("Number of categories:", len(set(categories)))


# Load DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained(
    "distilbert-base-uncased",
    output_hidden_states=True
)

model.eval()


# Extract layer-wise representations
with torch.no_grad():
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    outputs = model(**inputs)

hidden_states = outputs.hidden_states

layer_representations = []

for layer in hidden_states:
    mask = inputs["attention_mask"].unsqueeze(-1)
    masked_embeddings = layer * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1)
    mean_embeddings = summed / counts
    layer_representations.append(mean_embeddings.numpy())


# Store results
tfidf_accuracies = []
distilbert_accuracies = [[] for _ in range(7)]


# Use identical five stratified splits
for seed in range(5):

    indices = np.arange(len(sentences))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=categories
    )

    train_sentences = [sentences[i] for i in train_idx]
    test_sentences = [sentences[i] for i in test_idx]

    y_train = [categories[i] for i in train_idx]
    y_test = [categories[i] for i in test_idx]


    # -----------------------------
    # TF-IDF baseline
    # -----------------------------

    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000)
    )

    tfidf_model.fit(train_sentences, y_train)

    tfidf_predictions = tfidf_model.predict(test_sentences)

    tfidf_accuracy = accuracy_score(
        y_test,
        tfidf_predictions
    )

    tfidf_accuracies.append(tfidf_accuracy)


    # -----------------------------
    # DistilBERT layer-wise probing
    # -----------------------------

    for layer_number in range(7):

        X = layer_representations[layer_number]

        X_train = X[train_idx]
        X_test = X[test_idx]

        classifier = LogisticRegression(max_iter=1000)

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        distilbert_accuracies[layer_number].append(accuracy)


# -----------------------------
# Print results
# -----------------------------

print("\nExperiment 8: Identical-Split Comparison")

print("\nTF-IDF accuracies:")
print(tfidf_accuracies)

print(
    "TF-IDF mean:",
    np.mean(tfidf_accuracies)
)

print(
    "TF-IDF std:",
    np.std(tfidf_accuracies)
)


print("\nDistilBERT accuracies:")

for layer_number in range(7):

    mean_accuracy = np.mean(
        distilbert_accuracies[layer_number]
    )

    std_accuracy = np.std(
        distilbert_accuracies[layer_number]
    )

    print(
        f"Layer {layer_number}: "
        f"{distilbert_accuracies[layer_number]} "
        f"Mean = {mean_accuracy:.4f}, "
        f"Std = {std_accuracy:.4f}"
    )


# -----------------------------
# Save results
# -----------------------------

with open(
    "results/experiment_08_identical_split_comparison.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Experiment 8: Identical-Split Comparison\n\n"
    )

    f.write(
        "Model comparison:\n"
        "1. TF-IDF (unigrams and bigrams) + Logistic Regression\n"
        "2. DistilBERT layer-wise representations + Logistic Regression\n\n"
    )

    f.write(
        "Dataset:\n"
        "500 sentences across 5 semantic categories.\n"
        "100 sentences per category.\n\n"
    )

    f.write(
        "Method:\n"
        "Both models were evaluated using the same five "
        "stratified 80/20 train-test splits with random seeds 0-4.\n\n"
    )

    f.write("TF-IDF Results:\n")

    for i, accuracy in enumerate(tfidf_accuracies):
        f.write(
            f"Split {i + 1}: {accuracy:.4f}\n"
        )

    f.write(
        f"Mean accuracy: {np.mean(tfidf_accuracies):.4f}\n"
    )

    f.write(
        f"Standard deviation: {np.std(tfidf_accuracies):.4f}\n\n"
    )

    f.write("DistilBERT Results:\n")

    for layer_number in range(7):

        mean_accuracy = np.mean(
            distilbert_accuracies[layer_number]
        )

        std_accuracy = np.std(
            distilbert_accuracies[layer_number]
        )

        f.write(
            f"Layer {layer_number}: "
            f"Mean accuracy = {mean_accuracy:.4f}, "
            f"Standard deviation = {std_accuracy:.4f}\n"
        )

    f.write(
        "\nInterpretation:\n"
        "The identical-split comparison provides a direct comparison "
        "between lexical TF-IDF features and DistilBERT representations "
        "because both models are evaluated on exactly the same "
        "train-test splits.\n\n"
    )

    f.write(
        "Limitation:\n"
        "The dataset is manually constructed and may contain "
        "category-specific lexical or structural cues. Therefore, "
        "the comparison should not be interpreted as a general "
        "comparison between TF-IDF and DistilBERT.\n"
    )