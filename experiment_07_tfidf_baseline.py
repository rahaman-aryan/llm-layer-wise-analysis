
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# Load the existing 500-sentence dataset
from dataset_500 import sentences, categories

print("Number of sentences:", len(sentences))
print("Number of categories:", len(set(categories)))

accuracies = []

# Evaluate across five stratified train-test splits
for seed in range(5):
    train_sentences, test_sentences, y_train, y_test = train_test_split(
        sentences,
        categories,
        test_size=0.2,
        random_state=seed,
        stratify=categories
    )

    # Fit TF-IDF only on training data to prevent data leakage
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000)
    )

    model.fit(train_sentences, y_train)
    predictions = model.predict(test_sentences)

    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

    print(f"Split {seed + 1} accuracy: {accuracy:.4f}")

print("\nExperiment 7: TF-IDF Lexical Baseline")
print("Accuracies:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
print("Standard deviation:", np.std(accuracies))