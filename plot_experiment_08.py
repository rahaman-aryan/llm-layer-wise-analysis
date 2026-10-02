import matplotlib.pyplot as plt
import numpy as np


layers = np.arange(7)

distilbert_mean = [
    0.958,
    0.950,
    0.958,
    0.956,
    0.964,
    0.964,
    0.962
]

distilbert_std = [
    0.0147,
    0.0190,
    0.0160,
    0.0162,
    0.0196,
    0.0206,
    0.0194
]

tfidf_mean = 0.964
tfidf_std = 0.0162


plt.figure(figsize=(8, 5))

plt.errorbar(
    layers,
    distilbert_mean,
    yerr=distilbert_std,
    marker="o",
    capsize=4,
    label="DistilBERT"
)

plt.axhline(
    tfidf_mean,
    linestyle="--",
    label="TF-IDF"
)

plt.fill_between(
    layers,
    tfidf_mean - tfidf_std,
    tfidf_mean + tfidf_std,
    alpha=0.15
)

plt.xlabel("DistilBERT Layer")
plt.ylabel("Mean Accuracy")
plt.title("Experiment 8: TF-IDF vs DistilBERT on Identical Splits")

plt.xticks(layers)
plt.ylim(0.90, 1.00)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/experiment_08_identical_split_comparison.png",
    dpi=300
)

plt.show()