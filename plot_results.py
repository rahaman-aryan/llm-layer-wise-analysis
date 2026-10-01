import matplotlib.pyplot as plt

layers = [0, 1, 2, 3, 4, 5, 6]

within = [
    0.65651476,
    0.80632854,
    0.86643650,
    0.89133200,
    0.89489620,
    0.85384350,
    0.75778130
]

between = [
    0.57588970,
    0.72053283,
    0.80736660,
    0.83624000,
    0.83377624,
    0.74069450,
    0.59803480
]

plt.plot(layers, within, marker="o", label="Within-category")
plt.plot(layers, between, marker="o", label="Between-category")

plt.xlabel("Transformer Layer")
plt.ylabel("Cosine Similarity")
plt.title("Layer-wise Representation Similarity")

plt.xticks(layers)
plt.legend()
plt.grid(True)

plt.savefig("results/experiment_02_similarity_plot.png", dpi=300)
plt.show()