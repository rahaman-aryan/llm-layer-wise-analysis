import matplotlib.pyplot as plt

layers = [0, 1, 2, 3, 4, 5, 6]

mean_accuracy = [
    0.948,
    0.942,
    0.946,
    0.940,
    0.948,
    0.944,
    0.944
]

std_accuracy = [
    0.02135,
    0.02713,
    0.01855,
    0.02191,
    0.02040,
    0.01625,
    0.01625
]

plt.errorbar(
    layers,
    mean_accuracy,
    yerr=std_accuracy,
    marker="o",
    capsize=5
)

plt.xlabel("Transformer Layer")
plt.ylabel("Mean Classification Accuracy")
plt.title("Experiment 6: Stability of Layer-wise Probing")

plt.xticks(layers)
plt.ylim(0.85, 1.0)
plt.grid(True)

plt.savefig(
    "results/experiment_06_stability_plot.png",
    dpi=300
)

plt.show()