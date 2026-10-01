import matplotlib.pyplot as plt

layers = [0, 1, 2, 3, 4, 5, 6]

accuracy = [
    0.83333333,
    0.83333333,
    0.86666667,
    0.83333333,
    0.90000000,
    0.90000000,
    0.83333333
]

plt.plot(layers, accuracy, marker="o")

plt.xlabel("Transformer Layer")
plt.ylabel("Classification Accuracy")
plt.title("Experiment 5: Lexical-Controlled Layer-wise Probing")

plt.xticks(layers)
plt.ylim(0, 1)
plt.grid(True)

plt.savefig("results/experiment_05_lexical_control_plot.png", dpi=300)
plt.show()