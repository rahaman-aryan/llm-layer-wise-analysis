import matplotlib.pyplot as plt

layers = [0, 1, 2, 3, 4, 5, 6]

within = [
    0.68482465,
    0.80963340,
    0.86788370,
    0.89739950,
    0.90312520,
    0.86727077,
    0.74349725
]

between = [
    0.63235710,
    0.75155115,
    0.82926850,
    0.86480340,
    0.86843950,
    0.80756660,
    0.64099290
]

separation = [
    0.05246753,
    0.05808222,
    0.03861517,
    0.03259611,
    0.03468573,
    0.05970419,
    0.10250437
]

plt.plot(layers, within, marker="o", label="Within-category")
plt.plot(layers, between, marker="o", label="Between-category")
plt.plot(layers, separation, marker="o", label="Separation")

plt.xlabel("Transformer Layer")
plt.ylabel("Similarity / Separation")
plt.title("Experiment 3: Layer-wise Category Analysis")

plt.xticks(layers)
plt.legend()
plt.grid(True)

plt.savefig("results/experiment_03_similarity_plot.png", dpi=300)
plt.show()