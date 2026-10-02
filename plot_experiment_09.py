import matplotlib.pyplot as plt
import numpy as np

similarity_matrix = np.array([
    [1.0000, 0.7607, 0.6843, 0.6274, 0.5873, 0.5865, 0.3157],
    [0.7607, 1.0000, 0.8946, 0.8343, 0.7911, 0.7820, 0.4329],
    [0.6843, 0.8946, 1.0000, 0.9381, 0.8811, 0.8406, 0.4707],
    [0.6274, 0.8343, 0.9381, 1.0000, 0.9601, 0.8993, 0.5192],
    [0.5873, 0.7911, 0.8811, 0.9601, 1.0000, 0.9326, 0.5601],
    [0.5865, 0.7820, 0.8406, 0.8993, 0.9326, 1.0000, 0.6850],
    [0.3157, 0.4329, 0.4707, 0.5192, 0.5601, 0.6850, 1.0000]
])

layers = ["Layer 0", "Layer 1", "Layer 2", "Layer 3",
          "Layer 4", "Layer 5", "Layer 6"]

plt.figure(figsize=(8, 6))

plt.imshow(similarity_matrix)

plt.colorbar(label="Cosine Similarity")

plt.xticks(range(7), layers, rotation=45)
plt.yticks(range(7), layers)

plt.title("Layer-wise Representation Similarity")
plt.xlabel("Layer")
plt.ylabel("Layer")

plt.tight_layout()

plt.savefig(
    "results/experiment_09_layer_similarity.png",
    dpi=300
)

plt.show()