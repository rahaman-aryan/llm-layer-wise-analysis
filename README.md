# Layer-Wise Analysis of Representations in a Small Language Model

A research-oriented study of how sentence-level information is represented across the layers of DistilBERT.

## Overview

This project investigates how representations change across the seven hidden-state levels of `distilbert-base-uncased`.

The study uses layer-wise representation analysis, probing experiments, lexical controls, stability testing, and a TF-IDF baseline to examine whether predefined category information is recoverable from different layers.

The experiments are designed as a preliminary investigation rather than a claim about general semantic understanding.

## Model

* **Model:** DistilBERT (`distilbert-base-uncased`)
* **Framework:** PyTorch
* **Model interface:** Hugging Face Transformers
* **Classifier:** Logistic Regression
* **Evaluation:** Classification accuracy
* **Representation:** Attention-mask-weighted mean pooling of token representations

DistilBERT provides seven hidden-state representations:

* Layer 0 — initial embedding representation
* Layers 1–6 — Transformer layers

## Dataset

The main dataset contains **500 manually constructed sentences** distributed across five categories:

| Category   | Sentences |
| ---------- | --------: |
| Sports     |       100 |
| Animals    |       100 |
| Technology |       100 |
| Food       |       100 |
| Travel     |       100 |
| **Total**  |   **500** |

The dataset is stored centrally in `dataset_500.py` and reused across the relevant experiments.

Because the dataset is manually constructed, category-specific vocabulary and sentence patterns may introduce lexical or structural cues. This limitation is explicitly considered in the analysis.

## Experiments

### Experiment 1 — Baseline Representation Similarity

Measured average pairwise cosine similarity between sentence representations across layers.

**Observation:** Average similarity increased toward the middle layers and decreased toward the final layer.

This experiment established the initial analysis pipeline but did not distinguish within-category similarity from between-category similarity.

### Experiment 2 — Category Separation

Compared:

* within-category cosine similarity
* between-category cosine similarity
* category separation

where:

`separation = within-category similarity − between-category similarity`

Within-category similarity was higher than between-category similarity across all layers in the small dataset.

### Experiment 3 — Larger Category-Separation Dataset

Extended the category-separation analysis to a 50-sentence dataset covering five categories.

The experiment continued to show higher within-category than between-category similarity, although the manually constructed dataset limits the strength of the conclusion.

### Experiment 4 — Layer-Wise Probing

Used Logistic Regression classifiers to predict the predefined category from representations extracted at each DistilBERT layer.

Results:

| Layer | Accuracy |
| ----- | -------: |
| 0     |     0.92 |
| 1     |     0.90 |
| 2     |     0.93 |
| 3     |     0.92 |
| 4     |     0.93 |
| 5     |     0.93 |
| 6     |     0.93 |

**Observation:** Classification accuracy remained high across all layers, without a monotonic increase with layer depth.

### Experiment 5 — Lexical Control

Used a controlled dataset designed to reduce reliance on obvious category labels and vocabulary.

Results:

| Layer | Accuracy |
| ----- | -------: |
| 0     |   0.8333 |
| 1     |   0.8333 |
| 2     |   0.8667 |
| 3     |   0.8333 |
| 4     |   0.9000 |
| 5     |   0.9000 |
| 6     |   0.8333 |

The results suggest that lexical or surface-level information contributes to classification performance, while category information remains recoverable from the representations.

### Experiment 6 — Stability Across Train-Test Splits

Repeated the probing experiment across five stratified 80/20 train-test splits.

| Layer | Mean Accuracy | Standard Deviation |
| ----- | ------------: | -----------------: |
| 0     |         0.948 |            0.02135 |
| 1     |         0.942 |            0.02713 |
| 2     |         0.946 |            0.01855 |
| 3     |         0.940 |            0.02191 |
| 4     |         0.948 |            0.02040 |
| 5     |         0.944 |            0.01625 |
| 6     |         0.944 |            0.01625 |

**Observation:** Accuracy remained within a relatively narrow range across layers and did not show a monotonic increase with depth.

### Experiment 7 — TF-IDF Lexical Baseline

Evaluated a traditional lexical baseline using TF-IDF unigrams and bigrams with Logistic Regression.

Results across five stratified splits:

```text
0.98, 0.98, 0.95, 0.94, 0.97
```

* **Mean accuracy:** 0.964
* **Standard deviation:** 0.01625

The strong baseline demonstrates that lexical information provides substantial predictive information in the manually constructed dataset.

### Experiment 8 — Identical-Split Comparison

Compared TF-IDF and DistilBERT probing using the exact same five train-test splits.

| Representation     | Mean Accuracy | Standard Deviation |
| ------------------ | ------------: | -----------------: |
| TF-IDF             |         0.964 |             0.0162 |
| DistilBERT Layer 0 |         0.958 |             0.0147 |
| DistilBERT Layer 1 |         0.950 |             0.0190 |
| DistilBERT Layer 2 |         0.958 |             0.0160 |
| DistilBERT Layer 3 |         0.956 |             0.0162 |
| DistilBERT Layer 4 |         0.964 |             0.0196 |
| DistilBERT Layer 5 |         0.964 |             0.0206 |
| DistilBERT Layer 6 |         0.962 |             0.0194 |

Using identical splits makes the comparison more controlled than comparing results obtained from independently sampled splits.

## Overall Findings

Across these experiments:

1. Category information was recoverable from representations throughout the DistilBERT network.
2. Probing accuracy did not increase monotonically with layer depth.
3. The lexical-control experiment showed that surface-level cues contribute to classification performance.
4. The TF-IDF baseline achieved strong performance on the manually constructed dataset.
5. Repeated train-test splits produced relatively stable probing results.
6. Therefore, probing accuracy alone should not be interpreted as evidence that deeper layers necessarily contain more abstract semantic information.

## Limitations

This project is a preliminary research investigation and has several limitations:

* The main dataset is manually constructed.
* The five categories are relatively broad.
* Lexical and structural cues may remain in the dataset.
* The study uses a single pretrained model.
* The probing classifier is Logistic Regression.
* The experiments use relatively simple sentence-level mean pooling.
* Results have not yet been evaluated on an established external benchmark.
* The experiments do not establish causal explanations for what individual layers represent.

These limitations motivate future experiments using more controlled and independently sourced datasets.

## Repository Structure

```text
llm-layer-wise-analysis/
│
├── dataset_500.py
├── main.py
├── plot_results.py
│
├── experiment_03_dataset.py
├── plot_experiment_03.py
│
├── experiment_04_probing.py
├── experiment_04_harder_dataset.py
│
├── experiment_05_lexical_control.py
├── plot_experiment_05.py
│
├── experiment_06_stability.py
├── plot_experiment_06.py
│
├── experiment_07_tfidf_baseline.py
│
├── experiment_08_identical_split_comparison.py
├── plot_experiment_08.py
│
├── results/
│   ├── experiment_01_baseline.txt
│   ├── experiment_02_category_separation.txt
│   ├── experiment_03_large_dataset.txt
│   ├── experiment_04_probing.txt
│   ├── experiment_05_lexical_control.txt
│   ├── experiment_06_stability.txt
│   ├── experiment_07_tfidf_baseline.txt
│   └── experiment_08_identical_split_comparison.txt
│
├── requirements.txt
└── README.md
```

## Reproducibility

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment from the project directory:

```bash
python experiment_04_probing.py
```

Results and plots are stored in the `results/` directory.

## Tools and Libraries

* Python
* PyTorch
* Hugging Face Transformers
* scikit-learn
* NumPy
* Matplotlib
* Git / GitHub

## Research Direction

The project is intended to develop practical experience with:

* Transformer representations
* Layer-wise analysis
* Probing classifiers
* Experimental design
* Controlled comparisons
* Reproducibility
* Interpretation of model behavior
* Research-oriented evaluation
## Future Work

## Future Work

- [ ] Test representation stability with additional random seeds
- [ ] Evaluate the analysis on an externally sourced dataset
- [ ] Compare alternative sentence representation methods
- [ ] Investigate whether category information changes across model layers