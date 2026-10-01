import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    # Sports
    "The football team won the match.",
    "The player scored a goal.",
    "The tennis player practiced for the tournament.",
    "The basketball team trained every morning.",
    "The coach discussed the strategy with the players.",
    "The runner finished the race in first place.",
    "The cricket team prepared for the final.",
    "The goalkeeper stopped the ball.",
    "The athletes competed in the national championship.",
    "The referee ended the game.",

    # Animals
    "The cat is sleeping on the sofa.",
    "Dogs are common household animals.",
    "The elephant walked through the forest.",
    "Birds build nests in trees.",
    "The horse ran across the field.",
    "The dolphin swam near the boat.",
    "The tiger lives in the jungle.",
    "The rabbit ate a carrot.",
    "The monkey climbed the tall tree.",
    "The penguin lives in a cold environment.",

    # Technology
    "Artificial intelligence is changing technology.",
    "Machine learning is used in many applications.",
    "The computer processed the data quickly.",
    "The software was updated yesterday.",
    "Cloud computing provides remote access to resources.",
    "The smartphone uses a powerful processor.",
    "The programmer developed a new application.",
    "The database stores large amounts of information.",
    "The robot can perform repetitive tasks.",
    "The network connects multiple computers.",

    # Food
    "The chef prepared a fresh meal.",
    "The restaurant serves spicy food.",
    "The bread was baked in the oven.",
    "The soup contains several vegetables.",
    "The family ordered pizza for dinner.",
    "The fruit was stored in the refrigerator.",
    "The cook added salt to the dish.",
    "The cake was prepared for the celebration.",
    "The market sells fresh vegetables.",
    "The rice was cooked with spices.",

    # Travel
    "The family traveled to the mountains.",
    "The train arrived at the station.",
    "The tourists visited the historical monument.",
    "The airplane landed at the airport.",
    "The hotel was located near the beach.",
    "The travelers packed their bags.",
    "The bus departed early in the morning.",
    "The tourists explored the city.",
    "The journey took several hours.",
    "The passengers waited for their flight."
]

categories = [
    "sports", "sports", "sports", "sports", "sports",
    "sports", "sports", "sports", "sports", "sports",

    "animals", "animals", "animals", "animals", "animals",
    "animals", "animals", "animals", "animals", "animals",

    "technology", "technology", "technology", "technology", "technology",
    "technology", "technology", "technology", "technology", "technology",

    "food", "food", "food", "food", "food",
    "food", "food", "food", "food", "food",

    "travel", "travel", "travel", "travel", "travel",
    "travel", "travel", "travel", "travel", "travel"
]

inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states

print("Number of sentences:", len(sentences))
print("Number of layers:", len(hidden_states))

train_indices, test_indices = train_test_split(
    range(len(sentences)),
    test_size=0.2,
    random_state=42,
    stratify=categories
)

for layer_number, layer in enumerate(hidden_states):

    mask = inputs["attention_mask"].unsqueeze(-1)

    sentence_vectors = (
        (layer * mask).sum(dim=1) /
        mask.sum(dim=1)
    ).numpy()

    X_train = sentence_vectors[train_indices]
    X_test = sentence_vectors[test_indices]

    y_train = [categories[i] for i in train_indices]
    y_test = [categories[i] for i in test_indices]

    classifier = LogisticRegression(max_iter=1000)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Layer", layer_number)
    print("Accuracy:", accuracy)

# Experiment 4 uses a controlled dataset.
# A harder dataset will be tested separately to reduce lexical cues.