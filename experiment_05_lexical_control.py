import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    # Group 1
    "The group completed the activity successfully.",
    "The participants prepared carefully beforehand.",
    "The group worked together during the session.",
    "The participants arrived early for the event.",
    "The group continued working throughout the morning.",
    "The participants followed the instructions carefully.",
    "The group discussed the result afterward.",
    "The participants remained focused during the activity.",
    "The group prepared for another session.",
    "The participants completed the task before noon.",
    "The group gathered before starting.",
    "The participants worked together toward the objective.",
    "The group returned after a short break.",
    "The participants continued the activity in the afternoon.",
    "The group reviewed the outcome afterward.",
    "The participants prepared themselves for the next stage.",
    "The group arrived at the location together.",
    "The participants followed the planned schedule.",
    "The group continued despite the delay.",
    "The participants completed another session.",
    "The group discussed different approaches.",
    "The participants remained active throughout the event.",
    "The group prepared for the final stage.",
    "The participants waited before beginning.",
    "The group worked together during preparation.",
    "The participants returned the following day.",
    "The group completed the planned activity.",
    "The participants reviewed their performance.",
    "The group continued toward the next stage.",
    "The participants gathered after the activity.",

    # Group 2
    "The group moved quietly through the area.",
    "The participants stayed together during the journey.",
    "The group searched the surroundings carefully.",
    "The participants returned after several hours.",
    "The group remained near the same location.",
    "The participants explored the surrounding area.",
    "The group continued moving toward the trees.",
    "The participants stayed close to the shelter.",
    "The group waited near the water.",
    "The participants followed the familiar route.",
    "The group moved across the open field.",
    "The participants remained near the others.",
    "The group returned before sunset.",
    "The participants explored a nearby location.",
    "The group stayed together throughout the day.",
    "The participants moved toward the distant area.",
    "The group rested after traveling.",
    "The participants continued along the path.",
    "The group remained hidden for several minutes.",
    "The participants returned to the same place.",
    "The group moved slowly through the area.",
    "The participants stayed near the shelter.",
    "The group searched beneath the trees.",
    "The participants followed the others.",
    "The group crossed the open ground.",
    "The participants remained together during the journey.",
    "The group explored the surrounding region.",
    "The participants returned after a short wait.",
    "The group moved toward a quieter location.",
    "The participants stayed near the group.",

    # Group 3
    "The system processed the information successfully.",
    "The device received the latest instructions.",
    "The system stored the information securely.",
    "The program completed the operation.",
    "The device connected after several attempts.",
    "The system handled the incoming request.",
    "The program generated the required result.",
    "The device transferred the information.",
    "The system monitored the process continuously.",
    "The program stored the final result.",
    "The device processed the incoming information.",
    "The system completed the requested task.",
    "The program received new instructions.",
    "The device communicated with the system.",
    "The system recorded the activity.",
    "The program processed several inputs.",
    "The device maintained the connection.",
    "The system analyzed the available information.",
    "The program updated the stored records.",
    "The device transferred the required data.",
    "The system continued working without interruption.",
    "The program handled the operation successfully.",
    "The device received information from another system.",
    "The system generated another result.",
    "The program completed another processing cycle.",
    "The device stored the latest information.",
    "The system responded after receiving the request.",
    "The program organized the available data.",
    "The device processed the information locally.",
    "The system completed the final operation.",

    # Group 4
    "The meal was prepared before the gathering.",
    "The dish was placed on the table.",
    "The ingredients were mixed carefully.",
    "The meal was prepared earlier that day.",
    "The dish was served after preparation.",
    "The ingredients were stored safely.",
    "The meal was cooked before the guests arrived.",
    "The dish required careful preparation.",
    "The ingredients were arranged before cooking.",
    "The meal was served shortly afterward.",
    "The dish was prepared in advance.",
    "The ingredients were combined carefully.",
    "The meal was placed on the table.",
    "The dish was cooked for several minutes.",
    "The ingredients were measured before preparation.",
    "The meal was ready before evening.",
    "The dish was served to everyone.",
    "The ingredients were prepared beforehand.",
    "The meal was cooked slowly.",
    "The dish was placed beside the other items.",
    "The ingredients were mixed before heating.",
    "The meal was prepared for the gathering.",
    "The dish was served after cooking.",
    "The ingredients were kept in separate containers.",
    "The meal was prepared earlier in the morning.",
    "The dish was left to cool.",
    "The ingredients were added during preparation.",
    "The meal was served in several portions.",
    "The dish was prepared according to instructions.",
    "The ingredients were stored after preparation.",

    # Group 5
    "The travelers arrived after several hours.",
    "The passengers waited before departure.",
    "The visitors explored the surrounding area.",
    "The journey continued throughout the morning.",
    "The group stayed there for several days.",
    "The passengers prepared before leaving.",
    "The visitors returned after exploring the area.",
    "The journey required careful planning.",
    "The group reached the destination safely.",
    "The passengers continued their journey.",
    "The travelers left early in the morning.",
    "The group waited before beginning the journey.",
    "The visitors spent the afternoon exploring.",
    "The passengers arrived shortly before departure.",
    "The journey continued through the evening.",
    "The group planned the route in advance.",
    "The travelers stayed near the destination.",
    "The visitors returned before sunset.",
    "The passengers prepared their belongings.",
    "The journey took longer than expected.",
    "The group reached the area before noon.",
    "The travelers waited for the next departure.",
    "The visitors explored several nearby places.",
    "The passengers continued waiting at the terminal.",
    "The journey began shortly after sunrise.",
    "The group traveled together throughout the day.",
    "The travelers arrived at the destination.",
    "The visitors spent several hours in the area.",
    "The passengers prepared for the next part.",
    "The group returned after a short visit."
]

categories = (
    ["group1"] * 30 +
    ["group2"] * 30 +
    ["group3"] * 30 +
    ["group4"] * 30 +
    ["group5"] * 30
)

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