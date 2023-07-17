import json
import torch
import random

from settings import (INTENTS_FILE, PTH_FILE)
from functions import (NeuralNetwork, bag_of_words, tokenize)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

data = torch.load(PTH_FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "chippy_chippy"
print("Start Chatting! type 'quit' to exit. ")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents["intents"]:
            if tag == intent['intent']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')

    else:

        print(f"{bot_name}: I do not understand...")

