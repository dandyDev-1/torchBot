import json
import numpy as np
from nltk_utils import (tokenize, stem, bag_of_words)

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset)

from model import NeuralNetwork
from settings import (INTENTS_FILE, PTH_FILE)


with open(INTENTS_FILE, 'r') as f:
    intents = json.load(f)



all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['intent']
    tags.append(tag)

    for text in intent['text']:
        w = tokenize(text)
        all_words.extend(w)
        xy.append((w, tag))


ignore_chars = ['?', ',', '.', '!', '$']
all_words = [stem(w) for w in all_words if w not in ignore_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


trainX = []
trainY = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    trainX.append(bag)

    label = tags.index(tag)
    trainY.append(label)


trainX = np.array(trainX)
trainY = np.array(trainY)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(trainX)
        self.xData = trainX
        self.yData = trainY

    def __getitem__(self, index):
        return self.xData[index], self.yData[index]

    def __len__(self):
        return self.n_samples


# Hyperparameters

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(trainX[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, PTH_FILE)
print(f"training complete, file saved to {PTH_FILE}")
