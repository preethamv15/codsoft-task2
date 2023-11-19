import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
import json
from collections import Counter
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
learning_rate = 0.001
num_epochs = 5
batch_size = 32
image_dir = 'images/'
caption_path = 'captions.json'

# Load vocabulary
with open('vocab.json', 'r') as j:
    vocab = json.load(j)

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Image preprocessing
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

# Load pretrained model and remove the classification head
model = models.resnet152(pretrained=True)
modules = list(model.children())[:-1]
model = nn.Sequential(*modules)
model = model.to(device)

# RNN (LSTM) for captioning
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNN, self).__init()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(rnn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for images, captions in dataloader:
        # Forward pass
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        outputs = rnn(images, captions)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every 100 batches
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Save the model
torch.save(rnn.state_dict(), 'model.ckpt')