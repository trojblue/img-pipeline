import json

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class TagExpansionModel(nn.Module):
    def __init__(self, num_tags, embedding_dim, hidden_dim):
        super(TagExpansionModel, self).__init__()
        self.embedding = nn.Embedding(num_tags, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_tags)

    def forward(self, input_tags):
        embedded = self.embedding(input_tags)
        output, hidden = self.lstm(embedded)
        logits = self.linear(output[:, -1, :])
        return logits

class TagsDataset(Dataset):
    def __init__(self, folder_path, num_tags, tag_to_index, max_tag_len):
        self.folder_path = folder_path
        self.num_tags = num_tags
        self.tag_to_index = tag_to_index
        self.max_tag_len = max_tag_len

        self.file_list = os.listdir(folder_path)

    def __getitem__(self, index):
        file_path = os.path.join(self.folder_path, self.file_list[index])
        with open(file_path, 'r') as f:
            tags = f.readline().strip().split(', ')

        # Convert tags to indices
        tag_indices = [self.tag_to_index[tag] for tag in tags]

        # Pad or truncate input sequence to fixed length
        if len(tag_indices) >= self.max_tag_len:
            input_tags = torch.tensor(tag_indices[:self.max_tag_len - 1])
            target_tag = torch.tensor(tag_indices[self.max_tag_len - 1])
        else:
            input_tags = torch.tensor(tag_indices[:-1] + [0] * (self.max_tag_len - len(tag_indices)))
            target_tag = torch.tensor(tag_indices[-1])

        return input_tags, target_tag

    def __len__(self):
        return len(self.file_list)


def train(src_dir):
    # Load tag vocabulary
    # Load tag vocabulary
    tag_vocab = set()
    for filename in os.listdir(src_dir):
        with open(os.path.join(src_dir, filename), 'r') as f:
            tags = f.readline().strip().split(', ')
            tag_vocab.update(tags)

    # Create tag-to-index mapping
    tag_to_index = {tag: i for i, tag in enumerate(tag_vocab)}
    num_tags = len(tag_vocab)

    # Initialize model and optimizer on CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TagExpansionModel(num_tags=num_tags, embedding_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Create DataLoader for training data on CUDA device
    train_dataset = TagsDataset(src_dir, num_tags, tag_to_index, max_tag_len=50)

    train_loader = DataLoader(
        train_dataset,
        batch_size=30,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Train loop with progress bar
    num_epochs = 10
    for epoch in range(num_epochs):
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch in progress_bar:
            optimizer.zero_grad()
            input_tags, target_tag = batch
            input_tags, target_tag = input_tags.to(device), target_tag.to(device)
            logits = model(input_tags)
            loss = F.cross_entropy(logits, target_tag)
            loss.backward()
            optimizer.step()

            # Update progress bar with loss value
            progress_bar.set_postfix({'loss': loss.item()})

        # Close progress bar
        progress_bar.close()


    torch.save(model.state_dict(), 'tag_expansion_model.pth')
    with open ("vocab.json", "w") as f:
        json.dump(tag_to_index, f, indent=2)

def inference(num_tags):
    # Load saved model
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=32, hidden_dim=64)
    loaded_model.load_state_dict(torch.load('tag_expansion_model.pth'))

    with open ("vocab.json", "r") as f:
        tag_to_index = json.load(f)

    # Demo inference
    input_tags = torch.tensor([tag_to_index['1girl'], tag_to_index['long hair']])
    with torch.no_grad():
        output_probs = F.softmax(loaded_model(input_tags), dim=-1)
        top_indices = output_probs.topk(k=3).indices.squeeze().tolist()

    print('Input tags:', [list(tag_to_index.keys())[list(tag_to_index.values()).index(idx)] for idx in input_tags])
    print('Predicted tags:', [list(tag_to_index.keys())[list(tag_to_index.values()).index(idx)] for idx in top_indices])


def inf2():
    import json
    import torch

    # Load the trained model and tag vocabulary
    model = TagExpansionModel(num_tags=10, embedding_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load('tag_expansion_model.pth'))
    with open("vocab.json", "r") as f:
        tag_to_index = json.load(f)
    index_to_tag = {i: tag for tag, i in tag_to_index.items()}

    # Define a list of input tags
    input_tags = ['A', 'B', 'C']

    # Convert input tags to indices
    input_indices = [tag_to_index[tag] for tag in input_tags]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    # Generate output tags using the model
    output_indices = model.generate(input_tensor)
    output_tags = [index_to_tag[i.item()] for i in output_indices.squeeze()]

    print("Input tags:", input_tags)
    print("Expanded tags:", output_tags)

def inference2(num_tags):
    # Load saved model
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=32, hidden_dim=64)
    loaded_model.load_state_dict(torch.load('tag_expansion_model.pth'))

    with open("vocab.json", "r") as f:
        tag_to_index = json.load(f)

    # Demo inference
    input_tags = ['1girl', 'long hair']
    input_indices = [tag_to_index[tag] for tag in input_tags]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)
    with torch.no_grad():
        output_probs = F.softmax(loaded_model(input_tensor), dim=-1)
        top_indices = output_probs.topk(k=3).indices.squeeze().tolist()

    print('Input tags:', input_tags)
    print('Predicted tags:', [list(tag_to_index.keys())[list(tag_to_index.values()).index(idx)] for idx in top_indices])


if __name__ == '__main__':
    # src_dir = input("src_dir:")
    src_dir = "D:\Andrew\Pictures\==train\\t32.TXT"
    train(src_dir)
    # inference2(15314)
