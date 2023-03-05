from typing import *
import json
import os
import torch
import torch.nn as nn
import torch.cuda
import sdtools.txtops as tops
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
import random

class TagExpansionModel(nn.Module):
    def __init__(self, num_tags, embedding_dim, hidden_dim, num_layers=4, dropout=0.15):
        super(TagExpansionModel, self).__init__()
        self.embedding = nn.Embedding(num_tags, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_tags)
        )


    def forward(self, input_tags):
        embedded = self.embedding(input_tags)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        logits = self.mlp(hidden)
        return logits



class TagsDataset(Dataset):
    def __init__(self, file_path, num_tags, tag_to_index, max_tag_len):
        self.file_path = file_path
        self.num_tags = num_tags
        self.tag_to_index = tag_to_index
        self.max_tag_len = max_tag_len

        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        self.tag_indices = []


    def shuffle_tags(self, tags, prob, place):
        # Split the tags into a list
        tag_list = tags.split(', ')

        # Shuffle each tag with probability prob
        for i in range(len(tag_list)):
            if random.random() < prob:
                # Choose a random distance to shuffle
                shuffle_dist = random.randint(1, place)

                # Calculate the new index for the tag
                new_index = min(max(0, i + random.randint(-shuffle_dist, shuffle_dist)), len(tag_list) - 1)

                # Swap the tags
                tag_list[i], tag_list[new_index] = tag_list[new_index], tag_list[i]

        # Join the tags back into a comma-separated string
        return ', '.join(tag_list)
    def __getitem__(self, index):
        line = self.lines[index].strip()
        line = self.shuffle_tags(line, 0.2, 1)
        tags = line.split(', ')

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
        return len(self.lines)


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train(data_path):
    import wandb
    wandb.init(project='tag_expansion')

    # Load tag vocabulary
    tag_vocab = set()
    with open(data_path, 'r') as f:
        for line in f:
            tags = line.strip().split(', ')
            tag_vocab.update(tags)

    # Create tag-to-index mapping
    tag_to_index = {tag: i for i, tag in enumerate(tag_vocab)}
    num_tags = len(tag_vocab)

    # Initialize model and optimizer on CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TagExpansionModel(num_tags=num_tags, embedding_dim=64, hidden_dim=128).to(device)
    optimizer = AdamW(model.parameters())
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    # Create DataLoader for training data on CUDA device
    train_dataset = TagsDataset(data_path, num_tags, tag_to_index, max_tag_len=60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=704,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    # Train loop with progress bar
    num_epochs = 500

    # Set up wandb
    wandb.watch(model)
    wandb.config.update({
        'num_epochs': num_epochs,
        'batch_size': 704,
        'num_workers': 6,
        'learning_rate': 1e-3,
        'max_tag_len': 60,
        'embedding_dim': 128,
        'hidden_dim': 256
    })

    for epoch in range(num_epochs):
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_tags, target_tag = batch
            input_tags, target_tag = input_tags.to(device), target_tag.to(device)
            logits = model(input_tags)
            loss = F.cross_entropy(logits, target_tag)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss and learning rate to wandb
            wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})

            # Update progress bar with loss value
            progress_bar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        # Close progress bar
        progress_bar.close()

        if epoch % 20 == 0:
            # Save model and tag-to-index mapping
            torch.save(model.state_dict(), f'tag_expansion_model_epoch{epoch}.pth')
            with open ("vocab.json", "w") as f:
                json.dump(tag_to_index, f, indent=2)

    # Close wandb
    wandb.finish()



def inference(top_k = 10):

    # load json
    with open("vocab.json", "r") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)

    global model_path
    # Load saved model
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    loaded_model.load_state_dict(torch.load(model_path))

    with open("vocab.json", "r") as f:
        tag_to_index = json.load(f)

    # Demo inference
    while True:
        tag_str = input("input tags:")
        tag_list = [i.strip() for i in tag_str.split(",")]
        # input_tags = ['1girl', 'long hair']
        input_tags = tag_list
        input_indices = [tag_to_index[tag] for tag in input_tags]
        input_tensor = torch.tensor(input_indices).unsqueeze(0)
        with torch.no_grad():
            output_probs = F.softmax(loaded_model(input_tensor), dim=-1)
            top_indices = output_probs.topk(k=top_k).indices.squeeze().tolist()

        print('Input tags:', input_tags)
        print('Predicted tags:', [tag for idx in top_indices for tag, index in tag_to_index.items() if index == idx]
        )

def load_model(vocab_file=None, model_file=None, embedding_dim=64, hidden_dim=128):
    """
    从文件路径读取模型, 返回tag_to_index, loaded_model
    :param vocab_file: vocab.json
    :param model_file:tag_expansion_model.pth
    :return:
    :rtype:
    """
    global model_path
    # load json
    vocab_path = vocab_file if vocab_file else "vocab.json"
    with open(vocab_path, "r") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)

    # Load saved model
    model_path = model_file if model_file else model_path
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return tag_to_index, loaded_model

def inference_lstm_gradio(tag_to_index, loaded_model, input_tags, top_k:int=15):
    """
    gradio用的inference

    tag_to_index, loaded_model = load_model()

    :param tag_to_index: {"tag": int_index}
    :param loaded_model: pytorch model
    :param input_tags: ["1girl", "long hair"]
    :param top_k: int
    :return:
    """
    try:
        input_indices = [tag_to_index[tag] for tag in input_tags]
    except KeyError as e:
        return [f"[ERROR] key not found: {e}"]

    input_tensor = torch.tensor(input_indices).unsqueeze(0)
    with torch.no_grad():
        output_probs = F.softmax(loaded_model(input_tensor), dim=-1)
        top_indices = output_probs.topk(k=top_k).indices.squeeze().tolist()

    top_tags = [tag for idx in top_indices for tag, index in tag_to_index.items() if index == idx]
    tops.get_console_msg("INFO", f"lstm - Input tags: {input_tags}")
    tops.get_console_msg("INFO", f"lstm - Predicted tags: {top_tags}")

    return top_tags


# model_path = "tag_expansion_model_epoch480.pth"
model_path = "tag_expansion_model.pth"


if __name__ == '__main__':
    src_dir = "D:\Andrew\Pictures\==train\\t32.TXT"

    src_file = "D:\CSC3\expander\\tag_only.txt"
    # train(src_file)
    # embedding_dim = 128
    # hidden_dim = 256

    embedding_dim = 64
    hidden_dim = 128

    inference()
