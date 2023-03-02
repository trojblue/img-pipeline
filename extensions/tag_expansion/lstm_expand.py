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


class TagExpansionModel(nn.Module):
    def __init__(self, num_tags, embedding_dim, hidden_dim, num_layers=4, dropout=0.2):
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
        self.linear = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, input_tags):
        embedded = self.embedding(input_tags)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        logits = self.linear(hidden)
        return logits


class TagsDataset(Dataset):
    def __init__(self, folder_path, num_tags, tag_to_index, max_tag_len):
        self.folder_path = folder_path
        self.num_tags = num_tags
        self.tag_to_index = tag_to_index
        self.max_tag_len = max_tag_len

        self.file_list = os.listdir(folder_path)
        self.tag_indices = []


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
    model = TagExpansionModel(num_tags=num_tags, embedding_dim=64, hidden_dim=128).to(device)
    optimizer = AdamW(model.parameters())

    # Create DataLoader for training data on CUDA device
    train_dataset = TagsDataset(src_dir, num_tags, tag_to_index, max_tag_len=50)

    train_loader = DataLoader(
        train_dataset,
        batch_size=600,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    # Train loop with progress bar
    num_epochs = 500

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='D:\CSC3\img-pipeline\_tensor_logs')

    # Set up CosineAnnealingWarmRestarts scheduler
    T_0 = 10
    T_mult = 2
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

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

            # Log loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

            # Update progress bar with loss value
            progress_bar.set_postfix({'loss': loss.item()})

        # Close progress bar
        progress_bar.close()

        # Step the scheduler
        scheduler.step()

    # Save model and tag-to-index mapping
    torch.save(model.state_dict(), 'tag_expansion_model.pth')
    with open ("vocab.json", "w") as f:
        json.dump(tag_to_index, f, indent=2)

    # Close TensorBoard writer
    writer.close()



def inference(top_k = 10):

    # load json
    with open("vocab.json", "r") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)

    # Load saved model
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=64, hidden_dim=128)
    loaded_model.load_state_dict(torch.load('tag_expansion_model.pth'))

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

def load_model(vocab_file=None, model_file=None):
    """
    从文件路径读取模型, 返回tag_to_index, loaded_model
    :param vocab_file: vocab.json
    :param model_file:tag_expansion_model.pth
    :return:
    :rtype:
    """

    # load json
    vocab_path = vocab_file if vocab_file else "vocab.json"
    with open(vocab_path, "r") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)

    # Load saved model
    model_path = model_file if model_file else "tag_expansion_model.pth"
    loaded_model = TagExpansionModel(num_tags=num_tags, embedding_dim=64, hidden_dim=128)
    loaded_model.load_state_dict(torch.load(model_path))

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


if __name__ == '__main__':
    src_dir = "D:\Andrew\Pictures\==train\\t32.TXT"
    # train(src_dir)
    inference()
