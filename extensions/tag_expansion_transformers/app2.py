import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import clip


class TagExpansionDataset(Dataset):
    def __init__(self, file_path, max_length=10):
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.max_length = max_length
        self.examples = []

        with open(file_path, "r") as f:
            for line in f:
                tags = line.strip().split(", ")
                input_tags = tags[:-1]
                output_tags = tags[1:]
                self.examples.append((input_tags, output_tags))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_tags, output_tags = self.examples[index]
        input_tags = ", ".join(input_tags)
        output_tags = ", ".join(output_tags)

        input_embeddings = self.get_tag_embeddings(input_tags)
        output_embeddings = self.get_tag_embeddings(output_tags)

        return input_embeddings, output_embeddings

    def get_tag_embeddings(self, tags):
        text = ", ".join(tags)
        text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text = {k: v.to("cuda") for k, v in text.items()}

        with torch.no_grad():
            embeddings = self.clip_model.encode_text(text['input_ids'])
        embeddings = embeddings.squeeze().cpu().numpy()

        return embeddings


class TagExpansionModel:
    def __init__(self, model_name="facebook/bart-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_file_path, batch_size=16, num_epochs=3):
        train_dataset = TagExpansionDataset(train_file_path, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in train_dataloader:
                input_ids, attention_mask, target_ids = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                target_ids = target_ids.to(self.device)

                # Pad input_ids and target_ids to the maximum sequence length
                max_seq_len = self.model.config.max_length
                input_ids = nn.functional.pad(input_ids, (0, max_seq_len - input_ids.shape[1]),
                                              value=self.tokenizer.pad_token_id)
                attention_mask = nn.functional.pad(attention_mask, (0, max_seq_len - attention_mask.shape[1]), value=0)
                target_ids = nn.functional.pad(target_ids, (0, max_seq_len - target_ids.shape[1]),
                                               value=self.tokenizer.pad_token_id)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     decoder_input_ids=target_ids[:, :-1], labels=target_ids[:, 1:])
                loss = outputs.loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    def predict(self, tags, max_length=20, num_beams=5):
        input_ids = self.tokenizer.encode(tags, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(input_ids=input_ids, max_length=max_length, num_beams=num_beams,
                                            early_stopping=True)
        generated_tags = [self.tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in generated_ids]
        return generated_tags


def enumerate_csv():
    import csv
    import os

    directory = "D:\Andrew\Pictures\==train\\t32.TXT"
    file_contents = []

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                contents = file.read().replace('\n', ' ')
                file_contents.append(contents)

    with open("output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for content in file_contents:
            writer.writerow([content])


if __name__ == '__main__':
    # enumerate_csv()
    tag_expansion_model = TagExpansionModel()
    tag_expansion_model.train(train_file_path="output.csv", batch_size=16, num_epochs=3)
    generated_tags = tag_expansion_model.predict("A, B")
    # print(generated_tags)
