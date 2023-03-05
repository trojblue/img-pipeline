import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer


class TagCompletionModel:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict_tags(self, input_tags, max_length=10):
        input_string = "expand tags: " + ", ".join(input_tags)
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length + len(input_ids[0]),
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=0.92,
            repetition_penalty=1.0
        )
        predicted_tags = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).split(", ")
        return predicted_tags

    def train(self, train_dataset, validation_dataset, output_dir):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_total_limit=1,
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            predict_with_generate=True,
            logging_dir="./logs",
            logging_steps=500,
            overwrite_output_dir=True,
            save_steps=1000
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=validation_dataset,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


from transformers import LineByLineTextDataset

# Load the training data from a text file
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128,
)

# Load the validation data from a text file
validation_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="validation.txt",
    block_size=128,
)
model = TagCompletionModel()
model.train(train_dataset, validation_dataset, output_dir="output/")

