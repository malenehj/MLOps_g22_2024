import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from data.make_dataset import md
from models.model import model

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Following this guide: https://huggingface.co/docs/transformers/tasks/sequence_classification


# --- LOADING THE DATA --- #

train_dataset, val_dataset,test_dataset = md()

# --- MODEL TRAINING --- #

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")

# Defining a function to compute accuracy during training
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Defining the training arguments
training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,
    compute_metrics=compute_metrics,
)

trainer.train()


'''
from transformers import BertForSequenceClassification, AdamW
from models.model import model
from data.make_dataset import md
import torch

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 10
train_dataloader, val_dataloader, test_dataloader = md()

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")

# Training loop
for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_train_loss = 0
    model.train()

    i = 0

    num_batches = len(train_dataloader)
    print("Number of batches in train_dataloader:", num_batches)

    for step, batch in enumerate(train_dataloader):
        # Unpack this training batch from our dataloader.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        # Clear any previously calculated gradients
        model.zero_grad()

        # Forward pass
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs.loss
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Update parameters and take a step using the computed gradient.
        optimizer.step()

        print(i)
        i = i + 1

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

# Put model in evaluation mode
model.eval()

# Tracking variables
total_eval_accuracy = 0
total_eval_loss = 0

# Evaluate data for one epoch
for batch in val_dataloader:
    # Unpack this training batch from our dataloader.
    # Forward pass, calculate logit predictions.
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    loss = outputs.loss
    total_eval_loss += loss.item()
# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(val_dataloader)
print("  Validation Loss: {0:.2f}".format(avg_val_loss))
'''
