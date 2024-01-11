import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from data.make_dataset import md
from models.model import model
import wandb
from transformers import DistilBertTokenizerFast
import hydra
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from omegaconf import OmegaConf



# specify path of config file to later pass it to wandb
@hydra.main(config_path='./config', config_name='config')

def main(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.init(project='dtu_mlops24', notes="Testing the WANDB", config=config_dict)
    '''wandb.init(project='dtu_mlops24',
               notes="Testing the WANDB",
               config=config,  # specify config file to read the hyperparameters from
               )'''

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Following this guide: https://huggingface.co/docs/transformers/tasks/sequence_classification

    # --- LOADING THE DATA --- #

    train_dataset, val_dataset, test_dataset = md()

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
        learning_rate=config.lr,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
        # run_name="bertdistil",  # name of the W&B run
        logging_steps=400,  # how often to log to W&B
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

    # --- SAVING THE MODEL --- #

    model_path = './models'
    trainer.save_pretrained(model_path)

    print('Model saved successfully at: ', model_path)

main()


'''wandb.init(project='dtu_mlops24',
           notes="Testing the WANDB",
           config = config, #specify config file to read the hyperparameters from
           )


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
    learning_rate=config.lr,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    num_train_epochs=config.epochs,
    weight_decay=config.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="wandb",
    #run_name="bertdistil",  # name of the W&B run 
    logging_steps=400,  # how often to log to W&B
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

# --- SAVING THE MODEL --- #

model_path = './models'
trainer.save_pretrained(model_path)


print('Model saved successfully at: ', model_path)'''
