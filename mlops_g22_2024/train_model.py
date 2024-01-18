# import os
# import torch
# from transformers import Trainer, TrainingArguments, DistilBertTokenizerFast
# from mlops_g22_2024.data.make_dataset import md
# from mlops_g22_2024.models.model import model

# ...

"""
temp
"""
def main(config):
    # Convert Hydra configuration to a dictionary
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # # Loading the datasets
    # train_dataset, val_dataset, test_dataset = md()

    # # --- MODEL TRAINING --- #

    # # Set the training device (GPU or CPU).
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = torch.device(device)
    # print(f"Using device: {device}")

    # # Defining the training arguments
    # training_args = TrainingArguments(
    #     output_dir="./models",
    #     learning_rate=config.lr,
    #     per_device_train_batch_size=config.train_batch_size,
    #     per_device_eval_batch_size=config.eval_batch_size,
    #     num_train_epochs=config.epochs,
    #     weight_decay=config.weight_decay,
    # )

    # # Conditionally initialize WandB based on USE_WANDB
    # # trainer = Trainer(
    # #     model=model,
    # #     args=training_args,
    # #     train_dataset=train_dataset,
    # #     eval_dataset=val_dataset,
    # #     tokenizer=tokenizer,
    # #     data_collator=None,
    # #     callbacks=[],  # Disable WandB callbacks
    # # )

    # # trainer.train()

    # # Saving the trained model
    # model_path = 'models'
    # # trainer.save_model(model_path)

    # print('Model saved successfully at: ', model_path)

    return True

# Run the main function if the script is executed directly.
# main()
