import torch
import wandb
import hydra
import os
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments, DistilBertTokenizerFast
from data.make_dataset import md, EmotionDataset
from models.model import model
from omegaconf import OmegaConf
from torch.profiler import profile, record_function, ProfilerActivity

"""
    This script is designed for training a sequence classification model using the Hugging Face Transformers library. 
    The model is trained to classify emotional content in text data, employing the DistilBERT architecture.
    
    Key Features:
    - Data Loading: The script begins by loading preprocessed data
    - Tokenization: Utilizes the DistilBertTokenizerFast, converting text data into a format suitable for DistilBERT
    - Model Training: Sets up a training environment using PyTorch, with support for MPS and CPU
                      Training involves calculating accuracy as a key metric.
    - Integration with Weights & Biases (wandb): Employs wandb for experiment tracking and logging
    - Configuration Management: Uses Hydra for managing and organizing configurations, making the script flexible
    - Model Saving: After training, the model is saved for future use
    
    Usage:
    Run the script directly to initiate the training process. Make sure to adjust the configuration files in './config' 
    as needed for specific training requirements.
"""

# Set an environment variable for Hydra to display full error stack traces.
os.environ["HYDRA_FULL_ERROR"] = "1"


# Define the main function with Hydra's configuration management
@hydra.main(config_path='./config', config_name='config')
def main(config):
    use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if use_wandb and wandb_api_key:
        import wandb  # Import wandb only if needed
        wandb.login(key=wandb_api_key)
        # Convert Hydra configuration to a dictionary and initialize wandb for experiment tracking.
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(project='dtu_mlops24', notes="Testing the WANDB", config=config_dict)


    # Convert Hydra configuration to a dictionary and initialize wandb for experiment tracking.
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Loading the datasets
    train_dataset, val_dataset, test_dataset = md()

    # --- MODEL TRAINING --- #

    # Set the training device (GPU or CPU).
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Function to compute accuracy during training
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Defining the training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        learning_rate=config.train.lr,
        per_device_train_batch_size=config.train.train_batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        num_train_epochs=config.train.epochs,
        weight_decay=config.train.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb" if use_wandb else "none",
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
    #Profiler implementation
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #  record_shapes=True,
            #  with_stack=True,
            #  profile_memory=True,
            #  on_trace_ready=torch.profiler.tensorboard_trace_handler('profiler')) as prof:
        # trainer.train()
    # 

    # Saving the trained model
    model_path = '../models'

    trainer.train()
    
    trainer.save_model(model_path)

    print('Model saved successfully at: ', model_path)

# Run the main function if the script is executed directly.
main()
