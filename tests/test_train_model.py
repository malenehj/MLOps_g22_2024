import os
from transformers import DistilBertTokenizerFast
from mlops_g22_2024.train_model import main

def test_main():
    # Define the configuration for testing
    config = {
        "train": {
            "lr": 0.001,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "epochs": 10,
            "weight_decay": 0.01
        }
    }

    # Set up any necessary environment variables
    os.environ["USE_WANDB"] = "false"

    # Call the main function
    main(config)

    # Check if the trained model exists
    model_path = '../models'
    assert os.path.exists(model_path)

    # Load the trained model and tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Perform some sample predictions
    text_happy = "my life is amazing"
    text_tired = "I am very tired today"

    inputs_happy = tokenizer(text_happy, return_tensors="pt")
    inputs_tired = tokenizer(text_tired, return_tensors="pt")

    outputs_happy = model(**inputs_happy)
    outputs_tired = model(**inputs_tired)

    # Check if the model predicts the labels correctly
    expected_emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    assert outputs_happy.logits.argmax().item() in expected_emotions
    assert outputs_tired.logits.argmax().item() in expected_emotions

    # Clean up any temporary files or directories
    os.remove(model_path)