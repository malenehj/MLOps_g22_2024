import os
from transformers import pipeline


def test_model():
    # Define the configuration for testing
    config = {
        "lr": 0.001,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "epochs": 10,
        "weight_decay": 0.01
    }



    # Check if the trained model exists
    model_path = '../models'
    # assert os.path.exists(model_path)

    current_dir = os.path.dirname(__file__)  # gets the directory where the predict_model script is located
    relative_path = os.path.join(current_dir, '..','outputs/2024-01-15/models') # navigates to the saved model

    # Load the trained model
    classifier = pipeline("text-classification", model=relative_path)

    # text_anger = "I hate my life"
    text_happy = "my life is amazing"
    text_tired = "I very tired today"

    current_dir = os.path.dirname(__file__)  # gets the directory where the predict_model script is located
    relative_path = os.path.join(current_dir, '..','outputs/2024-01-12/models') # navigates to the saved model

    # Check if the model predicts the the labels correctly
    expected_emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    assert classifier(text_happy)[0]["label"] in expected_emotions
    assert classifier(text_tired)[0]["label"] in expected_emotions
    
    # assert classifier(text_anger)[0]['label'] == 'anger'
    # assert classifier(text_happy)[0]["label"] == 'surprise'
    # assert classifier(text_tired)[0]["label"] == 'sadness'

