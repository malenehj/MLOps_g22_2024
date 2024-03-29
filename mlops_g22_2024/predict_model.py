import os
from transformers import pipeline


def predict(text: str):
    #text = input('Write a sentence: ')  # the user inputs text to be predicted

    current_dir = os.path.dirname(__file__)  # gets the directory where the predict_model script is located
    relative_path = os.path.join(current_dir,'../models') # navigates to the saved model

    classifier = pipeline("text-classification", model=relative_path)

    return(classifier(text))
