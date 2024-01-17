'''
    make_dataset.py checks the processed folder for the datasets (if they have previously
    been added) - if yes, it loads them and returns them - if not:

    make_dataset.py loads the test.txt, train.txt and val.txt datasets from the .txt files
    and uses a transformer (DistilBertTokenizerFast) to tokanize the data and then saves it
    to the processed folder in the directory
'''

import os
import torch
from transformers import DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import sys

# sys.path.append("./")


# Method for loading the data
def load_data(file_name, filepath = 'data/raw'):
    # Construct the path relative to the current script
    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    relative_path = os.path.join(current_dir, '..', '..', filepath, file_name)
    texts, labels = [], []
    with open(relative_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';') # splitting the sentence from the label (emotion)
            texts.append(text.lower())  # lowercasing - not really needed here, everything lowercase (safety)
            labels.append(label)
        return texts, labels

# Encoding string labels to numerical format
def label_encoding(labels):
    """
    Label encoding converts each unique string label into a unique integer.

    Args:
    labels (list of str): A list of string labels to be encoded.

    Returns:
    numpy.ndarray: An array of encoded labels, where each label is represented
    as an integer.
    """
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

# Save processed dataset to a file for future use
def save_processed_data(dataset, file_name):
    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    processed_data_path = os.path.join(current_dir, '..', '..', 'data/processed', file_name + '_processed_data.pt')

    torch.save(dataset, processed_data_path)

# Dataset class for consistency
class EmotionDataset(torch.utils.data.Dataset):
    """
        A custom PyTorch Dataset for emotion data.

        This class is a subclass of torch.utils.data.Dataset and is used to handle
        the emotion data for a PyTorch model. It takes tokenized text data and
        corresponding labels and prepares them for model training or evaluation.

        Attributes:
        encodings (dict): Tokenized texts, typically output by a tokenizer.
        labels (list of int): Encoded labels corresponding to the tokenized texts.
    """
    # Initializes the EmotionDataset with encodings and labels
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        """
            Returns a data sample and its corresponding label at the given index.

            The method converts the data into PyTorch tensors, which makes them
            suitable for use with PyTorch models.

            Args:
            idx (int): The index of the data sample to retrieve.

            Returns:
            dict: A dictionary containing the data sample and its label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    # Returns the total number of samples in the dataset
    def __len__(self):
        return len(self.labels)

def check_processed_data_exists(filepath):
    return os.path.exists(filepath + '_data.pt')

def load_processed_data(filepath):
    # Load the saved tokenized texts and labels
    print()
    print()
    print()
    print(filepath + '_data.pt')
    print()
    print()
    print()
    dataset = torch.load(filepath + '_data.pt')
    return dataset

# Make Dataset method
def md():
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data/processed')
    '''
        Initialize the tokenizer for text processing using DistilBert.
        DistilBertTokenizerFast is a tokenizer that is paired with the 'distilbert-base-uncased' 
        model, which refers to a smaller and faster version of BERT that has been pre-trained on 
        a large corpus of text. 
        
        The 'uncased' version of the model does not make a distinction between uppercase and lowercase letters,
        which is generally beneficial for tasks where the case of letters is not important.
    '''
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # Check if datasets have already been processed and are ready to use
    if all(check_processed_data_exists(os.path.join(processed_data_path, dataset)) for dataset in ['train_processed'
                                                                                                   , 'val_processed'
                                                                                                   , 'test_processed']):
        print('Data already processed, loading it')

        # Load processed data
        train_dataset = load_processed_data(os.path.join(processed_data_path, 'train_processed'))
        val_dataset = load_processed_data(os.path.join(processed_data_path, 'val_processed'))
        test_dataset = load_processed_data(os.path.join(processed_data_path, 'test_processed'))
    else:
        print('Processing and loading data')
        # Load and process the data
        train_texts, train_labels = load_data('train.txt')
        train_labels = label_encoding(train_labels)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)

        val_texts, val_labels = load_data('val.txt')
        val_labels = label_encoding(val_labels)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        test_texts, test_labels = load_data('test.txt')
        test_labels = label_encoding(test_labels)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        # Creating Dataset objects
        train_dataset = EmotionDataset(train_encodings, train_labels)
        val_dataset = EmotionDataset(val_encodings, val_labels)
        test_dataset = EmotionDataset(test_encodings, test_labels)

        # Saving them
        save_processed_data(train_dataset, 'train')
        save_processed_data(val_dataset, 'val')
        save_processed_data(test_dataset, 'test')

    return train_dataset, val_dataset, test_dataset

md()

if __name__ == '__main__':
    # Get the data and process it
    pass