'''
make_dataset.py loads the test.txt, train.txt and val.txt datasets from the .txt files
and uses a transformer (AutoTokanizer) to tokanize the data
'''

import os
import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder

# Method for loading the data
def load_data(filepath):
    # Construct the path relative to the current script
    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    relative_path = os.path.join(current_dir, '..', '..', 'data/raw', filepath)
    texts, labels = [], []
    with open(relative_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';') # splitting the sentence from the label (emotion)
            texts.append(text.lower())  # lowercasing - not really needed here, everything lowercase (safety)
            labels.append(label)
        return texts, labels

# Method for label encoding
def label_encoding(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

# Method for saving the processed data
def save_processed_data(dataset, file_name):

    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    processed_data_path = os.path.join(current_dir, '..', '..', 'data/processed', file_name + '_processed_data.pt')

    torch.save(dataset, processed_data_path)


# old code: 
# -----------------------------------------------
    """
    # Extract and save tokenized data and labels
    encodings = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]

    # Save tokenized data and labels
    processed_data = {
        'encodings': encodings,
        'labels': labels
    }

    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    processed_data_path = os.path.join(current_dir, '..', '..', 'data/processed', filepath + '_processed_data.pt')

    torch.save(processed_data, processed_data_path)
    """
# -----------------------------------------------


# Dataset class for consistency
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def check_processed_data_exists(filepath):
    return os.path.exists(filepath + '_data.pt')

def load_processed_data(filepath):
    # Load the saved tokenized texts and labels
    dataset = torch.load(filepath + '_data.pt')
    return dataset

def md():
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data/processed')
    print(processed_data_path)

    # Initializing the tokenizer for DistilBERT and tokenizing data
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if all(check_processed_data_exists(os.path.join(processed_data_path, dataset)) for dataset in ['train_processed'
                                                                                                   , 'val_processed'
                                                                                                   , 'test_processed']):
        print('!!!Loading processed data!!!')
        # Load processed data
        train_dataset = load_processed_data(os.path.join(processed_data_path, 'train_processed'))
        val_dataset = load_processed_data(os.path.join(processed_data_path, 'val_processed'))
        test_dataset = load_processed_data(os.path.join(processed_data_path, 'test_processed'))
    else:
        print('!!!Processing data!!!')

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

        # Create Dataset objects
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