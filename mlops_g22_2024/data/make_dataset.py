'''
make_dataset.py loads the test.txt, train.txt and val.txt datasets from the .txt files
and uses a transformer (AutoTokanizer) to tokanize the data
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer
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
def save_processed_data(dataset, filepath):
    # Extract and save tokenized data and labels
    tokenized_texts = []
    labels = []

    for item in dataset:
        tokenized_texts.append(item[0])  # item[0] contains the tokenized text
        labels.append(item[1])           # item[1] contains the label

    # Save data
    current_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    relative_path = os.path.join(current_dir, '..', '..', 'data/processed', filepath)
    torch.save(tokenized_texts, relative_path + '_data.pt')
    torch.save(labels, relative_path + '_labels.pt')

# Dataset class for consistency
class EmotionDataset(Dataset):
    # Constructor
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    # Returns the number of items in the dataset
    def __len__(self):
        return len(self.texts)
    # Returns a specified item with id 'idx'
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return encoding, label

def check_processed_data_exists(filepath):
    return os.path.exists(filepath + '_data.pt') and os.path.exists(filepath + '_labels.pt')

def load_processed_data(filepath):
    # Load the saved tokenized texts and labels
    tokenized_texts = torch.load(filepath + '_data.pt')
    labels = torch.load(filepath + '_labels.pt')

    # Since the tokenized texts are saved as a list of dictionary-like objects,
    # you need to convert them to the proper tensor format for use in a TensorDataset.
    input_ids = torch.stack([tt['input_ids'].squeeze(0) for tt in tokenized_texts])
    attention_masks = torch.stack([tt['attention_mask'].squeeze(0) for tt in tokenized_texts])
    labels = torch.tensor(labels)

    # Create a TensorDataset from the loaded data
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset

def md():
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data/processed')

    if all(check_processed_data_exists(os.path.join(processed_data_path, dataset)) for dataset in
           ['train', 'val', 'test']):
        # Load processed data
        train_dataset = load_processed_data(os.path.join(processed_data_path, 'train'))
        val_dataset = load_processed_data(os.path.join(processed_data_path, 'val'))
        test_dataset = load_processed_data(os.path.join(processed_data_path, 'test'))

        # Create DataLoader objects
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        # Loading the data
        train_texts, train_labels = load_data('train.txt')
        train_labels = label_encoding(train_labels)

        # Load the validation data
        val_texts, val_labels = load_data('val.txt')
        val_labels = label_encoding(val_labels)

        # Load the test data
        test_texts, test_labels = load_data('test.txt')
        test_labels = label_encoding(test_labels)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Create Dataset and DataLoader for the training set
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create Dataset and DataLoader for the validation set
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create Dataset and DataLoader for the test set
        test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Save processed data
        save_processed_data(train_dataset, 'train')
        save_processed_data(val_dataset, 'val')
        save_processed_data(test_dataset, 'test')

        return train_dataloader, val_dataloader, test_dataloader

md()

if __name__ == '__main__':
    # Get the data and process it
    pass