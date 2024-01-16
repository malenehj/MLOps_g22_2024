import torch
from mlops_g22_2024.data.make_dataset import md, EmotionDataset, label_encoding, check_processed_data_exists


def test_load_and_process_data():
        
    train_data_1, val_data_1, test_data_1 = md()

    # # Testing type:
    assert type(train_data_1) == EmotionDataset
    assert type(val_data_1) == EmotionDataset
    assert type(test_data_1) == EmotionDataset

    # Testing if the dataloaders are not empty:
    assert len(train_data_1) > 0
    assert len(val_data_1) > 0
    assert len(test_data_1) > 0

    # Testing if data processed above is loaded rather than processed again
    train_data_2, val_data_2, test_data_2 = md()
    assert torch.all(train_data_1[:]['input_ids'] == train_data_2[:]['input_ids'])
    assert torch.all(val_data_1[:]['input_ids'] == val_data_2[:]['input_ids'])
    assert torch.all(test_data_1[:]['input_ids'] == test_data_2[:]['input_ids'])

    # Test label encoding
def test_label_encoding():
    labels = ['happy', 'sad', 'angry']
    encoded_labels = label_encoding(labels)
    assert len(encoded_labels) == len(labels)

def test_check_processed_data_exists():
    # Test with a non-existing processed data file
    non_existing_file = 'non_existing_processed'
    non_existing_path = non_existing_file + '_data.pt'
    assert not check_processed_data_exists(non_existing_path)

