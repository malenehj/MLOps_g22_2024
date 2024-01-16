from mlops_g22_2024.data.make_dataset import md, EmotionDataset

def test_process_data():
        
    train_data, val_data, test_data = md()

    # # Testing type:
    assert type(train_data) == EmotionDataset
    assert type(val_data) == EmotionDataset
    assert type(test_data) == EmotionDataset

    # Testing if the dataloaders are not empty:
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0