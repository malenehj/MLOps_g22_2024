import os
import pytest

from torch.utils.data import DataLoader
from mlops_g22_2024.data import make_dataset

# base data-sets fixture, for use in tests below
@pytest.fixture()
def data_loaders():
    return make_dataset.md()

def test_basic_data_import(data_loaders):
    train_dataloader, val_dataloader, test_dataloader = data_loaders

    # Testing if the dataloaders are of type DataLoader:
    assert type(train_dataloader) == DataLoader
    assert type(val_dataloader) == DataLoader
    assert type(test_dataloader) == DataLoader

    # Testing if the dataloaders are not empty:
    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0
    assert len(test_dataloader) > 0