import os
import pytest

os.system('ls')

from mlops_g22_2024.data import make_dataset

def test_basic_data_import():
    D = make_dataset.md()
    print('this is a print')