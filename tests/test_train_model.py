import os
from mlops_g22_2024.train_model import main

os.environ["HYDRA_FULL_ERROR"] = "1"

class config():

    def __init__(self):
        self.lr = 0.001
        self.train_batch_size = 32
        self.eval_batch_size = 64
        self.epochs = 5
        self.weight_decay = 0.01


def test_train_model_1():
    f = config()
    main(f)

"""
import os
from transformers import DistilBertTokenizerFast
from mlops_g22_2024.train_model import main

def test_train_model_saved():
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
    os.chdir("../mlops_g22_2024")
    # main()
    train_dict = {
    "lr": 2e-5,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "epochs": 2,
    "weight_decay": 0.01
}
    
    # print()
    # print()
    # print()
    # print("marioclioairoarioa")
    # print()
    # print()


    config_dict = {"train": train_dict}
    main(config_dict)


    # Check if the trained model exists
    model_path = '../models'
    assert os.path.exists(model_path)

"""