import os
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data

# If there are no raw data for the tests, create a raw folder with the pytest_data
in_folder = 'data/pytest_data'
out_folder = 'data/raw'
if not os.path.exists(out_folder) or len(os.listdir(out_folder)) == 0:
    from tests.copy_txt import copy_text_file
    copy_text_file(in_folder, out_folder, 'train.txt')
    copy_text_file(in_folder, out_folder, 'val.txt')
    copy_text_file(in_folder, out_folder, 'test.txt')