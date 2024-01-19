import os

def copy_text_file(input_folder, output_folder, filename):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct full paths for input and output files
    input_file_path = os.path.join(input_folder, filename)
    output_file_path = os.path.join(output_folder, filename)

    # Read content from the input file
    with open(input_file_path, 'r') as input_file:
        file_content = input_file.read()

    # Write content to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write(file_content)
