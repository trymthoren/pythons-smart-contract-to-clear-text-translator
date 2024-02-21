import os

def create_project_structure(base_path):
    # Define the project structure
    project_structure = {
        "data": ["raw_code_snippets.sol", "annotated_descriptions.txt"],
        "models": ["translator_model.py", "trained_model.bin"],
        "utils": ["solidity_parser.py", "data_preparation.py"],
        "web3_interaction": ["contract_interaction.py"],
        "training": ["train_model.py"],
        "app": ["main.py", "requirements.txt"]
    }

    # Create directories and files
    for directory, files in project_structure.items():
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'w') as f:
                if file == "requirements.txt":
                    # Add placeholder requirements
                    f.write("web3\npandas\nmatplotlib\ntransformers\ntorch\n")
                else:
                    # Add a comment to indicate the purpose of the file
                    f.write(f"# Placeholder for {file}\n")

    print("Project structure created successfully.")

if __name__ == "__main__":
    # Specify the base path for the project
    base_path = "SolidityToText"
    create_project_structure(base_path)
