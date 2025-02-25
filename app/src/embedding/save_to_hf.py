### `save_to_hf.py`

import logging
import os

from huggingface_hub import HfApi, Repository

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def push_model_to_huggingface(model_dir, model_name, hf_username):
    """Push the model to Hugging Face Hub using the Repository class."""
    try:
        # Create a new directory for the repository
        repo_dir = f"./{model_name}_repo"  # Specify a new directory
        os.makedirs(repo_dir, exist_ok=True)

        # Initialize the repository
        repo_id = f"{hf_username}/{model_name}"
        repo = Repository(local_dir=repo_dir, clone_from=repo_id)

        # Copy model files to the new repository directory
        for filename in os.listdir(model_dir):
            full_file_name = os.path.join(model_dir, filename)
            if os.path.isfile(full_file_name):
                os.rename(full_file_name, os.path.join(repo_dir, filename))

        # Add model files to the repository
        repo.git_add()
        repo.git_commit("Add custom segmentation model")
        repo.git_push()

        logging.info(f"Model pushed to Hugging Face Hub: {repo_id}")

    except Exception as e:
        logging.error(f"Failed to push model to Hugging Face Hub: {str(e)}")


if __name__ == "__main__":
    # Define parameters
    model_directory = (
        "src/data/processed/finetuned_arctic_ft"  # Directory where the model is saved
    )
    model_name = "finetuned_arctic_ft"  # Name for the model on Hugging Face
    hf_username = "vanessaprzybylo"  # Replace with your Hugging Face username

    # Push the model to Hugging Face
    push_model_to_huggingface(model_directory, model_name, hf_username)
