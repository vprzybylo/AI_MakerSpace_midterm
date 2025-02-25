# fine_tune_lama.py

import pandas as pd
import logging
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the embedding model
model_id = "Snowflake/snowflake-arctic-embed-l"  
logging.info(f"Loading model: {model_id}")
model = SentenceTransformer(model_id)

def load_synthetic_dataset():
    logging.info("Loading synthetic dataset...")
    df = pd.read_csv("../data/processed/synthetic_test_dataset.csv")
    # Convert to the format expected by the model
    examples = []
    for _, row in df.iterrows():
        examples.append(
            InputExample(texts=[row["user_input"], row["reference"]], label=1)
        )  # Assuming label 1 for positive pairs
    logging.info(f"Loaded {len(examples)} examples.")
    return examples

train_examples = load_synthetic_dataset()
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=[768, 512, 256, 128, 64]
)

EPOCHS = 1
warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

# Fine-tune the model
logging.info("Starting model training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path="data/processed/finetuned_arctic_ft",
    show_progress_bar=True,
)

logging.info("Model training completed.")