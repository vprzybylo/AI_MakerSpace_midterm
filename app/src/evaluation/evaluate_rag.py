import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

# Load environment variables
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

from embedding.model import EmbeddingModel
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag.chain import RAGChain
from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.testset import TestsetGenerator


def setup_rag(embedding_model_type):
    """Initialize RAG system for evaluation with specified embedding model."""
    logger.info("Setting up RAG system...")

    # Load documents
    data_path = root_dir / "data" / "raw" / "grid_code.pdf"
    if not data_path.exists():
        raise FileNotFoundError(f"PDF not found: {data_path}")

    loader = GridCodeLoader(str(data_path), pages=17)
    documents = loader.load_and_split()
    logger.info(f"Loaded {len(documents)} document chunks")

    # Initialize embedding model and vectorstore
    embedding_model = EmbeddingModel(model_type=embedding_model_type)
    vectorstore = VectorStore(embedding_model)
    vectorstore = vectorstore.create_vectorstore(documents)

    return RAGChain(vectorstore), documents


def generate_test_dataset(documents, n_questions=30):
    """Generate synthetic test dataset using RAGAS or load it if it already exists."""
    dataset_path = "../data/processed/synthetic_test_dataset.csv"

    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing synthetic test dataset from {dataset_path}...")
        return pd.read_csv(dataset_path)

    logger.info("Generating synthetic test dataset...")

    # Initialize the rate limiter
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # Make a request once every 1 second
        check_every_n_seconds=0.1,  # Check every 100 ms to see if allowed to make a request
        max_bucket_size=10,  # Controls the maximum burst size
    )

    # Initialize the chat model with the rate limiter
    model = init_chat_model("gpt-4o", temperature=0, rate_limiter=rate_limiter)

    # Initialize generator models
    generator_llm = LangchainLLMWrapper(model)
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create test set generator
    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings
    )

    # Generate synthetic test dataset
    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=n_questions
    )

    df = dataset.to_pandas()
    df.to_csv(dataset_path, index=False)  # Save as CSV
    logger.info(
        f"Generated synthetic dataset with {len(df)} test cases and saved to '{dataset_path}'."
    )
    return df


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def get_rag_response(rag_chain, question):
    """Get RAG response with retry logic"""
    return rag_chain.invoke(question)


def evaluate_rag_system(rag_chain, test_dataset):
    """Evaluate RAG system using RAGAS metrics"""
    logger.info("Starting RAGAS evaluation...")

    # Get RAG responses for each question
    eval_data = []

    # Iterate through DataFrame rows
    for _, row in test_dataset.iterrows():
        # Add delay between requests
        time.sleep(3)  # Wait 3 seconds between requests
        response = get_rag_response(rag_chain, row["user_input"])
        eval_data.append(
            {
                "user_input": row["user_input"],
                "response": response["answer"],
                "retrieved_contexts": [doc.page_content for doc in response["context"]],
                "ground_truth": row["reference"],  # Keep for faithfulness
                "reference": row["reference"],  # Keep for context_recall
            }
        )
        logger.info(f"Processed question: {row['user_input'][:50]}...")

    # Convert to pandas then to EvaluationDataset
    eval_df = pd.DataFrame(eval_data)
    logger.info("Sample evaluation data:")
    logger.info(eval_df.iloc[0].to_dict())
    eval_dataset = EvaluationDataset.from_pandas(eval_df)

    # Initialize RAGAS evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

    custom_run_config = RunConfig(timeout=360, max_workers=32)

    # Run evaluation
    results = evaluate(
        eval_dataset,
        metrics=[
            Faithfulness(),  # Measures how accurately the generated response reflects the ground truth.
            AnswerRelevancy(),  # Assesses the relevance of the answer to the user's question.
            ContextRecall(),  # Evaluates the ability of the model to retrieve relevant context from the documents.
            ContextPrecision(),  # Measures the precision of the retrieved contexts in relation to the user's question.
        ],
        llm=evaluator_llm,
        run_config=custom_run_config,
    )

    return results


def run_evaluation_with_model(rag_chain, test_dataset, embedding_model_type):
    """Run evaluation with the specified embedding model type."""
    logger.info(f"Running evaluation with {embedding_model_type} embeddings...")

    # Run evaluation
    results = evaluate_rag_system(rag_chain, test_dataset)

    logger.info(f"Evaluation Results for {embedding_model_type}:")
    logger.info(results)

    # Save results to CSV
    results_path = Path("../data/processed/")
    results_path.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame([results])
    results_df.to_csv(
        results_path / f"evaluation_results_{embedding_model_type}.csv", index=False
    )
    logger.info(
        f"Saved evaluation results to evaluation_results_{embedding_model_type}.csv"
    )


def main():
    """Main evaluation script"""
    logger.info("Starting RAG evaluation")

    try:
        # Setup RAG system with the fine-tuned embedding model
        rag_chain_finetuned, documents = setup_rag("finetuned")

        # Generate synthetic test dataset
        test_dataset = generate_test_dataset(documents)

        # Run evaluations with both embedding models
        run_evaluation_with_model(rag_chain_finetuned, test_dataset, "finetuned")

        # # Setup RAG system with the OpenAI embedding model
        # rag_chain_openai, _ = setup_rag("openai")

        # # Run evaluation with OpenAI embeddings
        # run_evaluation_with_model(rag_chain_openai, test_dataset, "openai")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
