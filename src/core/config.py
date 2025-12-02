import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

default_classifier_embeddings_path = os.path.join(
    PROJECT_ROOT, "data", "embeddings", "classifier_example_embeddings.json"
)
default_classifier_model = "sentence-transformers/all-MiniLM-L6-v2"
default_chroma_path = os.path.join(PROJECT_ROOT, "data", "embeddings", "chroma_storage")
default_collection_name = "scg_embeddings"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
NODE_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_embedding.json")
NODE_CONTEXT_HISTORY = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_context_history.json")
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
METRICS = "src/logs/metrics_log.json"
JUNIE_URL = "http://127.0.0.1:8000/ask_junie"
partition = "../../data/graph/partition.js"
scg_test = "../../data/graph/scgTest.gdf"
ccn_test = "../../data/graph/ccnTest.gdf"

projects = os.path.join(PROJECT_ROOT, "projects")
ground_truth = "src/core/ground_truth.json"
metrics_path = "src/logs/metrics_log.jsonl"
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
