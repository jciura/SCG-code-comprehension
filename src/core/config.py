import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

default_classifier_model = "sentence-transformers/all-MiniLM-L6-v2"
default_chroma_path = os.path.join(PROJECT_ROOT, "data", "embeddings", "chroma_storage")
default_collection_name = "scg_embeddings"

MODEL_NAME = "gemini-2.5-flash"
NODE_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_embedding.json")
NODE_CONTEXT_HISTORY = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_context_history.json")
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
METRICS = "metrics_log_spark.jsonl"
JUNIE_URL = "http://127.0.0.1:8000/ask_junie"
partition = "../../data/graph/partition.js"
scg_test = "../../data/graph/scgTest.gdf"
ccn_test = "../../data/graph/ccnTest.gdf"
COMBINED_MAX = 997160

projects = os.path.join(PROJECT_ROOT, "projects")
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"

RAGAS_MODEL = "gpt-4o-mini"
RAGAS_TIMEOUT = 280.0
RAGAS_MAX_TOKENS = 12000

JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_MAX_TOKENS = 500