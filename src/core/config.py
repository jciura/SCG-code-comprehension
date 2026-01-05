import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

default_chroma_path = os.path.join(PROJECT_ROOT, "data", "embeddings", "chroma_storage")
default_collection_name = "scg_embeddings"

MODEL_NAME = "gemini-2.5-flash"
NODE_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_embedding.json")
NODE_CONTEXT_HISTORY = os.path.join(PROJECT_ROOT, "data", "embeddings", "node_context_history.json")
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
METRICS = "metrics_log_spark.jsonl"
JUNIE_URL = "http://127.0.0.1:8000/ask_junie"
COMBINED_MAX = 4256


GPT_MODEL = "gpt-4o-mini"
RAGAS_TIMEOUT = 280.0
RAGAS_MAX_TOKENS = 12000

CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 500

HTTP_TIMEOUT = 200.0
CORS_ORIGINS = ["http://localhost:3000"]
RAG_TIMEOUT = 60.0

SCG_OUTPUT_FILE = "scgTest.gdf"
CRUCIAL_OUTPUT_FILE = "crucial.html"
PARTITION_OUTPUT_FILE = "partition.js"
CCN_OUTPUT_FILE = "ccnTest.gdf"

GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graph")
scg_test = os.path.join(GRAPH_DIR, SCG_OUTPUT_FILE)
ccn_test = os.path.join(GRAPH_DIR, CCN_OUTPUT_FILE)
partition = os.path.join(GRAPH_DIR, PARTITION_OUTPUT_FILE)

GEMINI_RATE_LIMIT_DELAY = 5
SERVER_URL = "http://127.0.0.1:8000"

LLAMA_TOKENIZER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
