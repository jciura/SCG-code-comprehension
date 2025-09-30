import json

from sentence_transformers import SentenceTransformer
from typing import Dict, List

example_questions: Dict[str, List[str]] = {
    "general": [
        "describe the project", "what is the system about", "what does it do",
        "how does the system work", "which modules exist", "what is the architecture",
        "what are the main components", "what is the overall flow", "how do modules interact",
        "what are the most important modules", "what can be improved in project"
    ],
    "medium": [
        "describe class", "describe method", "describe function", "where is method used",
        "where is class used", "where is function used", "where is variable used", "how is method used",
        "how is class used", "how is function used", "how is variable used", "what can be improved in class",
        "what can be improved in method"
    ],
    "specific": [
        "what are the parameters", "what does method return",
        "what does variable do", "how to call method", "how to call function"
    ]
}


def generate_and_save_classifier_embeddings(output_path: str) -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    example_embeddings = {
        label: model.encode(questions, convert_to_tensor=False).tolist()
        for label, questions in example_questions.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(example_embeddings, f, ensure_ascii=False, indent=2)


