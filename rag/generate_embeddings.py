import torch
from transformers import AutoTokenizer, AutoModel
import json

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(texts, MODEL_NAME, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
        batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings


def code_embeddings(chunks, output_path, MODEL_NAME):
    if not chunks:
        print("Warning: input chunks list is empty.")
        return

    chunk_contents = []
    for chunk in chunks:
        chunk_contents.append(chunk["content"])

    embeddings = generate_embeddings(chunk_contents, MODEL_NAME)
    with open(output_path, "w") as out_f:
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk["content"],
                "embedding": embeddings[i].tolist(),
                "metadata": {
                    "file": chunk.get("file"),
                    "type": chunk.get("type"),
                    "class": chunk.get("class"),
                    "method": chunk.get("method")
                }
            }
            out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")


def questions_embeddings(qa_data, output_path, MODEL_NAME):
    questions = [qa["question"] for qa in qa_data]
    all_embeddings = generate_embeddings(questions, MODEL_NAME)

    for i, qa in enumerate(qa_data):
        qa["embedding"] = all_embeddings[i].tolist()

    with open(output_path, "w") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)


def load_json(file_path, encoding="utf-8-sig"):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)

if __name__ == "__main__":
    MODEL_NAME = "microsoft/codebert-base"

    # questions
    qa_input_path = "../data/q&a_project.json"
    qa_output_path = "../embeddings/qa_embedding_project_fixed.json"
    qa_data = load_json(qa_input_path, 'ISO-8859-1')
    questions_embeddings(qa_data, qa_output_path, MODEL_NAME)

    # code
    code_input_path = "../data/project_chunks_fixed.json"
    code_output_path = "../embeddings/code_embedding_project_fixed.jsonl"
    # Na razie robię embedding 50 chunków testowo
    code_chunks = load_json(code_input_path)[:50]
    code_embeddings(code_chunks, code_output_path, MODEL_NAME)
