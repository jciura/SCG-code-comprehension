import json
from loguru import logger

from testing.evaluator import RAGEvaluator
from testing.judge import judge_answer
from testing.token_counter import count_tokens

JUDGE_HALLUCINATION_PROMPT = """You are a strict judge. Your task is to 
evaluate if the answer contains any hallucinations.

Judging rules:
1. Use only the information present in the source context.
2. Do not use external knowledge or assume missing details.

Input:
source context: {context}
question: {question}
answer: {answer}

Scoring scale:
1 - The answer contains hallucinations (even if only one).
0 - The answer doesn't contain hallucinations.

Output: Return 0 and 1, and describe hallucination"""

JUDGE_CORRECTNESS_PROMPT = """You are a strict judge. Your task is to 
evaluate the answer to the question based only on the provided key facts that need to be included in the answer.

Judging rules:
1. Use only the information present in the source context.
2. Do not use external knowledge or assume missing details.

Input:
key facts: {key_facts}
question: {question}
answer: {answer}

Scoring scale:
4 — Fully correct, all key facts are present in answer.
3 — Most of key facts is present in answer.
2 — Only half of key facts is present in answer.
1 — Less tham half of key facts is present in answer.

Output: Return only integer from 1 to 4."""


def evaluate_rag_metrics(ground_truth_file: str):
    evaluator = RAGEvaluator(ground_truth_file)

    with open("metrics_log.jsonl", "a", encoding="utf-8") as f_out:
        for q in evaluator.test_suite.questions:
            question = q.question
            answer = q.junie_stats.with_mcp.answer
            retrieved_contexts = q.junie_stats.with_mcp.used_context
            key_entities = q.key_entities
            gt_contexts = q.ground_truth_contexts

            results = evaluator.rag_metrics.full_evaluation(
                question, answer, retrieved_contexts, gt_contexts, key_entities)

            record = {
                "id": q.id,
                "question": question,
                "answer": answer,
                "results": results
            }

            f_out.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def evaluate_answers(ground_truth_file: str) -> None:
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    questions = ground_truth.get("questions", [])

    for q in questions:
        question = q["question"]
        junie_stats = q.get("junie_stats", {})
        key_facts = q.get("key_facts", [])

        mcp_stats = junie_stats.get("with_mcp", {})
        nomcp_stats = junie_stats.get("without_mcp", {})

        mcp_answer = mcp_stats.get("answer", "")
        nomcp_answer = nomcp_stats.get("answer", "")
        mcp_context = mcp_stats.get("used_context", [])

        mcp_hallucination_prompt = JUDGE_HALLUCINATION_PROMPT.format(
            context=mcp_context,
            question=question,
            answer=mcp_answer)

        response = judge_answer(mcp_hallucination_prompt)
        logger.info(f"Hallucination response: {response}")

        q["junie_stats"]["with_mcp"]["hallucination"] = response

        mcp_correctness_prompt = JUDGE_CORRECTNESS_PROMPT.format(
            key_facts=key_facts,
            question=question,
            answer=mcp_answer)

        response = judge_answer(mcp_correctness_prompt)
        logger.info(f"MCP correctness response: {response}")

        q["junie_stats"]["with_mcp"]["correctness"] = response

        nomcp_correctness_prompt = JUDGE_CORRECTNESS_PROMPT.format(
            key_facts=key_facts,
            question=question,
            answer=nomcp_answer)

        # Zakomentowane bo jeszcze nie ma odpowiedzi
        # response = judge_answer(nomcp_correctness_prompt)
        logger.info(f"No MCP correctness response: {response}")

        q["junie_stats"]["without_mcp"]["correctness"] = response

        q["junie_stats"]["with_mcp"]["tokens"] = count_tokens(mcp_answer, "gpt5")
        # q["junie_stats"]["without_mcp"]["tokens"] = count_tokens(mcp_answer, "gpt5")

    with open(ground_truth_file, "w", encoding="utf-8") as f_out:
        json.dump(ground_truth, f_out, indent=2, ensure_ascii=False)


def get_ground_context(label: str, node_embedding_file):
    with open(node_embedding_file, "r", encoding="utf-8") as f_emb:
        embeddings = json.load(f_emb)
        for emb in embeddings:
            if emb["label"] == label:
                return emb["code"]
        return None


def add_ground_context(ground_truth_file: str, node_embedding_file: str) -> None:
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    questions = ground_truth.get("questions", [])

    for q in questions:
        if q["key_entities"] != [] and q["ground_truth_contexts"] == []:
            for entity in q["key_entities"]:
                q["ground_truth_contexts"].append(get_ground_context(entity, node_embedding_file))

    with open(ground_truth_file, "w", encoding="utf-8") as f_out:
        json.dump(ground_truth, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    add_ground_context("ground_truth.json", "../../data/embeddings/node_embedding.json")
    evaluate_rag_metrics("ground_truth.json")
    evaluate_answers("ground_truth.json")
