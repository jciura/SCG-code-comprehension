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
3. A hallucination is information that is not present in the source context.

Input:
source context: {context}
question: {question}
answer: {answer}

Scoring scale:
0 - No hallucinations. All information in the answer is supported by the source context.
1 - Minor hallucinations. Small details added that are not in context but are reasonable inferences.
2 - Severe hallucinations. Major information not present in context.

Output: Return 0, 1 or 2, and describe hallucination"""

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
1 — Less than half of key facts is present in answer.

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
        key_facts = q.get("key_facts", [])
        ground_truth_context = q.get("ground_truth_contexts", [])
        ground_truth_context_str = "\n".join(ground_truth_context)[:8000]

        # WITH MCP
        mcp_stats = q.get("junie_stats", {}).get("with_mcp", {})
        mcp_answer = mcp_stats.get("answer", "")
        mcp_context = mcp_stats.get("used_context", [])
        if mcp_answer and mcp_answer.strip():
            logger.info(f"[{q['id']}] with MCP")
            context_str = "\n".join(mcp_context) if isinstance(mcp_context, list) else str(mcp_context)
            context_str = context_str[:8000]

            # Hallucination
            hallucination_prompt = JUDGE_HALLUCINATION_PROMPT.format(
                context=context_str, question=question, answer=mcp_answer)
            hallucination_response = judge_answer(hallucination_prompt)
            q["junie_stats"]["with_mcp"]["hallucination"] = hallucination_response
            logger.info(f"MCP Hallucination: {hallucination_response}")

            # Correctness
            key_facts_str = "\n".join(f"- {f}" for f in key_facts)
            correctness_prompt = JUDGE_CORRECTNESS_PROMPT.format(
                key_facts=key_facts_str, question=question, answer=mcp_answer)
            correctness_response = judge_answer(correctness_prompt)
            q["junie_stats"]["with_mcp"]["correctness"] = correctness_response
            logger.info(f"MCP Correctness: {correctness_response}")

            # Tokens
            q["junie_stats"]["with_mcp"]["tokens"] = count_tokens(mcp_answer, "gpt5")
        else:
            logger.warning(f"[{q['id']}] with MCP: no answer")

        # WITHOUT MCP
        nomcp_stats = q.get("junie_stats", {}).get("without_mcp", {})
        nomcp_answer = nomcp_stats.get("answer", "")

        if nomcp_answer and nomcp_answer.strip():
            logger.info(f"[{q['id']}] without MCP")

            # Hallucination
            hallucination_prompt = JUDGE_HALLUCINATION_PROMPT.format(
                context=ground_truth_context_str, question=question, answer=nomcp_answer)
            hallucination_response = judge_answer(hallucination_prompt)
            q["junie_stats"]["without_mcp"]["hallucination"] = hallucination_response
            logger.info(f"No MCP Hallucination: {hallucination_response}")

            # Correctness
            key_facts_str = "\n".join(f"- {f}" for f in key_facts)
            correctness_prompt = JUDGE_CORRECTNESS_PROMPT.format(
                key_facts=key_facts_str, question=question, answer=nomcp_answer)
            correctness_response = judge_answer(correctness_prompt)
            q["junie_stats"]["without_mcp"]["correctness"] = correctness_response
            logger.info(f"No MCP Correctness: {correctness_response}")

            # Tokens
            q["junie_stats"]["without_mcp"]["tokens"] = count_tokens(nomcp_answer, "gpt5")
        else:
            logger.warning(f"[{q['id']}] without MCP: no answer")

    with open(ground_truth_file, "w", encoding="utf-8") as f_out:
        json.dump(ground_truth, f_out, indent=2, ensure_ascii=False)


def get_ground_context(node: str, node_embedding_file):
    with open(node_embedding_file, "r", encoding="utf-8") as f_emb:
        embeddings = json.load(f_emb)
        for emb in embeddings:
            if emb["node"] == node:
                return emb["code"]
        return None


def add_ground_context(ground_truth_file: str, node_embedding_file: str) -> None:
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    questions = ground_truth.get("questions", [])

    for q in questions:
        if q["key_entities"] != [] and q["ground_truth_contexts"] == []:
            for entity in q["key_entities"]:
                context = get_ground_context(entity, node_embedding_file)
                if context:
                    q["ground_truth_contexts"].append(context)

    with open(ground_truth_file, "w", encoding="utf-8") as f_out:
        json.dump(ground_truth, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    add_ground_context("ground_truth_killbill.json", "../../data/embeddings/node_embedding.json")
    evaluate_rag_metrics("ground_truth_killbill.json")
    evaluate_answers("ground_truth_killbill.json")
