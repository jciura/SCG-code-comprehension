import json

from loguru import logger

from testing.evaluator import RAGEvaluator
from testing.judge import judge_answer
from testing.token_counter import count_tokens



JUDGE_PROMPT = """You are an expert code reviewer.
Your task is to compare two answers to the same question.

Focus only on the quality of the answers as presented to the user.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Evaluate based on:
1. How well the answer addresses the user's question
2. Completeness 
3. Structure and readability
4. Logical flow

Scoring:
- Score each answer from 1 (very bad) to 5 (excellent).
- Choose "equal" only if the answers are comparable in overall quality.

Output JSON only

Output format:
{{"winner":"A|B|equal","score_a":1-5,"score_b":1-5,"reasoning":"short explanation"}}

"""


def compare_answers(question: str, answer_a: str, answer_b: str) -> dict:
    if not answer_a or not answer_b:
        return {"winner": "N/A", "score_a": 0, "score_b": 0, "reasoning": "Missing answer"}
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a[:4000],
        answer_b=answer_b[:4000])
    response = judge_answer(prompt)
    try:
        result = json.loads(response)
        return result
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response was: {response}")
        return {"winner": "N/A", "score_a": 0, "score_b": 0, "reasoning": f"Parse error: {str(e)}"}


def evaluate_all(ground_truth_file: str) -> None:
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    for q in ground_truth.get("questions", []):
        question = q["question"]
        claude = q.get("claude_stats", {}).get("answer", "")
        claude_context = (q.get("claude_stats", {})
                          .get("used_context", []))
        junie_mcp = (q.get("junie_stats", {})
                     .get("with_mcp", {})
                     .get("answer", ""))
        junie_mcp_context = (q.get("junie_stats", {})
                             .get("with_mcp", {})
                             .get("used_context", []))
        junie_nomcp = (q.get("junie_stats", {})
                       .get("without_mcp", {})
                       .get("answer", ""))
        q["comparisons"] = {}
        if claude and junie_mcp:
            result = compare_answers(question, claude, junie_mcp)
            result["winner"] = ({"A": "claude", "B": "junie_mcp"}
                                .get(result.get("winner"),
                            result.get("winner")))
            q["comparisons"]["claude_vs_junie_mcp"] = result
            logger.info(f"[{q['id']}] Claude vs Junie MCP: {result['winner']}")
        if junie_mcp and junie_nomcp:
            result = compare_answers(question, junie_mcp, junie_nomcp)
            result["winner"] = ({"A": "with_mcp", "B": "without_mcp"}
                                .get(result.get("winner"),
                                     result.get("winner")))
            q["comparisons"]["mcp_vs_no_mcp"] = result
        if claude:
            q["claude_stats"]["tokens"] = count_tokens(claude, "gpt5")
        if claude_context:
            if isinstance(claude_context, list):
                context_str = "\n".join(claude_context)
            else:
                context_str = str(claude_context)
            q["claude_stats"]["context_tokens"] = count_tokens(context_str, "gpt5")
        if junie_mcp:
            q["junie_stats"]["with_mcp"]["tokens"] = count_tokens(junie_mcp, "gpt5")
        if junie_mcp_context:
            if isinstance(junie_mcp_context, list):
                context_str = "\n".join(junie_mcp_context)
            else:
                context_str = str(junie_mcp_context)
            q["junie_stats"]["with_mcp"]["context_tokens"] = count_tokens(context_str, "gpt5")
        if junie_nomcp:
            q["junie_stats"]["without_mcp"]["tokens"] = count_tokens(junie_nomcp, "gpt5")

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


def evaluate_rag_metrics(ground_truth_file: str):
    evaluator = RAGEvaluator(ground_truth_file)
    with open("metrics_log.jsonl", "a", encoding="utf-8") as f_out:
        for q in evaluator.test_suite.questions:
            question = q.question
            key_entities = q.key_entities
            gt_contexts = q.ground_truth_contexts
            record = {
                "id": q.id,
                "question": question,
                "ragas": {}
            }
            claude_answer = q.claude_stats.answer if q.claude_stats else ""
            claude_context = q.claude_stats.used_context if q.claude_stats else []
            if claude_answer and claude_context:
                results = evaluator.rag_metrics.full_evaluation(
                    question, claude_answer, claude_context, gt_contexts, key_entities)
                record["ragas"]["claude"] = results
                logger.info(f"[{q.id}] Claude RAGAS: {results['ragas']}")
            junie_answer = q.junie_stats.with_mcp.answer if q.junie_stats else ""
            junie_context = q.junie_stats.with_mcp.used_context if q.junie_stats else []
            if junie_answer and junie_context:
                results = evaluator.rag_metrics.full_evaluation(
                    question, junie_answer, junie_context, gt_contexts, key_entities)
                record["ragas"]["junie_mcp"] = results
                logger.info(f"[{q.id}] Junie MCP RAGAS: {results['ragas']}")

            f_out.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # add_ground_context("ground_truth_killbill.json", "../../data/embeddings/node_embedding.json")
    evaluate_rag_metrics("ground_truth_killbill.json")
    # evaluate_all("ground_truth_killbill.json")
