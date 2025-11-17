from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import re
from src.core.config import embedding_model


class RAGMetrics:

    def __init__(self, embedding_model_name: str = embedding_model):
        """
            Initializes the metrics engine with a sentence-embedding model.

            Args:
                embedding_model_name (str): Name/ID of the embedding model to load.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def context_precision(self, question: str, retrieved_contexts: List[str],ground_truth_answer: str, k: int = None) -> float:
        """
            Estimates precision of retrieved contexts vs. ground-truth answer.

            Counts the fraction of top-k retrieved contexts that are semantically
            similar to the ground-truth answer above a threshold.

            Args:
                question (str): (Unused here, reserved for future variants.)
                retrieved_contexts (List[str]): Retrieved context snippets.
                ground_truth_answer (str): Reference answer text.
                k (int, optional): Number of contexts to consider (defaults to all).

            Returns:
                float: Precision in [0, 1].
        """
        if not retrieved_contexts or not ground_truth_answer:
            return 0.0
        k = k or len(retrieved_contexts)
        contexts_to_check = retrieved_contexts[:k]
        gt_answer_emb = self.embedding_model.encode([ground_truth_answer])
        relevant_count = 0
        threshold = 0.4
        for context in contexts_to_check:
            context_emb = self.embedding_model.encode([context])
            similarity = cosine_similarity(gt_answer_emb, context_emb)[0][0]
            if similarity > threshold:
                relevant_count += 1
        precision = relevant_count / k
        # logger.debug(f"Context Precision: {precision:.3f}")
        return precision

    def context_recall(self, retrieved_contexts: List[str], ground_truth_contexts: List[str], adaptive_threshold: bool = True) -> float:
        """
            Estimates recall of retrieved contexts against ground-truth contexts.

            Uses cosine similarity of sentence embeddings; optionally adapts the
            similarity threshold from the distribution of pairwise scores.

            Args:
                retrieved_contexts (List[str]): Retrieved context snippets.
                ground_truth_contexts (List[str]): Expected/ideal context snippets.
                adaptive_threshold (bool): If True, choose a percentile-based threshold.

            Returns:
                float: Recall in [0, 1].
        """
        if not ground_truth_contexts:
            return 1.0
        if not retrieved_contexts:
            return 0.0
        gt_embeddings = self.embedding_model.encode(ground_truth_contexts)
        retrieved_embeddings = self.embedding_model.encode(retrieved_contexts)
        found_count = 0
        if adaptive_threshold:
            all_similarities = cosine_similarity(gt_embeddings, retrieved_embeddings)
            threshold = np.percentile(all_similarities, 60)
            threshold = max(0.6, min(0.8, threshold))
        else:
            threshold = 0.7
        for gt_emb in gt_embeddings:
            similarities = cosine_similarity([gt_emb], retrieved_embeddings)[0]
            if max(similarities) > threshold:
                found_count += 1
        recall = found_count / len(ground_truth_contexts)
        # logger.debug(f"Context Recall: {recall:.3f}")
        return recall

    def faithfulness(self, answer: str, retrieved_contexts: List[str], use_llm_decomposition: bool = None) -> float:
        """
            Scores whether answer claims are supported by the retrieved context.

            Decomposes the answer into claims (bullets, numbered items, sentences),
            then checks each claim's embedding similarity to the concatenated context.

            Args:
                answer (str): Generated answer text.
                retrieved_contexts (List[str]): Context used to produce the answer.
                use_llm_decomposition (bool, optional): Reserved for future LLM-based parsing.

            Returns:
                float: Faithfulness score in [0, 1].
        """
        if not answer.strip() or not retrieved_contexts:
            return 0.0
        claims = self._decompose_answer(answer)
        logger.info(claims)
        if not claims:
            return 0.0
        context_embeddings = [self.embedding_model.encode([ctx]) for ctx in retrieved_contexts]
        faithful_count = 0
        threshold = 0.45
        for claim in claims:
            logger.info(f"Claim: {claim}")
            claim_emb = self.embedding_model.encode([claim])
            max_sim = max(cosine_similarity(claim_emb, ctx_emb)[0][0] for ctx_emb in context_embeddings)
            logger.info("Max similarity: {:.4f}".format(max_sim))
            if max_sim > threshold:
                faithful_count += 1
        faithfulness_score = faithful_count / len(claims)
        return faithfulness_score

    def answer_relevance(self, question: str, answer: str) -> float:
        """
            Measures semantic similarity between the question and the answer.

            Args:
                question (str): Input question.
                answer (str): Generated answer.

            Returns:
                float: Relevance score in [0, 1].
        """
        if not answer.strip():
            return 0.0
        question_emb = self.embedding_model.encode([question])
        answer_emb = self.embedding_model.encode([answer])
        similarity = cosine_similarity(question_emb, answer_emb)[0][0]
        relevance = max(0.0, min(1.0, float(similarity)))
        # logger.debug(f"Answer Relevance: {relevance:.3f}")
        return relevance

    def context_entity_recall(self, retrieved_contexts: List[str], key_entities: List[str]) -> float:
        """
            Checks whether key entities appear within the retrieved context.

            Args:
                retrieved_contexts (List[str]): Retrieved contexts.
                key_entities (List[str]): Entities expected to be present.

            Returns:
                float: Fraction of entities found in [0, 1].
        """
        if not key_entities:
            return 1.0
        retrieved_text = " ".join(retrieved_contexts).lower()
        found_entities = sum(1 for entity in key_entities if entity.lower() in retrieved_text)
        entity_recall = found_entities / len(key_entities)
        # logger.debug(f"Context Entity Recall: {entity_recall:.3f}")
        return entity_recall

    def answer_semantic_similarity(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
            Measures semantic similarity between generated and ground-truth answers.

            Args:
                generated_answer (str): Model-produced answer.
                ground_truth_answer (str): Reference answer.

            Returns:
                float: Similarity score in [0, 1].
        """
        if not generated_answer.strip() or not ground_truth_answer.strip():
            return 0.0
        gen_emb = self.embedding_model.encode([generated_answer])
        gt_emb = self.embedding_model.encode([ground_truth_answer])
        similarity = cosine_similarity(gen_emb, gt_emb)[0][0]
        similarity_score = max(0.0, min(1.0, float(similarity)))
        # logger.debug(f"Answer Semantic Similarity: {similarity_score:.3f}")
        return similarity_score


    def _decompose_answer(self, answer: str) -> List[str]:
        """
            Splits an answer into claim-like spans for faithfulness checking.

            Parses bullet points, numbered items, and remaining sentences, then
            returns a list of unique, minimally long claims.

            Args:
                answer (str): Raw answer text.

            Returns:
                List[str]: Extracted claims.
        """
        claims: List[str] = []
        bullet_re = re.compile(r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\n|$)', re.MULTILINE | re.DOTALL)
        numbered_re = re.compile(r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\n|$)', re.MULTILINE | re.DOTALL)
        try:
            bullet_items = bullet_re.findall(answer)
        except Exception:
            bullet_items = []
        for item in bullet_items:
            txt = item.strip()
            if len(txt) > 10:
                claims.append(txt)
        try:
            numbered_items = numbered_re.findall(answer)
        except Exception:
            numbered_items = []
        for item in numbered_items:
            txt = item.strip()
            if len(txt) > 10:
                claims.append(txt)
        text_no_lists = bullet_re.sub('', answer)
        text_no_lists = numbered_re.sub('', text_no_lists)
        sentence_candidates = re.split(r'(?<=[.!?])\s+', text_no_lists)
        for s in sentence_candidates:
            s = s.strip()
            if len(s) > 15:
                claims.append(s)
        unique_claims: List[str] = []
        for c in claims:
            if len(c) > 10 and c not in unique_claims:
                unique_claims.append(c)
        return unique_claims


    def ragas_score(self, question: str, answer: str, retrieved_contexts: List[str],
            ground_truth_answer: str, ground_truth_contexts: List[str]) -> Dict[str, float]:
        """
            Computes a simple RAGAS-like aggregate with component metrics.

            Components:
              - context_precision
              - context_recall
              - faithfulness
              - answer_relevance

            The aggregate is the harmonic-like mean (4 / sum(1/x)) if all components
            are positive; otherwise 0.0.

            Args:
                question (str): Original user question.
                answer (str): Generated answer.
                retrieved_contexts (List[str]): Retrieved context snippets.
                ground_truth_answer (str): Reference answer text.
                ground_truth_contexts (List[str]): Reference context snippets.

            Returns:
                Dict[str, float]: Component scores and overall 'ragas_score'.
        """
        context_prec = self.context_precision(question, retrieved_contexts, ground_truth_answer)
        context_rec = self.context_recall(retrieved_contexts, ground_truth_contexts)
        faith = self.faithfulness(answer, retrieved_contexts)
        ans_rel = self.answer_relevance(question, answer)
        scores = [context_prec, context_rec, faith, ans_rel]
        ragas = 0.0
        all_positive = True
        for s in scores:
            if s <= 0:
                all_positive = False
                break
        if all_positive:
            total = 0
            for s in scores:
                total += 1 / s
            ragas = 4 / total

        return {
            "context_precision": round(context_prec, 3),
            "context_recall": round(context_rec, 3),
            "faithfulness": round(faith, 3),
            "answer_relevance": round(ans_rel, 3),
            "ragas_score": round(ragas, 3)
        }


    def full_evaluation(self, question: str, answer: str, retrieved_contexts: List[str],
                        ground_truth_answer: str, ground_truth_contexts: List[str], key_entities: List[str] = None, key_facts: List[str] = None) -> Dict[str, Any]:
        """
            Runs a complete evaluation and returns component + optional metrics.

            Args:
                question (str): Original question.
                answer (str): Generated answer to evaluate.
                retrieved_contexts (List[str]): Context snippets given to the model.
                ground_truth_answer (str): Reference answer.
                ground_truth_contexts (List[str]): Reference contexts.
                key_entities (List[str], optional): Entities expected in context.
                key_facts (List[str], optional): (Reserved) Expected facts.

            Returns:
                Dict[str, Any]: {
                    "ragas": {...component scores...},
                    "optional_metrics": {
                        "context_entity_recall": float (if key_entities),
                        "answer_semantic_similarity": float
                    }
                }
        """
        ragas = self.ragas_score(question, answer, retrieved_contexts, ground_truth_answer, ground_truth_contexts)
        optional = {}
        if key_entities:
            optional["context_entity_recall"] = round(self.context_entity_recall(retrieved_contexts, key_entities), 3)
        optional["answer_semantic_similarity"] = round(self.answer_semantic_similarity(answer, ground_truth_answer), 3)
        return {
            "ragas": ragas,
            "optional_metrics": optional,}
