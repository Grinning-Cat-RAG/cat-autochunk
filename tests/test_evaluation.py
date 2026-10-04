import asyncio
import json
import random

import numpy as np
from langchain_core.documents import Document

try:
    from .. import evaluation
except ImportError:  # pytest (see conftest.py)
    from autochunk_plugin import evaluation


def test_normalize_and_span_coverage():
    assert evaluation.normalize_text("  Hello\n\n  World ") == "hello world"
    assert evaluation.span_coverage("lazy dog", "the quick fox jumps over the lazy dog") == 1.0
    # half of the span is in the text (the span is split across two chunks)
    assert abs(evaluation.span_coverage("abcdefgh", "xxabcd") - 0.5) < 1e-9
    assert evaluation.span_coverage("", "anything") == 0.0


def test_parse_question_reply():
    passage = "The Cheshire Cat is a fictional cat popularised by Lewis Carroll in Alice's Adventures in Wonderland."

    reply = 'Sure! {"question": "Who popularised the Cheshire Cat?", "answer_excerpt": "popularised by Lewis Carroll"}'
    parsed = evaluation.parse_question_reply(reply, passage)
    assert parsed == {"question": "Who popularised the Cheshire Cat?", "excerpt": "popularised by lewis carroll"}

    # the excerpt is not in the passage
    reply = json.dumps({"question": "Who wrote Hamlet in England?", "answer_excerpt": "William Shakespeare wrote it"})
    assert evaluation.parse_question_reply(reply, passage) is None

    # not a JSON object / too short
    assert evaluation.parse_question_reply("I cannot answer", passage) is None
    assert evaluation.parse_question_reply('{"question": "Who?", "answer_excerpt": "Lewis Carroll"}', passage) is None


def test_llm_reply_to_text():
    class Message:
        content = [{"type": "text", "text": "a"}, "b"]

    assert evaluation.llm_reply_to_text("plain") == "plain"
    assert evaluation.llm_reply_to_text(Message()) == "ab"


def test_truncate_docs_and_pick_passages():
    docs = [Document(page_content="word " * 400, metadata={"page": 1}), Document(page_content="other " * 400)]
    truncated = evaluation.truncate_docs(docs, 2500)
    assert sum(len(d.page_content) for d in truncated) == 2500
    assert truncated[0].metadata == {"page": 1}

    sample = evaluation.SampleFile(name="f.txt", docs=truncated)
    passages = evaluation.pick_passages(sample, 2, random.Random(1))
    assert 1 <= len(passages) <= 2
    assert all(evaluation.MIN_PASSAGE_CHARS <= len(p) <= evaluation.PASSAGE_CHARS for p in passages)

    assert evaluation.pick_passages(evaluation.SampleFile(name="e", docs=[Document(page_content="short")]), 2, random.Random(1)) == []


def test_generate_questions_keeps_only_verbatim_excerpts():
    text = " ".join(f"Sentence number {i} talks about the topic {i} in detail." for i in range(200))
    sample = evaluation.SampleFile(name="a.txt", docs=[Document(page_content=text)])
    calls = []

    async def ask_llm(prompt: str) -> str:
        calls.append(prompt)
        passage = prompt.split('"""')[1].strip()
        if len(calls) % 2:
            excerpt = " ".join(passage.split()[5:15])
            return json.dumps({"question": "What does the sentence say about the topic?", "answer_excerpt": excerpt})
        return json.dumps({"question": "What is invented here?", "answer_excerpt": "this text is not in the passage at all"})

    questions = asyncio.run(evaluation.generate_questions([sample], ask_llm, 4, random.Random(0)))
    assert len(calls) == 4
    assert len(questions) == 2
    assert all(q.source == "a.txt" for q in questions)


class _BagOfWordsEmbedder:
    """Deterministic embedder: hashed bag of words."""
    dimension = 256

    def __init__(self):
        self.calls = 0

    async def embed_documents(self, texts):
        self.calls += len(texts)
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension)
            for word in evaluation.normalize_text(text).split():
                vector[hash(word.strip(".,")) % self.dimension] += 1
            vectors.append(vector.tolist())
        return vectors


def _fixed_split(chunks_by_source):
    async def split(chunker, docs):
        source = docs[0].metadata["source"]
        return [Document(page_content=c, metadata={"source": source}) for c in chunks_by_source[source]]
    return split


def test_chunker_evaluator_mrr_and_penalty():
    samples = [
        evaluation.SampleFile(name="cats.txt", docs=[Document(page_content="x", metadata={"source": "cats.txt"})]),
        evaluation.SampleFile(name="dogs.txt", docs=[Document(page_content="x", metadata={"source": "dogs.txt"})]),
    ]
    chunks = {
        "cats.txt": ["cats purr when they are happy", "cats sleep sixteen hours a day"],
        "dogs.txt": ["dogs bark at the mailman", "dogs love long walks in the park"],
    }
    questions = [
        evaluation.SyntheticQuestion(question="why do cats purr happy", excerpt="cats purr when they are happy", source="cats.txt"),
        evaluation.SyntheticQuestion(question="dogs walks park", excerpt="love long walks in the park", source="dogs.txt"),
        # excerpt not in any chunk: reciprocal rank 0
        evaluation.SyntheticQuestion(question="parrots talk", excerpt="parrots can talk a lot", source="cats.txt"),
    ]
    embedder = _BagOfWordsEmbedder()
    evaluator = evaluation.ChunkerEvaluator(
        samples=samples,
        questions=questions,
        cache=evaluation.EmbeddingCache(embed_documents=embedder.embed_documents),
        split=_fixed_split(chunks),
        top_k=3,
        size_penalty=0.0,
    )

    result = asyncio.run(evaluator.evaluate(chunker=None))
    assert result.error is None
    assert result.n_chunks == 4
    assert abs(result.mrr - 2 / 3) < 1e-9
    assert abs(result.hit_rate - 2 / 3) < 1e-9
    assert result.fitness == result.mrr

    # the embeddings are cached: a second evaluation embeds nothing new
    calls = embedder.calls
    asyncio.run(evaluator.evaluate(chunker=None))
    assert embedder.calls == calls

    evaluator.size_penalty = 1.0
    penalized = asyncio.run(evaluator.evaluate(chunker=None))
    assert abs(penalized.fitness - (penalized.mrr - penalized.avg_chunk_tokens / 1000)) < 1e-9


def test_chunker_evaluator_relevance_requires_the_same_source():
    samples = [
        evaluation.SampleFile(name="a.txt", docs=[Document(page_content="x", metadata={"source": "a.txt"})]),
        evaluation.SampleFile(name="b.txt", docs=[Document(page_content="x", metadata={"source": "b.txt"})]),
    ]
    chunks = {"a.txt": ["shared sentence about apples"], "b.txt": ["unrelated text"]}
    questions = [evaluation.SyntheticQuestion(question="apples", excerpt="shared sentence about apples", source="b.txt")]
    evaluator = evaluation.ChunkerEvaluator(
        samples=samples,
        questions=questions,
        cache=evaluation.EmbeddingCache(embed_documents=_BagOfWordsEmbedder().embed_documents),
        split=_fixed_split(chunks),
        top_k=2,
        size_penalty=0.0,
    )
    assert asyncio.run(evaluator.evaluate(chunker=None)).mrr == 0.0


def test_chunker_evaluator_without_chunks():
    samples = [evaluation.SampleFile(name="a.txt", docs=[Document(page_content="x", metadata={"source": "a.txt"})])]
    evaluator = evaluation.ChunkerEvaluator(
        samples=samples,
        questions=[],
        cache=evaluation.EmbeddingCache(embed_documents=_BagOfWordsEmbedder().embed_documents),
        split=_fixed_split({"a.txt": []}),
        top_k=2,
        size_penalty=0.0,
    )
    result = asyncio.run(evaluator.evaluate(chunker=None))
    assert result.fitness == evaluation.WORST_FITNESS
    assert result.error
