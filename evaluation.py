import asyncio
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Dict, List

import numpy as np
from langchain_core.documents import Document

from cat.log import log
from cat.services.factory.chunker import BaseChunker

# a chunk is relevant to a question when it comes from the same file and holds at least this share of the answer
# excerpt (contiguously): the excerpt may be split across two chunks
MIN_SPAN_COVERAGE = 0.6

PASSAGE_CHARS = 1500
MIN_PASSAGE_CHARS = 200
EMBEDDING_BATCH_SIZE = 64

# fitness of a candidate that cannot be evaluated (invalid configuration, failure, exhausted budget): far below any real
# fitness (MRR is in [0, 1]), but finite, to keep the arithmetic of the metaheuristics sound
WORST_FITNESS = -10.0

QUESTION_PROMPT = """You are building an evaluation set for a document retrieval system.
Read the passage and write ONE question that can be answered only by using the passage, and copy the exact excerpt of the passage that answers it.

Rules:
- The question must be specific and self-contained: never refer to "the passage", "the text" or "the document".
- Write the question in the same language as the passage.
- "answer_excerpt" must be copied VERBATIM from the passage: a contiguous span of 5 to 40 words.

Reply ONLY with a JSON object, without any other text: {{"question": "...", "answer_excerpt": "..."}}

Passage:
\"\"\"
{passage}
\"\"\"
"""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def span_coverage(span: str, text: str) -> float:
    """Share of the (normalized) span contiguously contained in the (normalized) text."""
    if not span:
        return 0.0
    if span in text:
        return 1.0
    matcher = SequenceMatcher(None, text, span, autojunk=False)
    match = matcher.find_longest_match(0, len(text), 0, len(span))
    return match.size / len(span)


def approximate_tokens(text: str) -> float:
    return len(text) / 4


@dataclass
class SampleFile:
    """A file of the datalake, parsed (not split yet) for the evaluation."""
    name: str
    docs: List[Document]

    @property
    def text_length(self) -> int:
        return sum(len(d.page_content) for d in self.docs)


@dataclass
class SyntheticQuestion:
    question: str
    excerpt: str  # normalized answer excerpt
    source: str


@dataclass
class EvaluationResult:
    fitness: float
    mrr: float = 0.0
    hit_rate: float = 0.0
    avg_chunk_tokens: float = 0.0
    n_chunks: int = 0
    error: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "avg_chunk_tokens": self.avg_chunk_tokens,
            "n_chunks": self.n_chunks,
            "error": self.error,
        }


def truncate_docs(docs: List[Document], max_chars: int) -> List[Document]:
    """Copy the documents, keeping at most max_chars characters overall."""
    result, remaining = [], max_chars
    for doc in docs:
        if remaining <= 0:
            break
        content = doc.page_content[:remaining]
        if content.strip():
            result.append(Document(page_content=content, metadata=dict(doc.metadata)))
        remaining -= len(content)
    return result


def pick_passages(sample: SampleFile, n: int, rng: random.Random) -> List[str]:
    """Pick n passages of the file (random windows, aligned to whitespace), to generate the questions from."""
    docs = [d for d in sample.docs if len(d.page_content.strip()) >= MIN_PASSAGE_CHARS]
    if not docs:
        return []

    passages: List[str] = []
    weights = [len(d.page_content) for d in docs]
    for _ in range(n * 3):  # some attempts may hit the same window
        if len(passages) >= n:
            break
        doc = rng.choices(docs, weights=weights, k=1)[0]
        text = doc.page_content
        if len(text) <= PASSAGE_CHARS:
            passage = text
        else:
            start = rng.randint(0, len(text) - PASSAGE_CHARS)
            # align to the next whitespace, to avoid cutting words
            while start > 0 and not text[start - 1].isspace() and start < len(text) - MIN_PASSAGE_CHARS:
                start += 1
            passage = text[start:start + PASSAGE_CHARS]
        passage = passage.strip()
        if len(passage) >= MIN_PASSAGE_CHARS and passage not in passages:
            passages.append(passage)
    return passages


def parse_question_reply(reply: str, passage: str) -> Dict[str, str] | None:
    """Parse the reply of the LLM; the excerpt must be (almost) verbatim in the passage."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    question = str(data.get("question") or "").strip()
    excerpt = normalize_text(str(data.get("answer_excerpt") or ""))
    if len(question) < 10 or len(excerpt.split()) < 3:
        return None

    # tolerate small differences (e.g. quotes, punctuation) between the excerpt and the passage
    if span_coverage(excerpt, normalize_text(passage)) < 0.9:
        return None

    return {"question": question, "excerpt": excerpt}


def llm_reply_to_text(reply: Any) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


async def generate_questions(
    samples: List[SampleFile],
    ask_llm: Callable[[str], Awaitable[str]],
    questions_per_file: int,
    rng: random.Random,
    concurrency: int = 4,
) -> List[SyntheticQuestion]:
    """Generate the synthetic evaluation set: questions answered by verbatim excerpts of the sampled files."""
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(sample: SampleFile, passage: str) -> SyntheticQuestion | None:
        async with semaphore:
            try:
                reply = await ask_llm(QUESTION_PROMPT.format(passage=passage))
            except Exception as e:
                log.warning(f"AutoChunk: question generation failed for {sample.name}: {e}")
                return None
        parsed = parse_question_reply(reply, passage)
        if parsed is None:
            return None
        return SyntheticQuestion(question=parsed["question"], excerpt=parsed["excerpt"], source=sample.name)

    tasks = [
        ask(sample, passage)
        for sample in samples
        for passage in pick_passages(sample, questions_per_file, rng)
    ]
    results = await asyncio.gather(*tasks)
    return [q for q in results if q is not None]


@dataclass
class EmbeddingCache:
    """Cache of the (normalized) embeddings of the texts, shared by all the candidates of a run."""
    embed_documents: Callable[[List[str]], Awaitable[List[List[float]]]]
    _vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: List[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._vectors))
        for i in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[i:i + EMBEDDING_BATCH_SIZE]
            vectors = await self.embed_documents(batch)
            for text, vector in zip(batch, vectors):
                array = np.asarray(vector, dtype=np.float32)
                norm = np.linalg.norm(array)
                self._vectors[self._key(text)] = array / norm if norm > 0 else array
        return np.vstack([self._vectors[self._key(t)] for t in texts])


SplitFunction = Callable[[BaseChunker, List[Document]], Awaitable[List[Document]]]


class ChunkerEvaluator:
    """Evaluate a chunker on the sample of the datalake: retrieval quality (MRR@k) of the synthetic questions over the
    chunks produced by the chunker, penalized by the average size of the chunks."""

    def __init__(
        self,
        samples: List[SampleFile],
        questions: List[SyntheticQuestion],
        cache: EmbeddingCache,
        split: SplitFunction,
        top_k: int,
        size_penalty: float,
    ):
        self.samples = samples
        self.questions = questions
        self.cache = cache
        self.split = split
        self.top_k = top_k
        self.size_penalty = size_penalty
        self._question_vectors: np.ndarray | None = None

    async def _get_question_vectors(self) -> np.ndarray:
        if self._question_vectors is None:
            self._question_vectors = await self.cache.embed([q.question for q in self.questions])
        return self._question_vectors

    async def evaluate(self, chunker: BaseChunker) -> EvaluationResult:
        chunk_texts: List[str] = []
        chunk_sources: List[str] = []
        for sample in self.samples:
            docs = [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in sample.docs]
            for chunk in await self.split(chunker, docs):
                if chunk.page_content.strip():
                    chunk_texts.append(chunk.page_content)
                    chunk_sources.append(sample.name)

        if not chunk_texts:
            return EvaluationResult(fitness=WORST_FITNESS, error="the chunker produced no chunks")

        chunk_vectors = await self.cache.embed(chunk_texts)
        question_vectors = await self._get_question_vectors()
        normalized_chunks = [normalize_text(t) for t in chunk_texts]

        scores = question_vectors @ chunk_vectors.T
        k = min(self.top_k, len(chunk_texts))

        reciprocal_ranks = []
        for qi, question in enumerate(self.questions):
            top = np.argsort(-scores[qi])[:k]
            rr = 0.0
            for rank, ci in enumerate(top, start=1):
                if chunk_sources[ci] != question.source:
                    continue
                if span_coverage(question.excerpt, normalized_chunks[ci]) >= MIN_SPAN_COVERAGE:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        mrr = float(np.mean(reciprocal_ranks))
        hit_rate = float(np.mean([rr > 0 for rr in reciprocal_ranks]))
        avg_tokens = float(np.mean([approximate_tokens(t) for t in chunk_texts]))
        fitness = mrr - self.size_penalty * avg_tokens / 1000

        return EvaluationResult(
            fitness=fitness, mrr=mrr, hit_rate=hit_rate, avg_chunk_tokens=avg_tokens, n_chunks=len(chunk_texts),
        )
