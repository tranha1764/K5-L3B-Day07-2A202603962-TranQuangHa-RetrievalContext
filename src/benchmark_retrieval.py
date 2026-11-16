"""
Script benchmark và so sánh các phương pháp Retrieval trên tập Dev/Test.
Đo đạc chính xác Hit@1, Hit@3 và Context Evidence Rate
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "starter"))

from baseline import hit, pack_context, validate
from src.retrievers import DenseRetriever, TfidfRetriever, HybridRetriever


def evaluate_retriever(retriever, questions, top_k=3, budget=800):
    """Chạy câu hỏi qua retriever, đóng gói context và đo đạc các chỉ số Hit."""
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked)
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)

        traces.append({
            "query_id": q["id"],
            "question": q["question"],
            "top_k": top_k,
            "budget": budget,
            "results": marked,
            "context_ids": ids,
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
        })
    return traces


def main():
    parser = argparse.ArgumentParser(description="Benchmark Retrieval Strategies")
    parser.add_argument("--chunks", type=Path, default=REPO_ROOT / "data" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data" / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--strategy", choices=("tfidf", "dense", "hybrid"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected_q = [q for q in questions if q.get("split") == args.split]
    validate(chunks, selected_q)

    # Khởi tạo phương pháp truy xuất tương ứng
    if args.strategy == "tfidf":
        retriever = TfidfRetriever()
    elif args.strategy == "dense":
        retriever = DenseRetriever()
    elif args.strategy == "hybrid":
        retriever = HybridRetriever(DenseRetriever(), TfidfRetriever())

    traces = evaluate_retriever(retriever, selected_q, top_k=args.top_k, budget=args.budget)

    # Tổng hợp các chỉ số theo yêu cầu của BENCHMARK.md
    n = len(traces)
    hit1_cnt = sum(t["hit1"] for t in traces)
    hit3_cnt = sum(t["hit3"] for t in traces)
    ctx_cnt = sum(t["context_hit"] for t in traces)
    median_time = statistics.median(t["elapsed_ms"] for t in traces)

    summary = {
        "strategy": args.strategy,
        "split": args.split,
        "top_k": args.top_k,
        "budget": args.budget,
        "n": n,
        "hit1": f"{hit1_cnt}/{n} ({hit1_cnt/n*100:.1f}%)",
        "hit3": f"{hit3_cnt}/{n} ({hit3_cnt/n*100:.1f}%)",
        "context_evidence_rate": f"{ctx_cnt}/{n} ({ctx_cnt/n*100:.1f}%)",
        "median_ms": round(median_time, 2),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
