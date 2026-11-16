"""
Script điều phối chạy Agent trên benchmark câu hỏi.
Lưu trace hoàn chỉnh từng tool call, query, context và câu trả lời ra file JSON.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "starter"))

from starter.agentic import run_agent
from src.retrievers import DenseRetriever
from src.controller import OpenAIController


def main():
    parser = argparse.ArgumentParser(description="Chạy Agentic RAG trên benchmark")
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "data" / "agent_cases.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Số câu tối đa để chạy thử")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "starter" / "results" / "agent_dev_traces.json")
    args = parser.parse_args()

    print("====================================================")
    print(f"   CHẠY AGENTIC RAG — SPLIT: {args.split.upper()}   ")
    print("====================================================")

    # 1. Khởi tạo Retriever và Controller
    retriever = DenseRetriever()
    controller = OpenAIController()

    # 2. Nạp câu hỏi theo split
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_cases = [c for c in all_cases if c.get("split") == args.split]
    if args.limit:
        selected_cases = selected_cases[:args.limit]

    print(f"-> Đã chọn {len(selected_cases)} câu hỏi để chạy thử nghiệm.")

    traces = []
    for idx, case in enumerate(selected_cases, start=1):
        cid = case["id"]
        question = case["question"]
        print(f"\n[{idx}/{len(selected_cases)}] Câu hỏi {cid}: \"{question}\"")

        start_time = time.perf_counter()
        try:
            result = run_agent(
                question=question,
                controller=controller,
                retriever=retriever,
                filters=case.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            trace_entry = {
                "query_id": cid,
                "split": args.split,
                "question": question,
                "elapsed_ms": elapsed_ms,
                **result,
            }
            print(f"   Trạng thái: {result['status']} | Số tool calls: {len(result['events'])} | Thời gian: {elapsed_ms:.1f}ms")
            print(f"   Trả lời: {result['answer']}")
            print(f"   Citations: {result['citations']}")
        except Exception as e:
            print(f"   [LỖI] {e}")
            trace_entry = {
                "query_id": cid,
                "split": args.split,
                "question": question,
                "status": "error",
                "error": str(e),
                "events": [],
                "citations": [],
            }
        traces.append(trace_entry)

    # 3. Lưu kết quả ra file traces.json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=> Đã lưu thành công {len(traces)} traces vào: {args.output}")


if __name__ == "__main__":
    main()
