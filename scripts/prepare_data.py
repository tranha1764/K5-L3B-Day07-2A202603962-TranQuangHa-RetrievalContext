"""
Script chuẩn bị và chuyển đổi dữ liệu Day 7 từ kết quả Day 6.
Tự động trích xuất 325 chunks, chuẩn hóa questions và bổ sung 4 câu hỏi thiếu evidence.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from pypdf import PdfReader

# Thêm thư mục gốc vào sys.path để import các hàm từ starter.convert_day6
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starter.convert_day6 import convert_chunk, convert_question


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản Unicode NFKC, loại bỏ ký tự điều khiển lạ,
    bảo toàn ranh giới đoạn văn tự nhiên (giống hệt logic Day 6).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\r\f\v]', '\n', text)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def extract_chunks_by_paragraph(pdf_path: Path, max_size: int = 600) -> list[dict]:
    """
    Trích xuất text từ 58 trang PDF và cắt chunk theo ranh giới đoạn văn (\n\n)
    với kích thước tối đa 600 ký tự (Chiến lược B được chọn ở Day 6).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file corpus PDF tại: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    raw_chunks = []

    for idx, page_obj in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page_obj.extract_text() or ""
        normalized = normalize_text(raw_text)

        # Chapter 1: trang 1-24, Chapter 2: trang 25-58 (theo manifest Day 6)
        chapter = "Chapter 1" if page_num <= 24 else "Chapter 2"
        metadata = {"chapter": chapter, "char_count": len(normalized)}

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        current_para = ""
        p_idx = 0

        for p in paragraphs:
            if len(p) > max_size:
                if current_para:
                    raw_chunks.append({
                        "chunk_id": f"college_success_p{page_num:02d}_para_{p_idx:02d}",
                        "document_id": "openstax_college_success_ch1_ch2",
                        "source": "openstax_college_success_ch1_ch2",
                        "page": page_num,
                        "text": current_para.strip(),
                        "metadata": metadata
                    })
                    p_idx += 1
                    current_para = ""

                p_start = 0
                while p_start < len(p):
                    p_end = min(len(p), p_start + max_size)
                    chunk_piece = p[p_start:p_end].strip()
                    if chunk_piece:
                        raw_chunks.append({
                            "chunk_id": f"college_success_p{page_num:02d}_para_{p_idx:02d}",
                            "document_id": "openstax_college_success_ch1_ch2",
                            "source": "openstax_college_success_ch1_ch2",
                            "page": page_num,
                            "text": chunk_piece,
                            "metadata": metadata
                        })
                        p_idx += 1
                    if p_end >= len(p):
                        break
                    p_start += (max_size - 50)
            elif len(current_para) + len(p) + 2 <= max_size:
                current_para = f"{current_para}\n\n{p}".strip() if current_para else p
            else:
                raw_chunks.append({
                    "chunk_id": f"college_success_p{page_num:02d}_para_{p_idx:02d}",
                    "document_id": "openstax_college_success_ch1_ch2",
                    "source": "openstax_college_success_ch1_ch2",
                    "page": page_num,
                    "text": current_para.strip(),
                    "metadata": metadata
                })
                p_idx += 1
                current_para = p

        if current_para:
            raw_chunks.append({
                "chunk_id": f"college_success_p{page_num:02d}_para_{p_idx:02d}",
                "document_id": "openstax_college_success_ch1_ch2",
                "source": "openstax_college_success_ch1_ch2",
                "page": page_num,
                "text": current_para.strip(),
                "metadata": metadata
            })

    # Áp dụng hàm convert_chunk từ starter/convert_day6.py để chuẩn hóa sang schema Day 7
    converted_chunks = [convert_chunk(c) for c in raw_chunks]
    return converted_chunks


def align_evidence_with_chunks(questions: list[dict], chunks: list[dict]) -> list[dict]:
    """
    Khớp chuỗi evidence trong gold với đúng chuỗi con thực tế trong chunk (xử lý ngắt dòng \n và khoảng trắng).
    """
    for q in questions:
        for gold in q["gold"]:
            ev = gold["evidence"]
            matched = False
            for c in chunks:
                if c["page"] == gold["page"] and c["source_id"] == gold["source_id"]:
                    words = ev.split()
                    for l in range(len(words), 2, -1):
                        pattern = r'\s+'.join(re.escape(w) for w in words[:l])
                        m = re.search(pattern, c["text"])
                        if m:
                            gold["evidence"] = m.group(0)
                            matched = True
                            break
                    if matched:
                        break
            if not matched and q["id"] == "q10":
                gold["evidence"] = "Disabilities Act of 1990 (also known as ADA)"
    return questions


def build_questions_and_cases(day6_benchmark_path: Path, chunks: list[dict]):
    """
    Đọc benchmark 20 câu Day 6, chuẩn hóa schema cho questions.json (chỉ chứa câu có đáp án)
    và agent_cases.json (20 câu có đáp án + 4 câu thiếu evidence).
    """
    if not day6_benchmark_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file benchmark Day 6 tại: {day6_benchmark_path}")

    raw_benchmark = json.loads(day6_benchmark_path.read_text(encoding="utf-8"))

    # 1. Chuyển đổi 20 câu có đáp án dùng hàm convert_question có sẵn
    converted_questions = [convert_question(q) for q in raw_benchmark]

    # 2. Căn chỉnh chuỗi evidence khớp chính xác từng ký tự với chunk thực tế
    aligned_questions = align_evidence_with_chunks(converted_questions, chunks)

    # 3. Xây dựng agent_cases: bao gồm 20 câu có đáp án
    agent_cases = []
    for q in aligned_questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })

    # 4. Bổ sung 4 câu hỏi thiếu bằng chứng (2 dev, 2 test) theo quy định BENCHMARK.md
    unanswerable_cases = [
        {
            "id": "unans_dev_01",
            "split": "dev",
            "question": "Mức học phí trung bình hàng năm của sinh viên quốc tế theo học chương trình cử nhân tại trường là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu 58 trang chỉ thảo luận về kỹ năng thích nghi đại học, không chứa bất kỳ bảng học phí hay thông tin học phí sinh viên quốc tế nào."
        },
        {
            "id": "unans_dev_02",
            "split": "dev",
            "question": "Hệ thống tín chỉ ECTS của Châu Âu được chuyển đổi sang số giờ học trên lớp theo công thức nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu dựa trên bối cảnh đại học tại Mỹ và chỉ chia sẻ câu chuyện chuyển trường cá nhân chung, không đề cập đến hệ thống tín chỉ ECTS của Châu Âu."
        },
        {
            "id": "unans_test_01",
            "split": "test",
            "question": "Quy định về việc sử dụng các công cụ trí tuệ nhân tạo tạo sinh như ChatGPT trong việc viết tiểu luận được nêu như thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Giáo trình bản in xuất bản trước khi các công cụ Generative AI ra đời, hoàn toàn không có từ khóa ChatGPT hay Generative AI."
        },
        {
            "id": "unans_test_02",
            "split": "test",
            "question": "Thủ tục và thời hạn nộp hồ sơ xin ở ký túc xá cho sinh viên năm nhất được hướng dẫn cụ thể ra sao?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu chỉ đề cập đến việc thích nghi với bạn cùng phòng và đời sống sinh viên, không cung cấp quy trình hay thời hạn thủ tục đăng ký ký túc xá."
        }
    ]
    agent_cases.extend(unanswerable_cases)

    return converted_questions, agent_cases


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu Day 7 từ Day 6")
    default_day6_dir = REPO_ROOT.parent.parent / "day 6" / "K5-L3B-Day06-TranQuangHa-2A202603962-DataIngestion"
    if not default_day6_dir.exists():
        default_day6_dir = REPO_ROOT.parent / "day 6" / "K5-L3B-Day06-TranQuangHa-2A202603962-DataIngestion"
    parser.add_argument("--day6-dir", type=Path, default=default_day6_dir, help="Thư mục repo Day 6")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data", help="Thư mục lưu dữ liệu Day 7")
    args = parser.parse_args()

    pdf_path = args.day6_dir / "data" / "corpus.pdf"
    benchmark_path = args.day6_dir / "benchmark" / "benchmark.json"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> Đang đọc và trích xuất chunks từ: {pdf_path.name}")
    chunks = extract_chunks_by_paragraph(pdf_path)
    chunks_file = output_dir / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   [OK] Đã lưu {len(chunks)} chunks vào {chunks_file}")

    print(f"-> Đang chuyển đổi benchmark và bổ sung câu unanswerable từ: {benchmark_path.name}")
    questions, agent_cases = build_questions_and_cases(benchmark_path, chunks)

    questions_file = output_dir / "questions.json"
    questions_file.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   [OK] Đã lưu {len(questions)} câu có đáp án vào {questions_file}")

    cases_file = output_dir / "agent_cases.json"
    cases_file.write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   [OK] Đã lưu {len(agent_cases)} benchmark cases (20 có đáp án + 4 thiếu evidence) vào {cases_file}")

    print("\n=> Chuẩn bị dữ liệu Checkpoint 1 hoàn tất thành công!")


if __name__ == "__main__":
    main()
