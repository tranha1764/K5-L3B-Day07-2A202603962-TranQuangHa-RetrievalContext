# Báo cáo Day 7

**Sinh viên:** Trần Quang Hà  
**MSSV:** 2A202603962  
**Lớp:** L3B  

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Giáo trình mở *OpenStax: College Success* (Chương 1 & Chương 2), giấy phép bản quyền Creative Commons Attribution 4.0 International (CC BY 4.0).
  - Tổng dung lượng: 58 trang PDF, được trích xuất và phân đoạn theo Chiến lược B (by-paragraph chunking) kế thừa từ Day 6.
  - Tổng số chunk: **325 chunks** có cấu trúc JSON thống nhất gồm `id`, `text`, `source_id`, `source`, `page`, `metadata`.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: PostgreSQL 16 tích hợp extension `pgvector`, vận hành trên Docker container cổng local `55433`.
  - Embedding model: OpenAI `text-embedding-3-small` với kích thước vector cố định **768 chiều** (được cấu hình qua tham số `dimensions=768` tương thích hoàn toàn với index Day 6). Khoảng cách vector đo bằng Cosine Distance (`vector_cosine_ops`).
- **Agent model, cách chạy và phân chia dev/test:**
  - Controller model: OpenAI `gpt-4o-mini` thông qua Structured Tool Calling (chế độ JSON response điều phối 2 hành động chính: `search` và `answer`).
  - Chia tập benchmark: Tổng cộng **24 câu hỏi** được thiết kế độc lập:
    - **Tập Dev (17 câu):** 15 câu có đáp án trong tài liệu (`q01`–`q15`) và 2 câu unanswerable (`unans_dev_01`, `unans_dev_02`).
    - **Tập Test (7 câu):** 5 câu có đáp án (`q16`–`q20`) và 2 câu unanswerable (`unans_test_01`, `unans_test_02`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án:** Có ground truth evidence xác định chính xác tới từng trang và đoạn văn trong sách. Tiêu chí chấm:
    - `0`: Sai lệch nội dung hoặc từ chối nhầm (false abstention).
    - `1`: Trả lời đúng một phần hoặc thiếu điều kiện ràng buộc.
    - `2`: Trả lời chính xác, đầy đủ và có trích dẫn `[chunk_id]` hợp lệ hỗ trợ trực tiếp.
  - **Câu thiếu evidence:** Đặt câu hỏi thực tế nhưng nằm ngoài phạm vi tài liệu (quy định ChatGPT/AI, quy chế ký túc xá). Tiêu chí chấm: Agent phải chủ động tìm kiếm và từ chối an toàn sau khi hết lượt tìm kiếm (`abstention_correct = True`), tuyệt đối không được ảo giác (hallucination).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF Lexical ($k=3$, b=800) | Tìm kiếm theo từ khóa truyền thống | dev | 9/15 (60.0%) | 14/15 (93.3%) | 9/15 (60.0%) | 0.66 ms / $0 |
| **Phương pháp 2** | Dense pgvector ($k=3$, b=800) | Tìm kiếm ngữ nghĩa vector 768d Day 6 | dev | 7/15 (46.7%) | 14/15 (93.3%) | 7/15 (46.7%) | 235.31 ms / ~$0.0001 |
| **Phương pháp 3** | Hybrid RRF ($k=3$, b=800) | Kết hợp Dense + Lexical qua RRF ($k_0=60$) | dev | 9/15 (60.0%) | **15/15 (100.0%)** | 9/15 (60.0%) | 218.06 ms / ~$0.0001 |
| **Cải thiện 1** | Hybrid RRF ($k=3$, **b=1600**) | Tăng context budget để tránh tràn context | dev | 9/15 (60.0%) | **15/15 (100.0%)** | **14/15 (93.3%)** | 231.32 ms / ~$0.0001 |
| **Cải thiện 2** | Hybrid RRF (**$k=5$**, b=1600) | Mở rộng danh sách ứng viên Top-k lên 5 | dev | 9/15 (60.0%) | **15/15 (100.0%)** | 14/15 (93.3%) | 233.80 ms / ~$0.0001 |
| **Cấu hình đã chọn** | Hybrid RRF ($k=3$, b=1600) | Đánh giá độc lập trên tập Test chưa thấy | test | 4/5 (80.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 323.08 ms / ~$0.0001 |
| **Baseline** | TF-IDF Lexical ($k=3$, b=800) | Đối chiếu Baseline trên cùng tập Test | test | 4/5 (80.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 0.75 ms / $0 |

### Phân tích hai lỗi thực tế trên tập Dev và lý do chọn cấu hình:

1. **Lỗi 1 — Hiện tượng khuyết điểm bổ trợ giữa TF-IDF và Dense (Câu `q08`):**
   - *Hiện tượng:* Ở câu `q08` (*"What does it mean to make learning intentional?"*), phương pháp TF-IDF bị trượt hoàn toàn (Hit@3 = False) do câu hỏi sử dụng các khái niệm trừu tượng, không trùng khớp mặt chữ với văn bản. Ngược lại, **Dense pgvector lại tìm trúng** (Hit@3 = True) nhờ embedding nắm bắt được ngữ nghĩa. Nhờ thuật toán Reciprocal Rank Fusion (RRF), phương pháp Hybrid đã tổng hợp được ưu thế của cả hai, đẩy Evidence Hit@3 trên tập Dev đạt mức tuyệt đối: **15/15 (100.0%)**.
2. **Lỗi 2 — Điểm nghẽn Context Budget (Các câu `q02`, `q04`, `q07`, `q11`, `q15`):**
   - *Hiện tượng:* Ở cấu hình Hybrid với `budget = 800`, mặc dù Hit@3 đạt 100% nhưng chỉ số đưa bằng chứng vào ngữ cảnh (`Context Evidence Rate`) chỉ đạt 60.0% (9/15 câu).
   - *Nguyên nhân:* Các câu trên có chunk chứa bằng chứng nằm ở Rank 2 hoặc Rank 3. Do mỗi chunk dài trung bình ~600 ký tự (kèm header định danh ~80 ký tự), giới hạn budget 800 chỉ vừa đủ nạp chunk ở Rank 1, khiến các chunk ở Rank sau bị cắt bỏ hoàn toàn trước khi đưa vào ngữ cảnh.
   - *Giải pháp & Kết quả:* Khi nâng budget lên **1600 ký tự** (Cải thiện 1), ngữ cảnh chứa vừa vặn 2–3 chunk liên quan nhất. Ngay lập tức, `Context Evidence Rate` tăng vọt từ **60.0% lên 93.3% (14/15 câu)**.
3. **Lý do chốt Cấu hình 4 (Hybrid RRF, $k=3$, budget 1600):**
   - Đạt tỷ lệ tìm thấy bằng chứng tối đa trong Top-3 (100%) và tỷ lệ đưa bằng chứng vào context cao nhất (93.3%).
   - Tối ưu hơn Cải thiện 2 ($k=5$): Không làm tăng token context thừa, giữ độ trễ thấp và tiết kiệm chi phí gọi LLM.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** Agent được trang bị công cụ `search(query, filter)` giao tiếp trực tiếp với pipeline Hybrid Retrieval (OpenAI embedding + PostgreSQL pgvector + TF-IDF index).
  - **Số lượt gọi & Điều kiện dừng:** Tối đa 3 lượt tìm kiếm (`max_searches = 3`). Agent dừng khi:
    - Tìm đủ thông tin và chuyển sang hành động `answer` (kèm trích dẫn định dạng `[chunk_id]`).
    - Hoặc hết số lượt tìm kiếm cho phép mà không đủ bằng chứng $\to$ kích hoạt dừng và trả lời từ chối an toàn: *"Không đủ bằng chứng để trả lời."*
  - **Chọn context:** Sử dụng cơ chế cộng dồn context theo dung lượng budget 1600 ký tự, ưu tiên giữ lại các chunk có độ tương đồng cao nhất từ các lượt tìm kiếm trước đó.
  - **Xử lý lỗi:** Nếu model trả lời câu hỏi nhưng thiếu trích dẫn hoặc trích dẫn ID không tồn tại trong context, hệ thống tự động phát hiện qua sự kiện `invalid_or_missing_citations` và hạ cấp an toàn về trạng thái `abstain`.

### Bảng chỉ số tổng hợp Agent trên tập Test (7 câu)

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Hybrid RRF (b=1600) / test** | **7/7 (100.0%)** | 2 đến 4 lượt (TB: 2.57) | 0: 2, 1: 0, 2: 3 (Mean: **1.20 / 2.0**) | **3/3 (100.0%)** | **3/3 (100.0%)** | **2/2 (100.0%)** | 2/5 (40.0%) |

### Bảng thẩm định chi tiết từng truy vấn trên tập Test (Human Audit)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q16** | Có đáp án | 1 search: *"college expectations guide most learning takes place"* $\to$ 3 chunks | Trả lời việc học chủ yếu diễn ra ngoài lớp học và tự học. Trích dẫn: `[college_success_p15_para_03]` | **Điểm: 2/2**. Citation hợp lệ (True). Bằng chứng trang 15 hỗ trợ đầy đủ câu trả lời. |
| **q17** | Có đáp án | 1 search: *"how relationships change and expand for students during college"* $\to$ 3 chunks (p15_para_00, p14_para_03) | *"Không đủ bằng chứng để trả lời."* | **Điểm: 0/2**. Từ chối nhầm (False abstention): Retrieval tìm trúng p15_para_00 nhưng do format citation từ LLM bị lỗi nên kích hoạt fallback. |
| **q18** | Có đáp án | 1 search: *"Myers-Briggs assessment application Chapter 2"* $\to$ 3 chunks | Trả lời ứng dụng bài test trong hội thảo doanh nghiệp và động lực nhóm. Trích dẫn: `[college_success_p49_para_00]` | **Điểm: 2/2**. Citation hợp lệ (True). Bằng chứng trang 49 hỗ trợ trực tiếp và đầy đủ. |
| **q19** | Có đáp án | 1 search: *"former STEM student discover studying Russian"* $\to$ 3 chunks (p26_para_00, p26_para_01) | *"Không đủ bằng chứng để trả lời."* | **Điểm: 0/2**. Từ chối nhầm (False abstention): Retrieval tìm trúng p26_para_00 nhưng gặp lỗi format citation từ LLM nên fallback từ chối. |
| **q20** | Có đáp án | 1 search: *"ancient philosophical obsession modern learning theories"* $\to$ 3 chunks | Trả lời nỗi ám ảnh từ cổ đại về cách con người tiếp thu tri thức. Trích dẫn: `[college_success_p27_para_00]` | **Điểm: 2/2**. Citation hợp lệ (True). Bằng chứng trang 27 hỗ trợ chính xác. |
| **unans_test_01** | Thiếu evidence | 3 searches: *"quy định sử dụng công cụ trí tuệ nhân tạo trong viết tiểu luận"* $\to$ đạt giới hạn | *"Không đủ bằng chứng để trả lời."* | **Abstention đúng (True)**. Quy định ChatGPT không có trong tài liệu; agent kiên quyết từ chối, không hallucination. |
| **unans_test_02** | Thiếu evidence | 3 searches: *"thủ tục thời hạn nộp hồ sơ xin ở ký túc xá sinh viên năm nhất"* $\to$ đạt giới hạn | *"Không đủ bằng chứng để trả lời."* | **Abstention đúng (True)**. Thủ tục ký túc xá nằm ngoài phạm vi tài liệu; agent kiên quyết từ chối, không hallucination. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Traces: `starter/results/agent_test/traces.json`
  - Nhãn audit: `starter/results/agent_test/reviews.json`
  - Metrics tổng hợp: `starter/results/agent_test/summary.json`
  - Lệnh chạy: `python src/agent_runner.py --split test --strategy hybrid --budget 1600 --output starter/results/agent_test/traces.json`
  - Tổng thời gian chạy 7 câu: 28.47 giây (trung vị ~4.0 giây/câu).
  - Chi phí API thực tế (gpt-4o-mini + text-embedding-3-small): xấp xỉ **$0.0022 USD** (~55 VNĐ).
- **Đánh đổi và Giới hạn:**
  - *Chất lượng vs Độ trễ:* Hybrid RRF tăng độ chính xác retrieval lên mức tối đa (100% Hit@3 trên Dev), nhưng độ trễ tăng từ 0.66ms (TF-IDF) lên ~230ms do cần tạo embedding qua mạng và tính toán rank fusion. Tuy nhiên, mức trễ này hoàn toàn chấp nhận được so với thời gian suy luận của LLM (~3-5 giây).
  - *Hiện tượng False Abstention:* Việc áp dụng cơ chế xác thực trích dẫn nghiêm ngặt (Strict Citation Verification) giúp hệ thống triệt tiêu hoàn toàn hallucination trên các câu hỏi thiếu bằng chứng (đạt 100% abstention đúng), nhưng đánh đổi bằng việc có 2 câu (`q17`, `q19`) bị từ chối nhầm do model sinh format citation chưa chuẩn.
