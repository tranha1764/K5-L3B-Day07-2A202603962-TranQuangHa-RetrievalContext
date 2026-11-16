"""
Module hiện thực Controller cho Agent dùng mô hình OpenAI gpt-4o-mini.
"""
import json
import os
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIController:
    """Controller gọi mô hình gpt-4o-mini ra quyết định tìm kiếm hoặc trả lời."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Quyết định hành động:
        - {"action": "search", "query": "..."}
        - {"action": "answer", "answer": "... [chunk-id]", "citations": ["chunk-id"]}
        """
        system_prompt = (
            "Bạn là một Agent thông minh chuyên nghiên cứu và trả lời câu hỏi dựa trên tài liệu.\n"
            "Nhiệm vụ của bạn là sử dụng công cụ tìm kiếm (search) để thu thập bằng chứng và đưa ra câu trả lời có nguồn trích dẫn.\n\n"
            "CÁC QUY TẮC BẮT BUỘC:\n"
            "1. Nếu chưa có ngữ cảnh tìm kiếm hoặc ngữ cảnh hiện tại chưa đủ thông tin, hãy yêu cầu tìm kiếm bằng cách trả về JSON:\n"
            "   {\"action\": \"search\", \"query\": \"cụm từ tìm kiếm ngắn gọn, trọng tâm\"}\n\n"
            "2. Nếu ngữ cảnh đã có đủ bằng chứng để trả lời câu hỏi:\n"
            "   - Hãy trả lời ngắn gọn, chính xác dựa trên bằng chứng trong ngữ cảnh.\n"
            "   - BẮT BUỘC trích dẫn mã chunk ngay sau thông tin được dẫn chứng theo định dạng [chunk_id].\n"
            "   - Liệt kê các chunk_id đó trong danh sách citations.\n"
            "   - Trả về JSON: {\"action\": \"answer\", \"answer\": \"câu trả lời ... [chunk_id]\", \"citations\": [\"chunk_id\"]}\n\n"
            "3. Nếu đã tìm kiếm nhưng tài liệu KHÔNG CHỨA bằng chứng hoặc không đủ thông tin:\n"
            "   - TUYỆT ĐỐI KHÔNG SUY ĐOÁN hay bịa đặt thông tin.\n"
            "   - Phải từ chối bằng chính xác câu: \"Không đủ bằng chứng để trả lời.\"\n"
            "   - Trả về JSON: {\"action\": \"answer\", \"answer\": \"Không đủ bằng chứng để trả lời.\", \"citations\": []}\n\n"
            "LƯU Ý: Chỉ xuất định dạng JSON hợp lệ, không kèm bất kỳ lời giải thích nào bên ngoài."
        )

        messages = [{"role": "system", "content": system_prompt}]

        user_content = f"Câu hỏi của người dùng: {question}\n\n"
        if history:
            latest = history[-1]
            user_content += (
                f"Lịch sử tìm kiếm trước đó:\n"
                f"- Các truy vấn đã tìm: {latest.get('queries', [])}\n"
                f"- Ngữ cảnh bằng chứng thu thập được:\n{latest.get('context', 'Chưa có')}\n\n"
                f"Hãy phân tích ngữ cảnh trên để quyết định tìm kiếm tiếp hay trả lời."
            )
        else:
            user_content += "Hiện tại chưa có ngữ cảnh nào. Hãy đưa ra truy vấn tìm kiếm đầu tiên."

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content or "{}"
        try:
            action = json.loads(content)
        except Exception:
            action = {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # Kiểm tra tính hợp lệ của action
        if action.get("action") not in ("search", "answer"):
            action = {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        return action
