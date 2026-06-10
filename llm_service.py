import requests
from config import settings

#大模型总结 Service 类
class LLMService:
    def __init__(self):
        """客户端初始化，直接从统一的 config 里调取秘钥和终点"""
        self.api_url = settings.LLM_API_BASE
        self.api_key = settings.SILICON_API_KEY
        self.model = settings.LLM_MODEL

    def generate_answer(self, question: str, context: str) -> str:
        """
        核心方法：将图谱打捞出来的绝密情报（Context）作为事实依据，
        强行束缚住大模型的思维，防止它瞎编（幻觉），并返回最终大白话回答。
        """
        if not context or not context.strip():
            return "抱歉，知识图谱中未打捞到相关线索，无法给出准确回答。"

        # 1. 焊死生产级 Prompt 模版（用标准后端黑话限制它的发挥）
        system_prompt = (
            "你是一位严谨的星河科技集团首席知识专家。\n"
            "请严格基于下面提供的【参考图谱上下文】来回答用户的提问。\n"
            "要求如下：\n"
            "1. 只能根据提供的事实进行回答，绝对不允许凭空捏造、瞎编乱造（严防幻觉）。\n"
            "2. 如果上下文里没有涉及用户问题的线索，请直说'基于现有图谱未找到相关事实'。\n"
            "3. 回答条理清晰，语气专业，直接说结果，不要客套。\n\n"
            "4. 允许在严格基于事实的前提下，进行合理的逻辑整合与概念提炼\n。"
            f"【参考图谱上下文】:\n{context}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 2. 组装 OpenAI 兼容的标准 Chat 结构体
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3,  # ⚠️ 极低温度：把大模型的创造力打压到极限，让它老老实实当事实复述者
            "stream": False  # 暂时不走流式，接口直接一次性返回大字符串
        }

        try:
            # 3. 轰炸硅基流动 DeepSeek 服务端
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            # 4. 剥离出最终的文本大白话
            result_json = response.json()
            answer = result_json["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            error_msg = f"❌ 调用大模型总结失败: {e}"
            print(error_msg)
            return error_msg