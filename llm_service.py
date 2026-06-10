import requests
from config import settings


# ==========================================
# 大模型总结微服务组件 (LLM Summary Service)
# ==========================================
# 职责：负责接收图谱检索层（Search Service）打捞上来的黄金上下文 facts，
# 运用标准 Prompt 机制对其进行思维锚定，最终调用 DeepSeek 提炼出确定性的白话文答案。
class LLMService:
    def __init__(self):
        """
        构造函数：客户端基础设施初始化
        解耦硬编码，统一从 config.py 的 settings 单例中动态调取秘钥、模型名称和 API 终点
        """
        self.api_url = settings.LLM_API_BASE
        self.api_key = settings.SILICON_API_KEY
        self.model = settings.LLM_MODEL

    def generate_answer(self, question: str, context: str) -> str:
        """
        核心业务方法：多路召回上下文融合与约束推理生成

        参数:
            question (str): 前端传入的用户原始提问文本
            context (str): 经由 Neo4j 拓扑网络动态裁剪、清洗拼接后的黄金事实上下文

        返回:
            str: 100% 严格基于事实提炼的无幻觉最终中文回答
        """
        # 🚫 前置卡口：如果知识图谱及传统 RAG 侧均未打捞到任何有效事实线索，
        # 立刻执行快速失败（Fail-Fast）机制，直接截断返回，拒绝让大模型在真空状态下胡编乱造
        if not context or not context.strip():
            return "抱歉，知识图谱中未打捞到相关线索，无法给出准确回答。"

        # 1. 焊死生产级 System Prompt 模版（用绝对律法束缚住 LLM 的发散思维）
        # 核心心法：明确界定专家身份（星河科技集团首席知识专家），并赋予其极高的严谨性约束
        system_prompt = (
            "你是一位严谨的星河科技集团首席知识专家。\n"
            "请严格基于下面提供的【参考图谱上下文】来回答用户的提问。\n"
            "要求如下：\n"
            "1. 只能根据提供的事实进行回答，绝对不允许凭空捏造、瞎编乱造（严防幻觉）。\n"
            "2. 如果上下文里没有涉及用户问题的线索，请直说'基于现有图谱未找到相关事实'。\n"
            "3. 回答条理清晰，语气专业，直接说结果，不要客套。\n\n"
            "4. 允许在严格基于事实的前提下，进行合理的逻辑整合与概念提炼。\n"
            f"【参考图谱上下文】:\n{context}"
        )

        # 2. 组装标准 HTTP 承载头（注入 Bearer Token 鉴权认证）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 3. 组装 OpenAI 兼容的标准 Chat Completions 协议结构体
        payload = {
            "model": self.model,  # 动态路由到统一配置的推理模型（如 DeepSeek-V3）
            "messages": [
                {"role": "system", "content": system_prompt},  # 注入防幻觉系统死规矩和事实库
                {"role": "user", "content": question}  # 注入用户当前问题
            ],
            "temperature": 0.3,  # ⚠️ 低温度阈值：压榨大模型的发散创造力，强行让其充当“无情的事实材料总结官”
            "stream": False  # 当前服务走标准 HTTP 短链接口阻断返回，流式传输（SSE）留待后续高阶重构
        }

        try:
            # 4. 同步轰炸第三方大模型中台（如硅基流动或私有化部署终点）
            # 设置 30 秒刚性超时阻断，防止上游响应卡死活活拖垮 FastAPI 连接池
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

            # 5. 状态码异常对账：如果 HTTP 状态码非 2xx（如 401 鉴权失败、502 满载），当场抛出异常
            response.raise_for_status()

            # 6. 顺着标准 JSON 协议的规范路径，像素级剥离出最终的文本大白话
            result_json = response.json()
            answer = result_json["choices"][0]["message"]["content"]
            return answer

        except Exception as e:
            # 🛡️ 异常防线：捕获包含网络抖动、协议解析错位在内的全链路未知异常
            error_msg = f"❌ 调用大模型总结失败: {e}"
            print(error_msg)  # 在服务器控制台打印第一手追溯日志
            return error_msg  # 将友好且规整的错误账单抛回上游 API 层