import os
from typing import Dict, Generator, Iterable

import requests
from neo4j import GraphDatabase

from config import settings

# 强行给硅基流动开辟绿色通道，无视 Mac 代理软件，彻底防死 503 报错
os.environ["NO_PROXY"] = "api.siliconflow.cn"


class GraphRAGLocalSearcher:
    def __init__(self):
        """初始化：对接本地 Neo4j 数据库和硅基流动 API"""
        self.uri = settings.neo4j_uri
        self.auth = (settings.neo4j_user, settings.neo4j_password)
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        self.api_key = settings.siliconflow_api_key
        self.embedding_api_base = settings.siliconflow_embedding_api_base
        self.chat_api_base = settings.siliconflow_chat_api_base
        self.embedding_model = settings.embedding_model
        self.chat_model = settings.chat_model

    def close(self):
        """优雅关闭数据库连接"""
        self.driver.close()

    def _get_question_embedding(self, question: str):
        """私有方法：现场将用户输入的大白话问题转换为 1024 维度的向量条形码"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": self.embedding_model, "input": question}
            response = requests.post(self.embedding_api_base, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ 问题向量化失败: {e}")
            return None

    def local_search(self, question: str, top_k: int = 3):
        question_vector = self._get_question_embedding(question)
        if not question_vector:
            return "向量化失败，无法查询。"

        cypher_query = """
        CALL db.index.vector.queryNodes('local_entity_search_index', $k, $v)
        YIELD node AS center_node, score
        MATCH (center_node)-[r:RELATED]-(neighbor:Entity)
        RETURN 
            center_node.name AS 核心实体,
            score AS 匹配分数,
            neighbor.name AS 邻居节点,
            r.description AS 关系白话文描述,
            neighbor.description AS 邻居本身介绍
        """

        context_chunks = []
        with self.driver.session() as session:
            result = session.run(cypher_query, v=question_vector, k=top_k)

            print("\n⚡️ [Neo4j 内存震荡] 围绕问题成功检索到与以下图谱线索最匹配的上下文：")
            print("-" * 70)

            for record in result:
                print(
                    f"🎯 命中实体: 【{record['核心实体']}】(相似度: {record['匹配分数']:.4f}) "
                    f"===> 顺藤摸瓜抓到邻居: 【{record['邻居节点']}】"
                )
                chunk = (
                    f"已知实体线索: {record['核心实体']} 与 {record['邻居节点']} 存在关联。\n"
                    f"具体关联细节: {record['关系白话文描述']}\n"
                    f"邻居背景补充: {record['邻居本身介绍']}\n"
                    f"----------------------------------------"
                )
                context_chunks.append(chunk)

        return "\n".join(context_chunks)

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, object]:
        context = self.local_search(question, top_k=top_k)
        answer = self._generate_answer(question, context)
        return {"question": question, "top_k": top_k, "context": context, "answer": answer}

    def stream_answer(self, question: str, top_k: int = 3) -> Generator[str, None, None]:
        context = self.local_search(question, top_k=top_k)
        yield "data: {\"type\": \"context\", \"content\": " + self._json_escape(context) + "}\n\n"
        yield "data: {\"type\": \"start\"}\n\n"

        answer = self._generate_answer(question, context)
        for chunk in self._chunk_text(answer):
            yield "data: {\"type\": \"delta\", \"content\": " + self._json_escape(chunk) + "}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"
        yield "data: [DONE]\n\n"

    def _generate_answer(self, question: str, context: str) -> str:
        if not context or context == "向量化失败，无法查询。":
            return "抱歉，我暂时没能从图谱中检索到可用上下文。"

        prompt = (
            "你是一个基于知识图谱的问答助手。请仅根据给定上下文回答用户问题，\n"
            "如果上下文不足以支持结论，请明确说明不知道，不要编造。\n\n"
            f"用户问题：{question}\n\n"
            f"检索到的图谱上下文：\n{context}\n\n"
            "请输出一段简洁、准确、自然的中文回答。"
        )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": "你是一个严谨的中文知识图谱问答助手。"},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": 0.2,
            }
            response = requests.post(self.chat_api_base, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content:
                    return str(content).strip()
        except Exception as e:
            print(f"❌ 答案生成失败: {e}")

        return "已检索到上下文，但当前未能生成最终答案。"

    @staticmethod
    def _chunk_text(text: str, size: int = 40) -> Iterable[str]:
        for i in range(0, len(text), size):
            yield text[i : i + size]

    @staticmethod
    def _json_escape(text: str) -> str:
        return (
            "\""
            + text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "")
            + "\""
        )
