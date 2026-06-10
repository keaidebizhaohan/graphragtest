import os
from neo4j import GraphDatabase
import requests
from config import settings

#图谱打捞服务类
class GraphRAGLocalSearcher:
    def __init__(self):
        """从统一的 config 注入连接信息"""
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        self.api_key = settings.SILICON_API_KEY
        self.api_base = settings.EMBEDDING_API_BASE
        self.embedding_model = settings.EMBEDDING_MODEL

    def close(self):
        self.driver.close()

    def _get_question_embedding(self, question: str):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"model": self.embedding_model, "input": question}
            response = requests.post(self.api_base, json=payload, headers=headers, timeout=10)
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ 问题向量化失败: {e}")
            return None

    def local_search(self, question: str, top_k: int = 3) -> str:
        """局部搜索：升级为 1-2 步图谱路径深度探测，打捞隐藏技术栈"""
        question_vector = self._get_question_embedding(question)
        if not question_vector:
            return ""

        # 🎯 深度匹配 1 到 2 步关联的点和线
        cypher_query = """
        CALL db.index.vector.queryNodes('local_entity_search_index', $k, $v)
        YIELD node AS center_node, score
        MATCH path = (center_node)-[r:RELATED*1..2]-(neighbor:Entity)
        RETURN 
            center_node.name AS 核心实体,
            score AS 匹配分数,
            neighbor.name AS 邻居节点,
            [rel in relationships(path) | rel.description] AS 关系链描述,
            neighbor.description AS 邻居背景描述
        LIMIT 20
        """

        context_chunks = []
        with self.driver.session() as session:
            result = session.run(cypher_query, v=question_vector, k=top_k)
            for record in result:
                # 🛠️ 把多步关系描述数组通过 " -> " 拼接成连贯的故事线
                rel_list = record['关系链描述']
                rel_story = " -> ".join([str(desc) for desc in rel_list if desc])

                chunk = (
                    f"已知实体线索: {record['核心实体']} 与 {record['邻居节点']} 存在 1-2 步内的间接或直接关联。\n"
                    f"链路关联细节: {rel_story}\n"
                    f"邻居背景补充: {record['邻居背景描述']}\n"
                    f"----------------------------------------"
                )
                context_chunks.append(chunk)

        return "\n".join(context_chunks)