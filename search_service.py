import os
from neo4j import GraphDatabase
import requests
from config import settings


# ==========================================
# 图谱网络局部打捞微服务组件 (GraphRAG Local Searcher)
# ==========================================
# 职责：负责接收用户的原始问题，利用向量中台获取其特征向量，
# 随后进入 Neo4j 触发高性能向量图遍历，顺藤摸瓜打捞 1-2 步深度因果网，为 LLM 提供结构化事实证据。
class GraphRAGLocalSearcher:
    def __init__(self):
        """
        构造函数：图数据库 Driver 基础设施初始化
        基于 Spring-like 单例思想，直接从统一的 config.py 调取连接凭证。
        在全局维护长连接池，避免每次 HTTP 请求进来时频繁触发 TCP 三次握手建连，压榨系统 IO 性能。
        """
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        self.api_key = settings.SILICON_API_KEY
        self.api_base = settings.EMBEDDING_API_BASE
        self.embedding_model = settings.EMBEDDING_MODEL

    def close(self):
        """
        释放资源回调：在 Web 容器销毁（FastAPI Shutdown）时触发，
        优雅关闭 Neo4j 驱动连接池，安全释放底层的物理套接字。
        """
        self.driver.close()

    def _get_question_embedding(self, question: str):
        """
        内部辅助方法：调用文本嵌入向量中台（Text Embedding Pipeline）
        将用户的大白话转义为符合向量货架维度的 1024 维高维浮点数稠密向量（Dense Vector）。
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"model": self.embedding_model, "input": question}
            # 刚性设置 10 秒网络超时阻断，防止上游响应死锁拖垮当前 Service 线程
            response = requests.post(self.api_base, json=payload, headers=headers, timeout=10)
            response.raise_for_status()  # 状态码异常卡口对账
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            # 🛡️ 异常防线：捕获网络抖动或鉴权错位，打印第一手追溯日志并执行 Fail-Fast 阻断
            print(f"❌ 问题向量化失败: {e}")
            return None

    def local_search(self, question: str, top_k: int = 3) -> str:
        """
        核心业务方法：局部图谱向量混合路径打捞（Hybrid Paths Search）

        参数:
            question (str): 前端传入的用户原始提问
            top_k (int): 向量检索初步锚定的核心实体候选集数量，默认为 3

        返回:
            str: 拼接好、洗数脱水后的高价值图谱事实上下文大字符串
        """
        # 1. 触发前置流水线，获取问题的向量条形码
        question_vector = self._get_question_embedding(question)
        if not question_vector:
            return ""

        # 2. 🎯 焊死工业级混合检索 Cypher 语句（包含 1-2 步跨节点拓扑路径打捞）
        # 核心设计：
        # - Step 1: 调用经典向量存储过程 `queryNodes`，在毫秒级内用余弦相似度定位到最相关的中心点（center_node）。
        # - Step 2: 以此中心点为原点，使用 `MATCH path` 磁铁般向外疯狂辐射 1 到 2 步深度（*1..2），把隐藏在深处的连线全盘端出。
        # - 💡 开发备忘（白天删除剪枝拓展点）：
        #   若要在白天平滑过滤已逻辑删除的文本，直接在下方的 WHERE 条件里追加针对关系连线的断桥逻辑：
        #   WHERE ALL(rel IN relationships(path) WHERE NOT "今日黑名单ID" IN rel.source_doc_ids)
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

        # 3. 开启数据库轻量级只读 Session 会话（Read-Only Transaction）
        with self.driver.session() as session:
            # 传入向量和 K 值参数，轰炸 Neo4j 内存引擎
            result = session.run(cypher_query, v=question_vector, k=top_k)

            # 4. 遍历游标账单，进行数据提取与拼装
            for record in result:
                # 🛠️ 核心拼装逻辑：由于我们升级到了 1-2 步的多步变长路径，捞出来的关系描述（description）是一个数组。
                # 运用 Python 推导式动态提取所有合法的白话文描述，并用 " -> " 物理符号将其穿针引线串成故事线。
                rel_list = record['关系链描述']
                rel_story = " -> ".join([str(desc) for desc in rel_list if desc])

                # 5. 打包塑造成规整的结构化白话文线索块（Context Chunk）
                chunk = (
                    f"已知实体线索: {record['核心实体']} 与 {record['邻居节点']} 存在 1-2 步内的间接或直接关联。\n"
                    f"链路关联细节: {rel_story}\n"
                    f"邻居背景补充: {record['邻居背景描述']}\n"
                    f"----------------------------------------"
                )
                context_chunks.append(chunk)

        # 6. 用标准回车换行符将所有的线索块“大聚合”，交卷给上游 API 展现层
        return "\n".join(context_chunks)