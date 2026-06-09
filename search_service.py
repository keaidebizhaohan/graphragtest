import os
from neo4j import GraphDatabase
import requests

# 强行给硅基流动开辟绿色通道，无视 Mac 代理软件，彻底防死 503 报错
os.environ['NO_PROXY'] = 'api.siliconflow.cn'


class GraphRAGLocalSearcher:
    def __init__(self):
        """初始化：对接本地 Neo4j 数据库和硅基流动 API"""
        # 1. 绑定 Neo4j 连接指针
        self.uri = "bolt://localhost:7687"
        self.auth = ("neo4j", "12345678")
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)

        # 2. 绑定硅基流动大模型密钥
        self.api_key = "sk-qbbgyitgrrdbnyaunuwuthezqtrslhtbjuhoukyotlojvjwr"
        self.api_base = "https://api.siliconflow.cn/v1/embeddings"
        self.embedding_model = "BAAI/bge-m3"

    def close(self):
        """优雅关闭数据库连接"""
        self.driver.close()

    def _get_question_embedding(self, question: str):
        """私有方法：现场将用户输入的大白话问题转换为 1024 维度的向量条形码"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.embedding_model,
                "input": question
            }
            response = requests.post(self.api_base, json=payload, headers=headers, timeout=10)
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ 问题向量化失败: {e}")
            return None

    def local_search(self, question: str, top_k: int = 3):
        """
        核心公开方法：局部搜索（Local Mode）
        1. 向量定位最相关的实体
        2. 顺藤摸瓜（1步遍历）抓出周围所有的关系文本与邻居描述
        """
        # 1. 把问题转成向量
        question_vector = self._get_question_embedding(question)
        if not question_vector:
            return "向量化失败，无法查询。"

        # 2. 纯 Cypher 降维打击（一行精妙的 Cypher 搞定向量比对+全量图遍历）
        # 这里利用了我们之前在 Neo4j 里焊死的 local_entity_search_index 索引货架
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

        # 3. 发射 Cypher 语句进入 Neo4j 内存世界
        with self.driver.session() as session:
            result = session.run(cypher_query, v=question_vector, k=top_k)

            print(f"\n⚡️ [Neo4j 内存震荡] 围绕问题成功检索到与以下图谱线索最匹配的上下文：")
            print("-" * 70)

            for record in result:
                # 打印出人性化的 debug 信息，让你看清打捞过程
                print(f"🎯 命中实体: 【{record['核心实体']}】(相似度: {record['匹配分数']:.4f}) "
                      f"===> 顺藤摸瓜抓到邻居: 【{record['邻居节点']}】")

                # 拼接成准备喂给大模型的完美大白话上下文（Context）
                chunk = (
                    f"已知实体线索: {record['核心实体']} 与 {record['邻居节点']} 存在关联。\n"
                    f"具体关联细节: {record['关系白话文描述']}\n"
                    f"邻居背景补充: {record['邻居本身介绍']}\n"
                    f"----------------------------------------"
                )
                context_chunks.append(chunk)

        # 4. 把打捞出来的所有小碎块拼成一个完美的长文本
        perfect_context = "\n".join(context_chunks)
        return perfect_context


# ==================== 🎬 模拟线上用户提问的主函数 ====================
if __name__ == "__main__":
    # 1. 实例化我们的全新搜索类
    searcher = GraphRAGLocalSearcher()

    # 2. 模拟线上用户的抠细节提问（局部搜索 Local Mode 最擅长的领域）
    user_question = "中国银行（香港）和星河科技有什么业务往来？小韩在里面负责什么？"
    print(f"🤖 线上用户真实提问: '{user_question}'")
    print("⏳ 正在调用硅基流动转换问题向量并请求图谱，请稍候...")

    # 3. 一击必杀，顺藤摸瓜
    final_llm_context = searcher.local_search(user_question, top_k=2)

    print("-" * 70)
    print("🎁 [最终交卷] 吐给大模型（DeepSeek）的完美上下文（Context）:")
    print("-" * 70)
    if final_llm_context:
        print(final_llm_context)
    else:
        print("🕳 哎呀，图谱库里空空如也，什么都没捞着！")

    # 4. 业务结束，关闭数据库
    searcher.close()