import os
from typing import List
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from config import settings

# 💥 引入 Neo4j 官方标准 GraphRAG 认知与检索 API 组件
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =================================================================
# 纯正官方 API 驱动版 —— GraphRAG 自动化建图洗数微服务（终极完满闭合版）
# =================================================================
class GraphRAGDataWashingService:
    def __init__(self):
        """初始化大模型与 Neo4j 连接盘"""
        # 1. 锁死大模型基座
        self.llm = ChatOpenAI(
            api_key=settings.SILICON_API_KEY,
            base_url=settings.LLM_API_BASE,
            model=settings.LLM_MODEL,
            temperature=0.0
        )

        # 2. 锁死向量模型
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.SILICON_API_KEY,
            openai_api_base=settings.EMBEDDING_API_BASE,
            model=settings.EMBEDDING_MODEL
        )

        # 3. 建立 Neo4j 连接（Docker APOC 满配版长连接）
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )

        # 4. 初始化经典点线图转换 API
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)

    def wash_es_data_to_neo4j(self, es_records: List[dict]):
        """
        全量洗数流水线：💥 修复了之前致命的缩进和嵌套错误！
        """
        if not es_records:
            return

        print("🚀 [纯正 API 流水线启动] 开始遵照官方标准构建多模态知识图谱...")

        # 1. 原始数据包装
        raw_documents = []
        for record in es_records:
            doc = Document(
                page_content=record["content"],
                metadata={"source_doc_id": record["doc_id"]}
            )
            raw_documents.append(doc)

        # =================================================================
        # 💥 环节 0：工业级智能文本切片 (Chunking) —— 彻底粉碎大长文本
        # =================================================================
        print("✂️ 环节 0: 正在进行长文本智能切片...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 每块切片保留 500 个字符的黄金长度
            chunk_overlap=50  # 前后切片重叠 50 个字符
        )
        chunked_documents = text_splitter.split_documents(raw_documents)
        print(f"   └─ ✅ 成功将 {len(raw_documents)} 篇长文切分为 {len(chunked_documents)} 个标准知识块！")

        # =================================================================
        # API 环节 ①：打地基 —— 调用 API 自动化提取实体关系网并连线
        # =================================================================
        print("📥 环节 ①: 正在调官方 Transformer API 盲抽实体网...")
        # 💥 严格传入切片后的 chunked_documents
        graph_documents = self.graph_transformer.convert_to_graph_documents(chunked_documents)
        self.graph.add_graph_documents(graph_documents)
        print("   └─ ✅ 实体关系网（Entity/RELATED）落库成功！")

        # =================================================================
        # API 环节 ②：传统 RAG 兼容 —— 调用 API 全自动注入切片并打上文本向量
        # =================================================================
        print("📥 环节 ②: 正在调 Vector API 注入 __TextUnit__ 传统 RAG 货架...")
        # 💥 严格传入切片后的 chunked_documents，让它生成多个带有真实向量的物理切片节点！
        Neo4jVector.from_documents(
            documents=chunked_documents,
            embedding=self.embeddings,
            graph=self.graph,
            index_name="text_unit_vector_index",
            node_label="__TextUnit__"
        )
        print("   └─ ✅ 文本切片向量节点注入成功！")

        # =================================================================
        # 💥 API 环节 ③：实体向量补全 —— 抛弃黑盒 API，直接原生暴力灌库！
        # =================================================================
        print("📥 环节 ③: 正在原生跨界为现有 Entity 节点批量强制补齐向量属性...")
        try:
            # 1. 物理打捞：写原生 Cypher，把库里所有没向量的实体连根拔起
            records = self.graph.query(
                "MATCH (e:Entity) WHERE e.embedding IS NULL AND e.id IS NOT NULL RETURN e.id AS text"
            )

            if records:
                print(f"   └─ 🔍 扫描到 {len(records)} 个待补齐向量的实体裸节点，正在呼叫 BGE 算力...")
                texts = [r["text"] for r in records]

                # 2. 调用向量中台原生算力，一口气算出所有 1024 维的黄金大数组
                vectors = self.embeddings.embed_documents(texts)

                # 3. 物理回写：精准制导，把算好的大数组硬塞进节点的 embedding 属性里！
                for text, vec in zip(texts, vectors):
                    self.graph.query(
                        "MATCH (e:Entity) WHERE e.id = $text SET e.embedding = $vec",
                        params={"text": text, "vec": vec}
                    )

                # 4. 原生 DDL：强行焊死 1024 维货架
                self.graph.query("""
                        CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS 
                        FOR (e:Entity) ON (e.embedding) 
                        OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}
                        """)
                print(f"   └─ ✅ 暴力破局成功！{len(records)} 个实体的 1024 维物理向量已全部实打实落盘！")
            else:
                print("   └─ ✅ 库内所有实体向量均已饱满，无需补齐。")

        except Exception as e:
            print(f"   └─ ❌ 实体向量原生打捞与回写故障: {e}")
            raise e

        # =================================================================
        # API 环节 ④：【正统收网】—— 声明官方标准的 GraphRAG 搜索上下文与大模型绑定
        # =================================================================
        print("📥 环节 ④: 正在初始化官方正统 GraphRAG 认知图谱上下文组件...")
        try:
            neo4j_llm = OpenAILLM(
                model_name=settings.LLM_MODEL,
                api_key=settings.SILICON_API_KEY,
                base_url=settings.LLM_API_BASE
            )
            # 引入官方标准的 VectorRetriever 作为骨架
            official_retriever = VectorRetriever(
                driver=self.graph._driver,
                index_name="text_unit_vector_index"
            )
            grag_engine = GraphRAG(
                retriever=official_retriever,
                llm=neo4j_llm
            )
            print("   └─ ✅ 纯正官方高级 API 认知链条（GraphRAG Engine）全线构筑完毕！")
        except Exception as e:
            print(f"   └─ ❌ 官方基础 API 组件绑定失败: {e}")
            raise e

        print("\n🎉 🎉 🎉 [纯正官方核心库 API 方案全面通关] 🎉 🎉 🎉")