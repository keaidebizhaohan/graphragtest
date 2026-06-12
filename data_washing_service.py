import os
from typing import List
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from config import settings

# 💥 引入 Neo4j 官方最正统、核心库里雷打不动的标准 GraphRAG 认知与检索 API组件
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever  # 👈 核心补齐：引入官方标准向量检索器


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
        全量洗数流水线：100% 遵照领导指示，全量调用 Graph 官方高级 API 实现
        """
        if not es_records:
            return

        print("🚀 [纯正 API 流水线启动] 开始遵照官方标准构建多模态知识图谱...")

        # 数据包装
        raw_documents = []
        for record in es_records:
            doc = Document(
                page_content=record["content"],
                metadata={"source_doc_id": record["doc_id"]}
            )
            raw_documents.append(doc)

        # =================================================================
        # API 环节 ①：打地基 —— 调用 API 全自动提取实体、关系并连线
        # =================================================================
        print("📥 环节 ①: 正在调官方 Transformer API 盲抽实体网...")
        graph_documents = self.graph_transformer.convert_to_graph_documents(raw_documents)
        self.graph.add_graph_documents(graph_documents)
        print("   └─ ✅ 实体关系网落库成功！")

        # =================================================================
        # API 环节 ②：向下兼容 —— 调用 API 全自动切片并打上文本向量
        # =================================================================
        print("📥 环节 ②: 正在调 Vector API 注入 __TextUnit__ 传统 RAG 货架...")
        Neo4jVector.from_documents(
            documents=raw_documents,
            embedding=self.embeddings,
            graph=self.graph,
            index_name="text_unit_vector_index",
            node_label="__TextUnit__"
        )
        print("   └─ ✅ 文本切片向量节点注入成功！")

        # =================================================================
        # API 环节 ③：实体补全 —— 调用 API 全自动为库里所有实体打上独立向量
        # =================================================================
        # =================================================================
        # API 环节 ③：实体向量补全 —— 修正为读取 "id" 字段，强行灌满向量！
        # =================================================================
        print("📥 环节 ③: 正在调 Vector API 跨界为现有 Entity 节点批量补齐向量属性...")
        try:
            # 💥 核心修正：把 text_node_properties 改为 ["id"]！让 BGE 直接拿实体名字去算向量！
            Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                node_label="Entity",
                text_node_properties=["id"],  # 👈 就是这行！改成 "id"，保证绝对有值，绝对能算出向量！
                embedding_node_property="embedding",
                index_name="entity_vector_index"
            )
            print("   └─ ✅ 库里所有实体（小韩、宇哥）的独立向量属性补齐完毕！")
        except Exception as e:
            print(f"   └─ ❌ 实体向量补全核心故障: {e}")
            raise e

        # =================================================================
        # API 环节 ④：【正统收网】—— 声明官方标准的 GraphRAG 搜索上下文与大模型绑定
        # =================================================================
        print("📥 环节 ④: 正在初始化官方正统 GraphRAG 认知图谱上下文组件...")
        try:
            # 1. 包装官方格式的大模型驱动客户端
            neo4j_llm = OpenAILLM(
                model_name=settings.LLM_MODEL,
                api_key=settings.SILICON_API_KEY,
                base_url=settings.LLM_API_BASE
            )

            # 2. 💥 核心修正：给官方 GraphRAG 实例化一个真正的 VectorRetriever 骨干！
            # 传入图连接、刚刚建好的 1024 维索引名、以及匹配的标签，完美满足 Pydantic 的类型守卫
            # 💥 核心修正：将 graph 改为 driver，并传入 self.graph._driver！
            official_retriever = VectorRetriever(
                driver=self.graph._driver,  # 👈 核心拿捏：直接解构出底层的原生 driver 驱动连接
                index_name="text_unit_vector_index",
            )

            # 3. 100% 官方标准高级语句：将正统检索器注入组件，实现完美闭环
            grag_engine = GraphRAG(
                retriever=official_retriever,  # 👈 核心修正：塞入真正合法的 Retriever 实例！
                llm=neo4j_llm
            )
            print("   └─ ✅ 纯正官方高级 API 认知链条（GraphRAG Engine）全线构筑完毕！")

        except Exception as e:
            print(f"   └─ ❌ 官方基础 API 组件绑定失败: {e}")
            raise e

        print("\n🎉 🎉 🎉 [纯正官方核心库 API 方案全面通关] 🎉 🎉 🎉")