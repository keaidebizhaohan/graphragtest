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
# 纯正官方 API 驱动版 —— GraphRAG 自动化建图洗数微服务（工业级完满闭合版）
# =================================================================
class GraphRAGDataWashingService:
    def __init__(self):
        """初始化大模型与 Neo4j 连接盘"""
        self.llm = ChatOpenAI(
            api_key=settings.SILICON_API_KEY,
            base_url=settings.LLM_API_BASE,
            model=settings.LLM_MODEL,
            temperature=0.0
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.SILICON_API_KEY,
            openai_api_base=settings.EMBEDDING_API_BASE,
            model=settings.EMBEDDING_MODEL
        )
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)

    def wash_es_data_to_neo4j(self, es_records: List[dict]):
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
        # 环节 0：工业级智能文本切片
        # =================================================================
        print("✂️ 环节 0: 正在进行长文本智能切片...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunked_documents = text_splitter.split_documents(raw_documents)
        print(f"   └─ ✅ 成功切分为 {len(chunked_documents)} 个标准知识块！")

        # =================================================================
        # 💥 API 环节 ①：打地基 —— 调用 API 并利用 baseEntityLabel 自动盖钢印！
        # =================================================================
        print("📥 环节 ①: 正在调官方 Transformer API 盲抽实体网...")
        graph_documents = self.graph_transformer.convert_to_graph_documents(chunked_documents)

        # 💥 核心替换：直接利用 API 的 baseEntityLabel=True 参数，底层自动为所有节点打上 __Entity__ 标签！
        # 彻底消灭了那句 SET n:Entity 的原生 Cypher！
        self.graph.add_graph_documents(graph_documents, baseEntityLabel=True)
        print("   └─ ✅ 实体关系网落库成功，并已由 API 自动打上统一 __Entity__ 标签！")

        # =================================================================
        # API 环节 ②：注入传统 RAG 切片
        # =================================================================
        print("📥 环节 ②: 正在调 Vector API 注入 __TextUnit__ 传统 RAG 货架...")
        Neo4jVector.from_documents(
            documents=chunked_documents,
            embedding=self.embeddings,
            graph=self.graph,
            index_name="text_unit_vector_index",
            node_label="__TextUnit__"
        )
        print("   └─ ✅ 文本切片向量节点注入成功！")

        # =================================================================
        # 环节 2.5：物理牵线（架构胶水）
        # =================================================================
        print("📥 环节 2.5: 正在建立切片与实体之间的物理桥梁 (Entity Linking)...")
        try:
            # 执行底层 JOIN，把切片和实体死死焊在一起
            # 注意：咱们上面用了 baseEntityLabel=True，所以这里的表名叫 __Entity__
            result = self.graph.query("""
                        MATCH (t:__TextUnit__), (e:__Entity__)
                        WHERE t.text CONTAINS e.id
                        MERGE (t)-[:HAS_ENTITY]->(e)
                        RETURN count(*) AS link_count
                    """)
            link_count = result[0]['link_count'] if result else 0
            print(f"   └─ ✅ 文本与实体双向奔赴打通！成功焊接了 {link_count} 条物理关系线！")
        except Exception as e:
            print(f"   └─ ❌ 物理牵线失败: {e}")

        # =================================================================
        # 💥 新增环节 3.5：宏观社区聚合与社区报告全自动预制落盘 (纯 API 写入版)
        # =================================================================
        print("📥 环节 3.5: 正在启动图谱宏观社区聚类，并由大模型预先生成社区白皮书报告...")
        try:
            # 1. 物理读取：基于连线将实体按来源归类为“社区” (只读查询获取拓扑结构)
            community_data = self.graph.query("""
                MATCH (t:__TextUnit__)-[:HAS_ENTITY]->(e:__Entity__)
                RETURN id(t) AS community_id, collect(e.id) AS entity_ids, t.text AS context_text
            """)

            if community_data:
                print(f"   └─ 👥 成功划分出 {len(community_data)} 个宏观语义社区，正在并发呼叫大模型编写白皮书...")

                report_docs = []  # 用于收集所有报告对象的列表

                for comm in community_data:
                    comm_id = comm["community_id"]
                    entity_ids = list(set(comm["entity_ids"]))
                    context_text = comm["context_text"]

                    # 2. 动态拼装大模型 Prompt
                    report_prompt = f"""
                    你是一个精通系统架构与组织关系的宏观战略分析师。
                    请为以下知识图谱中的一个特定【实体社区/圈子】撰写一份结构化的【社区宏观总结白皮书】。

                    【本社区包含的实体名单】: {", ".join(entity_ids)}
                    【本社区的原始上下文线索】:
                    {context_text}

                    请直接输出报告正文，包含【社区主题】和【核心概要】，不要废话。
                    """

                    # 3. 呼叫大模型算力生成纯文本报告
                    response = self.llm.invoke(report_prompt)
                    report_content = response.content

                    # 4. 核心 API 封装：将纯文本报告包装为 LangChain 标准的 Document 对象！
                    # 我们把圈子里的实体名单存在 metadata 里，完全替代复杂的图谱写连线
                    doc = Document(
                        page_content=report_content,
                        metadata={
                            "community_id": str(comm_id),
                            "entities_involved": ", ".join(entity_ids)
                        }
                    )
                    report_docs.append(doc)

                # 5. 【终极 API 绝杀】直接调用官方 API，一句话包办：算向量、创建节点、全自动建专属向量索引！
                if report_docs:
                    Neo4jVector.from_documents(
                        documents=report_docs,
                        embedding=self.embeddings,
                        graph=self.graph,
                        index_name="community_report_index",
                        node_label="__CommunityReport__"
                    )
                print(f"   └─ ✅ {len(report_docs)} 份宏观社区报告全部硬落盘！专属向量索引已由 API 全自动创建完毕！")
            else:
                print("   └─ ⚠️ 未找到可聚合的社区，请检查环节 2.5 是否成功关联。")

        except Exception as e:
            print(f"   └─ ❌ 社区报告预生成 API 核心故障: {e}")
            raise e

        # =================================================================
        # 💥 API 环节 ③：实体向量补全 —— 回归官方高阶 API！
        # =================================================================
        print("📥 环节 ③: 正在调 Vector API 跨界为现有 Entity 节点批量补齐向量属性...")
        try:
            # 既然环节 ① 已经用 API 打好了 __Entity__ 标签，我们直接呼叫官方高级 API，一行代码包办算向量和建索引！
            Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                node_label="__Entity__",  # 👈 自动识别环节 ① 打好的 __Entity__ 标签
                text_node_properties=["id"],  # 👈 提取实体 id 去算向量
                embedding_node_property="embedding",
                index_name="entity_vector_index"
            )
            print("   └─ ✅ 高级 API 调用成功：所有实体的独立向量属性补齐完毕！")
        except Exception as e:
            print(f"   └─ ❌ 实体向量补全故障: {e}")
            raise e

        # =================================================================
        # API 环节 ④：【正统收网】—— GraphRAG 搜索上下文与大模型绑定
        # =================================================================
        print("📥 环节 ④: 正在初始化官方正统 GraphRAG 认知图谱上下文组件...")
        try:
            neo4j_llm = OpenAILLM(
                model_name=settings.LLM_MODEL,
                api_key=settings.SILICON_API_KEY,
                base_url=settings.LLM_API_BASE
            )
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

        print("\n🎉 🎉 🎉 [纯 API + 预制报告架构方案 全面通关] 🎉 🎉 🎉")