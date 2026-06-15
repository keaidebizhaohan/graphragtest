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
from graphdatascience import GraphDataScience
import pandas as pd


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

        # 💥 初始化原生 GDS Python 客户端
        self.gds = GraphDataScience(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

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
            # 这里的轻量 Cypher 是必须的胶水层，用来做精准的字符串包含匹配
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
        # 环节 3.5：宏观社区聚合与预制报告 (Leiden + GDS)
        # 必须与环节 2.5 同级缩进；写在 2.5 的 except 里时，链接成功则整段不执行。
        # =================================================================
        print("�� 环节 3.5: 正在启动官方 GDS 引擎进行全图宏观社区聚类...")
        G = None
        try:
            try:
                if self.gds.graph.exists("community_graph").exists:
                    self.gds.graph.drop("community_graph")
            except Exception:
                pass

            entity_count = self.graph.query(
                "MATCH (e:__Entity__) RETURN count(e) AS c"
            )[0]["c"]
            if entity_count < 2:
                print(
                    f"   └─ ⚠️ 实体仅 {entity_count} 个，跳过 Leiden 与社区报告。"
                )
            else:
                co = self.graph.query(
                    """
                    MATCH (t:__TextUnit__)-[:HAS_ENTITY]->(e1:__Entity__),
                          (t)-[:HAS_ENTITY]->(e2:__Entity__)
                    WHERE elementId(e1) < elementId(e2)
                    MERGE (e1)-[r:CO_OCCURS_IN_UNIT]->(e2)
                    RETURN count(r) AS c
                    """
                )
                co_count = co[0]["c"] if co else 0
                print(
                    f"   └─ 🔗 共现边 CO_OCCURS_IN_UNIT 就绪（本批 {co_count} 条）"
                )

                rel_types = self.graph.query(
                    "MATCH (:__Entity__)-[r]->(:__Entity__) "
                    "RETURN DISTINCT type(r) AS rel_type"
                )
                rel_projection = {
                    row["rel_type"]: {"orientation": "UNDIRECTED"}
                    for row in rel_types
                }
                if not rel_projection:
                    print(
                        "   └─ ⚠️ 仍无 __Entity__→__Entity__ 关系，无法 GDS 投影，跳过社区报告。"
                    )
                else:
                    G, _ = self.gds.graph.project(
                        "community_graph",
                        "__Entity__",
                        rel_projection,
                    )
                    print(
                        f"   └─ 📊 GDS 投影: {G.node_count()} 节点, "
                        f"{G.relationship_count()} 条无向边"
                    )

                    community_df = self.gds.leiden.stream(G)
                    nodes = self.gds.util.asNodes(
                        community_df["nodeId"].to_list()
                    )
                    community_df["entity_name"] = [
                        node.get("id") or node.get("name") for node in nodes
                    ]

                    grouped_communities = (
                        community_df.groupby("communityId")["entity_name"]
                        .apply(list)
                        .reset_index()
                    )
                    multi = grouped_communities[
                        grouped_communities["entity_name"].apply(len) >= 2
                    ]
                    print(
                        f"   └─ �� Leiden 共 {len(grouped_communities)} 个社区，"
                        f"≥2 实体的 {len(multi)} 个，正在生成报告..."
                    )

                    text_vector_store = Neo4jVector.from_existing_index(
                        embedding=self.embeddings,
                        url=settings.NEO4J_URI,
                        username=settings.NEO4J_USER,
                        password=settings.NEO4J_PASSWORD,
                        index_name="text_unit_vector_index",
                        text_node_property="text",
                    )

                    report_docs = []
                    for _, row in multi.iterrows():
                        comm_id = row["communityId"]
                        entity_list = [
                            str(x) for x in row["entity_name"] if x
                        ]
                        if len(entity_list) < 2:
                            continue

                        search_query = " ".join(entity_list)
                        context_docs = text_vector_store.similarity_search(
                            search_query, k=5
                        )
                        context_text = "\n".join(
                            d.page_content for d in context_docs
                        )

                        report_prompt = f"""
你是一个精通系统架构与组织关系的宏观战略分析师。
请为以下知识图谱中的一个特定【实体社区/圈子】撰写一份结构化的【社区宏观总结白皮书】。

【本社区包含的核心实体名单】: {", ".join(entity_list)}
【本社区的背景知识参考】:
{context_text}

请直接输出报告正文，包含【社区主题】和【核心概要】，不要废话。
"""
                        response = self.llm.invoke(report_prompt)
                        report_docs.append(
                            Document(
                                page_content=response.content,
                                metadata={
                                    "community_id": str(comm_id),
                                    "entities_involved": ", ".join(
                                        entity_list
                                    ),
                                },
                            )
                        )

                    if report_docs:
                        Neo4jVector.from_documents(
                            documents=report_docs,
                            embedding=self.embeddings,
                            graph=self.graph,
                            index_name="community_report_index",
                            node_label="__CommunityReport__",
                        )
                        print(
                            f"   └─ ✅ {len(report_docs)} 份社区报告已写入 "
                            "__CommunityReport__"
                        )
                    else:
                        print(
                            "   └─ ⚠️ 未生成报告（社区均 <2 实体或实体名为空）。"
                        )

        except Exception as e:
            print(f"   └─ ❌ 官方 GDS 社区聚类故障: {e}")
            raise
        finally:
            if G is not None:
                try:
                    self.gds.graph.drop(G)
                except Exception:
                    pass

        # =================================================================
        # 💥 API 环节 ③：实体向量补全 —— 回归官方高阶 API！
        # =================================================================
        print("📥 环节 ③: 正在调 Vector API 跨界为现有 Entity 节点批量补齐向量属性...")
        try:
            Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                node_label="__Entity__",
                text_node_properties=["id"],
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

        print("\n🎉 🎉 🎉 [纯 API + GDS 宏观社区架构方案 全面通关] 🎉 🎉 🎉")