from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from config import settings

# ==========================================================
# 图谱网络局部打捞微服务组件 (100% 纯官方高阶 API 驱动版)
# ==========================================================
class GraphRAGLocalSearcher:
    # 核心四驾马车参数，死死咬合洗数端
    VECTOR_INDEX_NAME = "text_unit_vector_index"
    NODE_LABEL = "__TextUnit__"  # 严格对齐洗数端的双下划线标签
    EMBEDDING_PROPERTY = "embedding"
    TEXT_PROPERTIES = ["text"]  # 严格对齐 LangChain 默认存储字段

    def __init__(self):
        """
        初始化高阶向量图谱检索器
        💥 彻底干掉了 _ensure_vector_index() 这种手写驱动函数！
        """
        # 1. 声明向量中台模型
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.SILICON_API_KEY,
            openai_api_base=settings.EMBEDDING_API_BASE,
            model=settings.EMBEDDING_MODEL
        )

        # 2. 🚀 终极标准 API 语句：一行代码，包办【检查索引是否存在】、【自动建 1024 维货架】、【加载长连接】
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            index_name=self.VECTOR_INDEX_NAME, # 传入索引名
            node_label=self.NODE_LABEL, # 传入标签名
            embedding_node_property=self.EMBEDDING_PROPERTY, # 传入向量属性名
            text_node_properties=self.TEXT_PROPERTIES, # 传入文本属性名
        )

    def close(self):
        """连接池已由高阶 API 自动托管，此处留空保持接口兼容"""
        pass

    def local_search(self, question: str, top_k: int = 3) -> str:
        """
        局部检索：纯 API 驱动
        自动包办：问题语义向量化 -> 拓扑网络相似度打捞
        """
        docs = self.vector_store.similarity_search(question, k=top_k)

        context_chunks = []
        for doc in docs:
            chunk = (
                f"【图谱切片线索】:\n{doc.page_content}\n"
                f"事实出处文档ID: {doc.metadata.get('source_doc_id', '未知')}\n"
                f"----------------------------------------"
            )
            context_chunks.append(chunk)

        return "\n".join(context_chunks)

    def global_search(self, question: str, top_k: int = 3) -> str:
        """全局检索预留"""
        return ""