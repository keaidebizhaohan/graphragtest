import os

# 强行给硅基流动开辟绿色通道，彻底解决 Mac 代理软件引发的 503 向量化报错
os.environ['NO_PROXY'] = 'api.siliconflow.cn'


class Settings:
    # 1. Neo4j 数据库配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "12345678"
    # 2. 硅基流动 Embedding 模型配置
    SILICON_API_KEY: str = "sk-qbbgyitgrrdbnyaunuwuthezqtrslhtbjuhoukyotlojvjwr"
    EMBEDDING_API_BASE: str = "https://api.siliconflow.cn/v1/embeddings"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # 3. 硅基流动 大语言模型（LLM）对话配置
    # 这里我们选用更聪明的、适合做 GraphRAG 总结的 DeepSeek-V3 或者 R1
    LLM_API_BASE: str = "https://api.siliconflow.cn/v1/chat/completions"
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3"  # 也可以换成 deepseek-ai/DeepSeek-R1


settings = Settings()