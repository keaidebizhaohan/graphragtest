import os

# 强行给硅基流动开辟绿色通道，彻底解决 Mac 代理软件引发的 503 向量化报错
os.environ['NO_PROXY'] = 'api.siliconflow.cn'


class Settings:
    # 1. Neo4j 数据库配置 (对应你刚刚用 Docker 满配拉起的全新容器)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "12345678"

    # 2. 统一密钥
    SILICON_API_KEY: str = "sk-qbbgyitgrrdbnyaunuwuthezqtrslhtbjuhoukyotlojvjwr"

    # 3. 硅基流动 Embedding 模型配置 (💥 核心修正：砍掉末尾的 /embeddings)
    EMBEDDING_API_BASE: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"  # 严格对齐 settings.yaml

    # 4. 硅基流动 大语言模型（LLM）配置 (💥 核心修正：砍掉末尾的 /chat/completions)
    LLM_API_BASE: str = "https://api.siliconflow.cn/v1"
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3"


settings = Settings()