from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"

    siliconflow_api_key: str = "sk-qbbgyitgrrdbnyaunuwuthezqtrslhtbjuhoukyotlojvjwr"
    siliconflow_embedding_api_base: str = "https://api.siliconflow.cn/v1/embeddings"
    siliconflow_chat_api_base: str = "https://api.siliconflow.cn/v1/chat/completions"
    embedding_model: str = "BAAI/bge-m3"
    chat_model: str = "deepseek-ai/DeepSeek-V3"


settings = Settings()
