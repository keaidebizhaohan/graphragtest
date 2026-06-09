import os
import pandas as pd
from neo4j import GraphDatabase
import requests

# 强行隔离代理，死守硅基流动通道
os.environ['NO_PROXY'] = 'api.siliconflow.cn'

# ==================== 1. 基础配置对账 ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")

API_KEY = "sk-qbbgyitgrrdbnyaunuwuthezqtrslhtbjuhoukyotlojvjwr"
API_BASE = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"


def get_embedding(text):
    if not text or not text.strip():
        return [0.0] * 1024
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": EMBEDDING_MODEL, "input": text}
        response = requests.post(API_BASE, json=payload, headers=headers, timeout=10)
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return [0.0] * 1024


def start_migration():
    print("📖 正在从本地磁盘读取 Parquet 成果物...")
    entities_df = pd.read_parquet("output/entities.parquet")
    relationships_df = pd.read_parquet("output/relationships.parquet")
    reports_df = pd.read_parquet("output/community_reports.parquet")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        print("🧹 正在清理 Neo4j 历史旧数据...")
        session.run("MATCH (n) DETACH DELETE n")

        # 1. 搬运实体节点（全属性解锁版）
        print("🚀 正在将实体洗成 :Entity 节点 (全属性包含)...")
        entity_rows = []
        for _, row in entities_df.iterrows():
            title_val = row.get("title", "")
            desc_val = row.get("description") or ""

            embedding_text = f"{title_val}: {desc_val}" if desc_val else title_val
            vector = get_embedding(embedding_text)

            entity_rows.append({
                "id": str(row.get("id")),
                "name": str(title_val),
                "description": str(desc_val),
                "type": str(row.get("type", "UNKNOWN")),
                "frequency": int(row.get("frequency", 1)),
                "degree": int(row.get("degree", 0)),
                "embedding": vector
            })

        session.run("""
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            SET e.name = row.name,
                e.description = row.description,
                e.type = row.type,
                e.frequency = row.frequency,
                e.degree = row.degree,
                e.description_embedding = row.embedding
        """, rows=entity_rows)
        print(f"✅ 成功导入 {len(entity_rows)} 个实体节点（属性已拉满）！")

        # 2. 搬运社区报告（full_content + summary 双剑合璧版）
        print("🚀 正在将社区总结洗成 :CommunityReport 节点 (包含完整内容与摘要)...")
        report_rows = []
        for _, row in reports_df.iterrows():
            full_content_val = row.get("full_content") or ""
            summary_val = row.get("summary") or ""
            title_val = row.get("title", "")

            # 使用 full_content 算向量，如果为空用 summary
            calc_text = full_content_val if full_content_val else summary_val
            vector = get_embedding(calc_text if calc_text else title_val)

            report_rows.append({
                "id": str(row.get("id")),
                "title": str(title_val),
                "full_content": str(full_content_val),
                "summary": str(summary_val),
                "level": int(row.get("level", 0)),
                "rating_explanation": str(row.get("rating_explanation", "")),
                "embedding": vector
            })

        session.run("""
            UNWIND $rows AS row
            MERGE (c:CommunityReport {id: row.id})
            SET c.title = row.title,
                c.full_content = row.full_content,
                c.summary = row.summary,
                c.level = row.level,
                c.rating_explanation = row.rating_explanation,
                c.content_embedding = row.embedding
        """, rows=report_rows)
        print(f"✅ 成功导入 {len(report_rows)} 个社区总结报告（双正文已齐备）！")

        # 3. 编织关系连线
        print("🔗 正在绘制知识图谱连线...")
        rel_rows = []
        for _, row in relationships_df.iterrows():
            source_val = row.get("source") or row.get("source_id")
            target_val = row.get("target") or row.get("target_id")
            rel_rows.append({
                "source": str(source_val),
                "target": str(target_val),
                "description": str(row.get("description", ""))
            })

        session.run("""
            UNWIND $rows AS row
            MATCH (source:Entity) WHERE source.id = row.source OR source.name = row.source
            MATCH (target:Entity) WHERE target.id = row.target OR target.name = row.target
            MERGE (source)-[r:RELATED]->(target)
            SET r.description = row.description
        """, rows=rel_rows)

        total_rels = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS c").single()["c"]
        print(f"✅ 成功拉起 {total_rels} 条知识图谱连线！")

    driver.close()
    print("🎉 【终极全能版】全属性、全正文、全连线无缝通关！")


if __name__ == "__main__":
    start_migration()