import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

from search_service import GraphRAGLocalSearcher
from llm_service import LLMService

app = FastAPI(title="星河科技集团 - 工业级 GraphRAG 线上问答系统")

# 初始化全局单例组件
searcher = GraphRAGLocalSearcher()
llm_service = LLMService()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    context: str
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def graph_rag_chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")

    try:
        # 1. 派出一路兵马：打捞微观“实体拓扑网络” 局部检索
        local_context = searcher.local_search(request.question, top_k=3)

        # 2. 🚀 派出二路兵马：打捞宏观“社区白皮书报告”
        global_context = searcher.global_search(request.question, top_k=3)

        # 3. ⚖️ 黄金融合：强行组装无死角的超级参考上下文
        super_context = (
            f"=== 【第一部分：底层微观关系网络（细节）】 ===\n{local_context}\n\n"
            f"=== 【第二部分：高层宏观社区报告（大局）】 ===\n{global_context}"
        )

        # 4. 丢给大模型进行思维束缚和总结提炼
        final_answer = llm_service.generate_answer(request.question, super_context)

        # 文本脱水清洗
        clean_context = re.sub(r'\n{3,}', '\n\n', super_context.strip())
        clean_answer = re.sub(r'\n{3,}', '\n\n', final_answer.strip())

        return ChatResponse(
            question=request.question.strip(),
            context=clean_context,
            answer=clean_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部对账故障: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    searcher.close()
    print("🔌 已安全断开 Neo4j 图数据库连接池。")

if __name__ == "__main__":
    # ⚠️ 调试记得把 reload 改成 False，断点才会生效！
    uvicorn.run("main:app", host="127.0.0.1", port=8099, reload=False)