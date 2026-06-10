import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 引入我们前几步封装好的服务组件
from search_service import GraphRAGLocalSearcher
from llm_service import LLMService
import re  # 引入正则


# 1. 初始化 FastAPI 容器
app = FastAPI(
    title="星河科技集团 - 工业级 GraphRAG 线上问答系统",
    description="基于 FastAPI + Neo4j 向量图谱 + DeepSeek-V3 的生产级 RAG 闭环"
)

# 2. 全局单例实例化，避免每次接口请求都重复创建连接池
searcher = GraphRAGLocalSearcher()
llm_service = LLMService()


# 3. 定义请求 DTO（限制前端传参格式）
class ChatRequest(BaseModel):
    question: str  # 用户的大白话提问


# 4. 定义返回 DTO（规范后端响应格式）
class ChatResponse(BaseModel):
    question: str  # 用户的原始提问
    context: str  # 图谱精准捞出来的上下文（供对账、前端高亮或 Debug 审查）
    answer: str  # 大模型总结出来的最终大白话答案


# 5. 编写核心 HTTP POST 业务接口


@app.post("/api/chat", response_model=ChatResponse)
async def graph_rag_chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")

    try:
        # A. 顺藤摸瓜
        graph_context = searcher.local_search(request.question, top_k=3)

        # B. 融会贯通
        final_answer = llm_service.generate_answer(request.question, graph_context)

        # 🛠️ 【后端清洗机制】强行把 3 个及以上的连续换行符，全部压缩成 1 个或 2 个
        clean_context = re.sub(r'\n{3,}', '\n\n', graph_context.strip())
        clean_answer = re.sub(r'\n{3,}', '\n\n', final_answer.strip())

        # C. 规范交卷
        return ChatResponse(
            question=request.question.strip(),
            context=clean_context,
            answer=clean_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部对账故障: {str(e)}")

# 6. 优雅生命周期管理：当 Web 容器关闭时，断开图数据库长连接
@app.on_event("shutdown")
def shutdown_event():
    searcher.close()
    print("🔌 已安全断开 Neo4j 图数据库连接池。")


# 7. 生产级标准启动主函数
if __name__ == "__main__":
    # 监听本地 8080 端口，配置 reload=True 允许热重载（改代码自动生效）
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)