import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re  # 引入正则模块，用于对 LLM 返回的文本进行清洗脱水

# 💡 显式导入封装好的微服务组件（遵循 Python 物理目录即包路径的哲学，文件即模块）
from search_service import GraphRAGLocalSearcher
from llm_service import LLMService

# ==========================================
# 1. 初始化 FastAPI Web 容器容器
# ==========================================
# 类似于 Spring Boot 中配置 Swagger UI 与 WebFlux 核心容器
app = FastAPI(
    title="星河科技集团 - 工业级 GraphRAG 线上问答系统",
    description="基于 FastAPI + Neo4j 向量图谱 + DeepSeek-V3 的生产级 RAG 闭环"
)

# ==========================================
# 2. 全局单例实例化（避免内存震荡与频繁建连）
# ==========================================
# 核心心法：在全局作用域初始化，确保只在 Web 服务刚启动时触发一次构造函数（__init__）
# 避免在每个 HTTP 接口请求进来时重复创建 Neo4j Driver 连接池，压榨物理机性能
searcher = GraphRAGLocalSearcher()
llm_service = LLMService()


# ==========================================
# 3. 定义请求 DTO（限制前端传参格式）
# ==========================================
# 基于 Pydantic 实现强类型校验，如果前端传入的 JSON 格式不对，FastAPI 会自动拦截并吐回 422 报错
class ChatRequest(BaseModel):
    question: str  # 用户输入的大白话提问文本


# ==========================================
# 4. 定义返回 DTO（规范后端响应格式）
# ==========================================
# 规范出参结构，便于前端（或 Apifox 联调）建立标准的对账账单
class ChatResponse(BaseModel):
    question: str  # 用户的原始提问（剔除前后空字符串）
    context: str   # 图数据库经过“动态切线剪枝”后，精准捞出来的白话文上下文事实
    answer: str    # 低温度限制下，大模型融会贯通、无幻觉提炼出来的最终大白话答案


# ==========================================
# 5. 编写核心 HTTP POST 业务接口
# ==========================================
# 采用标准 POST 方法，response_model 绑定返回 DTO，实现自动化的出参序列化与脱敏
@app.post("/api/chat", response_model=ChatResponse)
async def graph_rag_chat(request: ChatRequest):
    """
    核心业务流程（三步多路召回与总结闭环）：
    用户提问 -> 问题向量化 -> 局部图路径动态裁剪 -> 组装注入大模型 -> 洗数脱水返回
    """
    # 🚫 卡口一：前端前置防空校验，防止空串或全是空格的无效请求轰炸接口
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")

    try:
        # A. 顺藤摸瓜：拿着问题去 Neo4j 执行经典存储过程与 1-2 步拓扑深度打捞
        # 此时如果是在白天，会通过后端动态切线机制，在查询瞬间把黑名单绑定的死连线全部斩断
        graph_context = searcher.local_search(request.question, top_k=3)

        # B. 融会贯通：把打捞出来的干净图谱事实作为依据，送进低温度（temperature=0.3）的 DeepSeek-V3
        # 强行束缚住大模型的思维逻辑，严防幻觉，输出有血有肉的答案
        final_answer = llm_service.generate_answer(request.question, graph_context)

        # 🛠️ 关键清洗脱水层：防止 Markdown 列表语法的天然换行在 JSON 转义时被无限放大
        # 利用正则机制，强行把 3 个及以上的连续换行符（\n\n\n...），全部收拢压缩成标准的 2 个换行（\n\n）
        clean_context = re.sub(r'\n{3,}', '\n\n', graph_context.strip())
        clean_answer = re.sub(r'\n{3,}', '\n\n', final_answer.strip())

        # C. 规范交卷：打包成结构化 JSON 账单并返回给调用方
        return ChatResponse(
            question=request.question.strip(),
            context=clean_context,
            answer=clean_answer
        )
    except Exception as e:
        # 🛡️ 兜底防护：一键捕获整个链条中的意外故障，统一抛出标准 500，防止崩溃日志直接暴漏给前端
        raise HTTPException(status_code=500, detail=f"服务器内部对账故障: {str(e)}")


# ==========================================
# 6. 优雅生命周期管理（断开长连接）
# ==========================================
# 相当于 Spring 容器关闭时的销毁回调（@PreDestroy）
@app.on_event("shutdown")
def shutdown_event():
    # 当 Web 容器（Uvicorn）收到关闭信号时，优雅断开 Neo4j 图数据库的 Bolt 连接池，释放物理套接字
    searcher.close()
    print("🔌 已安全断开 Neo4j 图数据库连接池。")


# ==========================================
# 7. 生产级标准常驻内存启动主函数
# ==========================================
# 类似于 Spring Boot 应用的入口 main 方法
if __name__ == "__main__":
    # 1. 端口（port）已经切到冷门的 8099，完美避开本地 Java 的 Jetty 撞车事故
    # 2. ⚠️ 注意：如果要配合 PyCharm / IDEA 的绿色小甲虫（Debug）打红点加断点调试，
    #    强烈建议在本地临时将 reload 改为 False（关闭热重载）。
    #    因为 reload=True 会导致 Uvicorn 偷偷 Fork 出一个独立子进程来跑 Web 服务，导致 IDE 无法绑定 Debugger，断点会死活进不去！
    uvicorn.run("main:app", host="127.0.0.1", port=8099, reload=True)