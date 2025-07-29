# 导入必要的模块
from fastapi import FastAPI, HTTPException, Request, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import time
import shutil
import logging
from datetime import datetime

# LangChain 和 Supabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from supabase.client import Client, create_client

# --- 配置 ---

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量读取API密钥和Supabase配置
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # 使用Service Key，因为它有权写入数据库

if not all([MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("环境变量 MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY 必须全部设置！")

# --- 初始化客户端 ---

# 初始化 FastAPI 应用
app = FastAPI(title="AI教育单智能体系统 (Supabase版)", description="基于LangChain和Supabase向量数据库的智能体")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Supabase 客户端
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase 客户端初始化成功")

# 初始化 LangChain 组件
# 注意：流式处理在Serverless环境中可能需要特殊配置，这里暂时简化
llm = ChatOpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    model="moonshot-v1-8k", # 使用推荐的模型名称
    temperature=0.7,
    streaming=False # 在Serverless中，直接返回完整响应更简单可靠
)

embeddings_model = OpenAIEmbeddings(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    model="moonshot-v1-embedding" # 使用Kimi的嵌入模型
)

# --- 数据模型 ---

class LessonPlanRequest(BaseModel):
    grade: str
    module: str
    knowledge_point: str
    duration: int
    preferences: List[str]
    custom_requirements: Optional[str] = None
    use_rag: bool = True
    user_id: Optional[str] = None

# 注意：响应模型现在可以简化，因为前端会直接和Supabase交互来获取和管理教案
class GeneratedLessonPlan(BaseModel):
    title: str
    objectives: List[Dict[str, str]]
    key_points: List[str]
    difficult_points: List[str]
    resources: List[Dict[str, str]]
    teaching_process: List[Dict[str, Any]]
    evaluation: str
    extension: str

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    use_rag: bool = True
    user_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    references: Optional[List[Dict[str, str]]] = None

# --- LangChain 核心逻辑 ---

lesson_plan_template = """
你是一位专业的人工智能教育专家，负责为教师生成高质量的教案。请根据以下信息生成一份详细的教案：
学段与年级: {grade}
课程模块: {module}
核心知识点: {knowledge_point}
课时: {duration}课时（每课时40分钟）
教学偏好: {preferences}
自定义要求: {custom_requirements}
请确保教案符合相关的教育框架和要求。
教案应包含以下部分：
1. 教学目标
2. 教学重难点
3. 教学资源建议
4. 教学过程设计
5. 教学评价
6. 拓展建议
请以JSON格式输出，包含以下字段：
- title: 教案标题
- objectives: 教学目标列表
- key_points: 教学重点列表
- difficult_points: 教学难点列表
- resources: 教学资源列表
- teaching_process: 教学过程列表
- evaluation: 教学评价
- extension: 拓展建议
相关参考资料:
{context}
"""
lesson_plan_prompt = PromptTemplate.from_template(lesson_plan_template)
lesson_plan_chain = LLMChain(llm=llm, prompt=lesson_plan_prompt)

qa_template = """
你是一位专业的人工智能教育专家，负责回答教师关于人工智能教学的问题。
请根据以下信息回答问题：
问题: {question}
背景信息: {context}
请确保你的回答专业、具体、清晰易懂。
"""
qa_prompt = PromptTemplate.from_template(qa_template)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


# --- 辅助函数 ---

def get_relevant_documents_from_db(query_text: str, top_k: int = 3) -> str:
    """从Supabase数据库中检索与查询相关的文档"""
    try:
        # 1. 为查询文本生成向量
        query_embedding = embeddings_model.embed_query(query_text)

        # 2. 调用数据库函数进行相似度搜索
        res = supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold': 0.7,  # 相似度阈值，可根据效果调整
            'match_count': top_k
        }).execute()

        if not res.data:
            logger.info("在数据库中未找到相关文档")
            return "无"

        # 3. 格式化检索到的内容作为上下文
        context = "\n".join([doc['content'] for doc in res.data])
        logger.info(f"已从数据库检索到 {len(res.data)} 条相关内容")
        return context
    except Exception as e:
        logger.error(f"从数据库检索文档时出错: {e}")
        return "检索知识库时发生错误"


# --- API 路由 ---

@app.get("/")
async def root():
    return {"message": "AI教育单智能体系统API (Supabase版)"}

@app.post("/generate-lesson-plan", response_model=GeneratedLessonPlan)
async def generate_lesson_plan(request: LessonPlanRequest):
    """生成教案"""
    try:
        context = ""
        if request.use_rag:
            query = f"{request.grade} {request.module} {request.knowledge_point}"
            context = get_relevant_documents_from_db(query)

        preferences_str = ", ".join(request.preferences) if request.preferences else "无特殊偏好"
        custom_requirements_str = request.custom_requirements if request.custom_requirements else "无特殊要求"

        response_text = lesson_plan_chain.run(
            grade=request.grade,
            module=request.module,
            knowledge_point=request.knowledge_point,
            duration=request.duration,
            preferences=preferences_str,
            custom_requirements=custom_requirements_str,
            context=context
        )

        try:
            # 尝试直接解析JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果失败，尝试从Markdown代码块中提取
            import re
            match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            logger.error(f"无法解析LLM返回的JSON: {response_text}")
            raise HTTPException(status_code=500, detail="无法解析生成的教案数据")

    except Exception as e:
        logger.error(f"生成教案时出错: {e}")
        raise HTTPException(status_code=500, detail=f"生成教案失败: {e}")


@app.post("/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """回答教学问题"""
    try:
        context = request.context or ""
        if request.use_rag:
            retrieved_context = get_relevant_documents_from_db(request.question)
            context = f"{context}\n\n检索到的相关信息:\n{retrieved_context}"

        answer = qa_chain.run(question=request.question, context=context)
        
        # 简化响应，只返回核心答案
        return {"answer": answer}
    except Exception as e:
        logger.error(f"回答问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"回答问题失败: {e}")


@app.post("/knowledge/upload", status_code=201)
async def upload_knowledge(file: UploadFile = File(...)):
    """上传知识库文件，处理并存入Supabase数据库"""
    try:
        # 1. 将上传的文件保存到临时位置
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 根据文件类型加载文档
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式，目前仅支持PDF和TXT")
        
        documents = loader.load()

        # 3. 分割文本
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info(f"文件 {file.filename} 被分割成 {len(texts)} 个文本块")

        # 4. 为每个文本块生成向量并准备数据
        documents_to_insert = []
        for text in texts:
            embedding = embeddings_model.embed_query(text.page_content)
            documents_to_insert.append({
                'content': text.page_content,
                'metadata': {'source': file.filename},
                'embedding': embedding
            })
        
        # 5. 将数据批量插入到Supabase数据库
        res = supabase.table('documents').insert(documents_to_insert).execute()
        
        # 6. 清理临时文件
        shutil.rmtree(temp_dir)

        return {
            "message": f"文件 {file.filename} 已成功处理并存入知识库",
            "chunks_added": len(documents_to_insert)
        }
    except Exception as e:
        logger.error(f"上传知识库文件时出错: {e}")
        # 清理可能的临时文件
        if os.path.exists("temp_uploads"):
            shutil.rmtree("temp_uploads")
        raise HTTPException(status_code=500, detail=f"上传知识库文件失败: {e}")

# 用于本地调试
if __name__ == "__main__":
    import uvicorn
    # 确保在本地运行时，能加载.env文件
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8001)