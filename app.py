# 导入必要的模块
from fastapi import FastAPI, HTTPException, Request, Body, File, UploadFile, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from gotrue.errors import AuthApiError

# --- 配置 ---

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量读取API密钥和Supabase配置
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # 使用Service Key

if not all([MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("环境变量 MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY 必须全部设置！")

# --- 初始化客户端 ---

# 初始化 FastAPI 应用
app = FastAPI(title="AI教育单智能体系统 (Supabase版)", description="基于LangChain和Supabase向量数据库的智能体")

# 配置CORS
origins = [
    "https://mvp-frontend-ln39.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化一个拥有服务权限的Supabase客户端
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
logger.info("Supabase 管理员客户端初始化成功")

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

# --- 认证依赖 ---

auth_scheme = HTTPBearer()

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """验证JWT并返回用户信息"""
    try:
        user_response = supabase_admin.auth.get_user(token.credentials)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="无效的用户凭证")
        return user
    except AuthApiError as e:
        logger.error(f"JWT 验证失败: {e}")
        raise HTTPException(status_code=401, detail=f"认证失败: {e}")
    except Exception as e:
        logger.error(f"处理认证时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

# --- 数据模型 ---

class LessonPlanRequest(BaseModel):
    grade: str
    module: str
    knowledge_point: str
    duration: int
    preferences: List[str]
    custom_requirements: Optional[str] = None
    use_rag: bool = True

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
        query_embedding = embeddings_model.embed_query(query_text)
        res = supabase_admin.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold': 0.7,
            'match_count': top_k
        }).execute()

        if not res.data:
            logger.info("在数据库中未找到相关文档")
            return "无"

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
async def generate_lesson_plan(request: LessonPlanRequest, user: dict = Depends(get_current_user)):
    """生成教案并为用户保存"""
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

        lesson_plan_data = None
        try:
            lesson_plan_data = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if match:
                lesson_plan_data = json.loads(match.group(1))
            else:
                logger.error(f"无法解析LLM返回的JSON: {response_text}")
                raise HTTPException(status_code=500, detail="无法解析生成的教案数据")
        
        # 将生成的教案存入数据库，并与用户关联
        insert_res = supabase_admin.table('lesson_plans').insert({
            'user_id': user.id,
            'title': lesson_plan_data.get('title', '未命名教案'),
            'grade': request.grade,
            'module': request.module,
            'knowledge_point': request.knowledge_point,
            'duration': request.duration,
            'preferences': request.preferences,
            'custom_requirements': request.custom_requirements,
            'content': lesson_plan_data # 存储完整的JSON内容
        }).execute()

        if not insert_res.data:
            raise HTTPException(status_code=500, detail="保存教案到数据库失败")

        return lesson_plan_data

    except Exception as e:
        logger.error(f"生成教案时出错: {e}")
        raise HTTPException(status_code=500, detail=f"生成教案失败: {e}")


@app.post("/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest, user: dict = Depends(get_current_user)):
    """回答教学问题"""
    try:
        context = request.context or ""
        if request.use_rag:
            retrieved_context = get_relevant_documents_from_db(request.question)
            context = f"{context}\n\n检索到的相关信息:\n{retrieved_context}"

        answer = qa_chain.run(question=request.question, context=context)
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"回答问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"回答问题失败: {e}")


@app.post("/knowledge/upload", status_code=201)
async def upload_knowledge(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """上传知识库文件，处理并存入Supabase数据库"""
    if not user:
        raise HTTPException(status_code=403, detail="只有登录用户才能上传知识库")
        
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式，目前仅支持PDF和TXT")
        
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        documents_to_insert = []
        for text in texts:
            embedding = embeddings_model.embed_query(text.page_content)
            documents_to_insert.append({
                'content': text.page_content,
                'metadata': {'source': file.filename, 'uploader_id': user.id},
                'embedding': embedding
            })
        
        supabase_admin.table('documents').insert(documents_to_insert).execute()
        shutil.rmtree(temp_dir)

        return {
            "message": f"文件 {file.filename} 已成功处理并存入知识库",
            "chunks_added": len(documents_to_insert)
        }
    except Exception as e:
        logger.error(f"上传知识库文件时出错: {e}")
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