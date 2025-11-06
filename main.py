# V1.0 M6: 最終版 - RAG (檢索-增強-生成) 完整流程

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import logging
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import openai # <-- [模組 6] R4/R6 核心：引入 OpenAI

# --- 設定日誌 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- R2 安全心法：讀取環境變數 ---
DATABASE_URL = os.environ.get("DATABASE_URL")
# [模組 6] R2 核心：安全地讀取 OpenAI 金鑰
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# --- [模組 6] R6 實踐：初始化 OpenAI 客戶端 ---
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API 金鑰載入成功。")
else:
    logger.warning("未找到 OPENAI_API_KEY，G (生成) 步驟將會失敗。")

# --- [模組 3] R6 實踐：載入 Embedding 模型 ---
embedding_model = None
EMBEDDING_DIM = 384
try:
    logger.info("正在載入 SentenceTransformer 模型 ('all-MiniLM-L6-v2')...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer 模型載入成功。")
except Exception as e:
    logger.error(f"模型載入失敗: {e}")

# --- [模組 2] R6 實踐：建立資料庫連線引擎 ---
engine = None
if DATABASE_URL:
    logger.info("偵測到 DATABASE_URL，正在嘗試連線...")
    try:
        db_url_psycopg2 = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
        engine = create_engine(db_url_psycopg2)

        with engine.connect() as connection:
            logger.info("資料庫連線成功！開始初始化資料表...")
            connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR({EMBEDDING_DIM}) 
            )
            """))
            connection.commit()
            logger.info(f"資料表 'knowledge_chunks' 已成功驗證/建立 (維度: {EMBEDDING_DIM})。")

    except Exception as e:
        logger.error(f"資料庫初始化失敗: {e}")
        engine = None
else:
    logger.warning("未找到 DATABASE_URL，跳過資料庫初始化。")

# --- FastAPI 應用程式 ---
app = FastAPI()

class KnowledgeItem(BaseModel):
    content: str

@app.get("/")
def read_root():
    if engine and embedding_model and OPENAI_API_KEY:
        return {"message": "V1.0 M6 (最終版): CI/CD 成功，DB、Embedding、OpenAI 金鑰皆已就緒！"}
    else:
        return {"message": "V1.0 M6 (最終版): 警告！有服務未就緒 (請檢查日誌)。"}

# --- [模組 4] R6 核心任務：建立 "S (儲存)" API ---
@app.post("/add-knowledge")
def add_knowledge(item: KnowledgeItem):
    if not engine or not embedding_model:
        raise HTTPException(status_code=503, detail="資料庫或 AI 模型尚未準備就緒")

    try:
        vector = embedding_model.encode(item.content).tolist()
        with engine.connect() as connection:
            connection.execute(
                text("INSERT INTO knowledge_chunks (content, embedding) VALUES (:content, :embedding::VECTOR)"),
                params={"content": item.content, "embedding": str(vector)}
            )
            connection.commit()
        return {"message": "知識新增成功", "content": item.content}
    except Exception as e:
        logger.error(f"寫入知識時失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {e}")

# --- [模組 6] R6 核心任務：建立 "RAG" API (升級 /query) ---
@app.get("/query")
def query_knowledge(question: str):
    if not engine or not embedding_model or not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="一個或多個服務尚未準備就緒 (DB, Embedding, or OpenAI)")

    if not question:
        raise HTTPException(status_code=400, detail="請提供 'question' 參數")

    try:
        # --- 1. R (檢索 - Retrieval) 步驟 ---
        # (R6 承襲 M5)
        logger.info(f"步驟 1 [R]: 正在為問題編碼: {question[:20]}...")
        query_vector = embedding_model.encode(question).tolist()

        logger.info("步驟 1 [R]: 正在資料庫中執行向量搜尋...")
        retrieved_chunks = []
        with engine.connect() as connection:
            sql_query = text(f"""
            SELECT content, embedding <=> :query_vector AS distance
            FROM knowledge_chunks
            ORDER BY distance
            LIMIT 3
            """)
            result_proxy = connection.execute(
                sql_query,
                params={"query_vector": str(query_vector)}
            )
            retrieved_chunks = [row._asdict() for row in result_proxy]

        logger.info(f"步驟 1 [R]: 檢索到 {len(retrieved_chunks)} 筆相關知識。")

        # --- 2. A (增強 - Augmentation) 步驟 ---
        # R6 心法：這就是「提示工程 (Prompt Engineering)」！
        logger.info("步驟 2 [A]: 正在建構傳送給 LLM 的「提示詞 (Prompt)」...")

        # (R6 實踐：將檢索到的知識組合成一個「上下文」)
        context_string = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])

        # R6 提示詞模板 (RAG 核心)
        prompt_template = f"""
        你 (AI) 是一個專業的「文件問答助理」。
        請你 (AI) **只**根據下方提供的「上下文 (Context)」來回答「使用者的問題 (Question)」。
        如果「上下文」中沒有足夠的資訊，請你 (AI) 回答：「根據我 (AI) 所擁有的資料，我 (AI) 無法回答這個問題。」

        ---
        [上下文 (Context)]
        {context_string}
        ---
        [使用者的問題 (Question)]
        {question}
        ---

        [你 (AI) 的答案]
        """

        logger.info("步驟 2 [A]: 提示詞建構完畢。")

        # --- 3. G (生成 - Generation) 步驟 ---
        logger.info("步驟 3 [G]: 正在呼叫 OpenAI (gpt-3.5-turbo) API...")

        # (R6 實踐：使用 openai v1.x+ 的 "ChatCompletion" API)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_template}
            ]
        )

        generated_answer = chat_completion.choices[0].message.content
        logger.info("步驟 3 [G]: OpenAI API 回應成功。")

        return {
            "message": "RAG 檢索成功", 
            "generated_answer": generated_answer, # <-- R6: 這是 LLM (GPT-3.5) 生成的答案
            "retrieved_context": retrieved_chunks # <-- R6: 這是 R (檢索) 到的原文 (用於偵錯)
        }

    except Exception as e:
        logger.error(f"RAG 流程執行時失敗: {e}")
        # (R7 偵錯心法：將詳細錯誤印在日誌)
        logger.error(f"詳細錯誤 (Traceback):", exc_info=True)
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {e}")

# --- (uvicorn 啟動部分保持不變) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)