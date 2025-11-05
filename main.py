# V1.0 M3: 引入 sentence-transformers 並建立 /add-knowledge 端點

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # 用於 API 的資料驗證
import uvicorn
import os
import logging
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer # <-- [模組 3] 新增

# --- R5 心法：讀取環境變數 ---
DATABASE_URL = os.environ.get("DATABASE_URL")

# --- [模組 3] R6 實踐：載入 Embedding 模型 ---
# R4 心法：在全域載入模型 (啟動時載入一次)，而不是在 API 呼叫時才載入。
# "all-MiniLM-L6-v2" 是一個輕量、快速、高品質的通用模型。
try:
    logger.info("正在載入 SentenceTransformer 模型...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # 我們模型的向量維度是 384。(注意：這與 R5 昨天的 1536 不同，我們稍後修正)
    EMBEDDING_DIM = 384
    logger.info("SentenceTransformer 模型載入成功。")
except Exception as e:
    logger.error(f"模型載入失敗: {e}")
    embedding_model = None

# --- R6 實踐：建立資料庫連線引擎 ---
engine = None
if DATABASE_URL:
    logger.info("偵測到 DATABASE_URL，正在嘗試連線...")
    try:
        db_url_psycopg2 = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
        engine = create_engine(db_url_psycopg2)

        with engine.connect() as connection:
            logger.info("資料庫連線成功！開始初始化資料表...")

            # 2. [模組 3] R5/R6 修正：
            # R5 心法：我們必須確保 DB 欄位 (VECTOR) 的維度
            # 與 R6 載入的模型 (EMBEDDING_DIM) 一致！
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
        engine = None # 保持 R7 的偵錯心法
else:
    logger.warning("未找到 DATABASE_URL，跳過資料庫初始化。")

# --- FastAPI 應用程式 ---
app = FastAPI()

# R6 實踐：這是一個 Pydantic 模型，用於驗證 API 請求的 body
class KnowledgeItem(BaseModel):
    content: str

@app.get("/")
def read_root():
    if engine and embedding_model:
        return {"message": "V1.0 M3: CI/CD 成功，資料庫連線成功，Embedding 模型載入成功！"}
    elif not engine:
        return {"message": "V1.0 M3: CI/CD 成功，但資料庫連線失敗 (請檢查日誌)。"}
    else:
        return {"message": "V1.0 M3: CI/CD 成功，資料庫連線成功，但 Embedding 模型載入失敗！"}

# --- [模組 3] R6 核心任務：建立 "寫入" API ---
@app.post("/add-knowledge")
def add_knowledge(item: KnowledgeItem):
    if not engine or not embedding_model:
        raise HTTPException(status_code=503, detail="資料庫或 AI 模型尚未準備就緒")

    try:
        # 1. R6 核心：將文字轉換為向量
        logger.info(f"正在為內容編碼: {item.content[:20]}...")
        vector = embedding_model.encode(item.content).tolist() # .tolist() 轉換為 Python 列表

        # 2. R5 核心：將資料寫入資料庫
        with engine.connect() as connection:
            # R5 心法：使用 "參數化查詢" (params) 來防止 SQL 注入
            connection.execute(
                text("INSERT INTO knowledge_chunks (content, embedding) VALUES (:content, :embedding)"),
                params={
                    "content": item.content,
                    "embedding": str(vector) # pgvector 接受向量的字串格式
                }
            )
            connection.commit()

        logger.info("新知識寫入成功。")
        return {"message": "知識新增成功", "content": item.content}

    except Exception as e:
        logger.error(f"寫入知識時失敗: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {e}")

# --- (uvicorn 啟動部分保持不變) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)