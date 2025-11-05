# V1.0 M2.1: 修正 pgvector 環境錯誤
from fastapi import FastAPI
import uvicorn
import os
import logging # 引入日誌工具
from sqlalchemy import create_engine, text

# --- R5 心法：讀取環境變數 ---
# 讀取 R5 教授要我們找的 DATABASE_URL
DATABASE_URL = os.environ.get("DATABASE_URL")

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- R6 實踐：建立資料庫連線引擎 ---
# (我們只在 DATABASE_URL 存在時才嘗試連線)
engine = None
if DATABASE_URL:
    logger.info("偵測到 DATABASE_URL，正在嘗試連線...")
    try:
        # 建立 SQLAlchemy 引擎
        # (注意：Railway 提供的 URL 可能是 postgresql://... 我們要換成 postgresql+psycopg2://)
        db_url_psycopg2 = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
        engine = create_engine(db_url_psycopg2)

        # --- R5 教授的 SQL 指令 ---
        with engine.connect() as connection:
            logger.info("資料庫連線成功！開始初始化資料表...")

            # 1. 安裝 pgvector 擴充套件
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # 2. 建立 RAG 知識庫表格
            # (R5 心法：VARCHAR(1536) 假設我們用 OpenAI 的 1536 維向量)
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR(1536) 
            )
            """))

            # 提交事務
            connection.commit()
            logger.info("資料表 'knowledge_chunks' 已成功驗證/建立。")

    except Exception as e:
        logger.error(f"資料庫初始化失敗: {e}")
        engine = None
else:
    logger.warning("未找到 DATABASE_URL，跳過資料庫初始化。")


# --- FastAPI 應用程式 ---
app = FastAPI()

@app.get("/")
def read_root():
    if engine:
        return {"message": "V1.0 CI/CD 驗證成功 (RAG-MVP)，且已成功連線並初始化資料庫！"}
    else:
        return {"message": "V1.0 CI/CD 驗證成功 (RAG-MVP)，但資料庫連線失敗 (請檢查日誌)。"}

# ... (uvicorn 啟動部分保持不變) ...
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)