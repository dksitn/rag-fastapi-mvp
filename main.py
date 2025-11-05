from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    # 這是我們的 "Hello World"
    return {"message": "V1.0 CI/CD 驗證成功 (RAG-MVP)"}

if __name__ == "__main__":
    # 這是關鍵：Railway 會透過 $PORT 環境變數告訴我們要在哪個埠口啟動
    port = int(os.environ.get("PORT", 8080))
    # 監聽 0.0.0.0 才能讓 Railway 存取到
    uvicorn.run(app, host="0.0.0.0", port=port)