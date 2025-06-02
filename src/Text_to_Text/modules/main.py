import uvicorn
from .api_routes import app

if __name__ == "__main__":
    print("🚀 AI Assistant running with FastAPI...")
    uvicorn.run(app, host="127.0.0.1", port=8989, log_level="info") 