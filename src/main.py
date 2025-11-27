from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API",
        "description": "API para detecção de fraudes usando modelos de ML.",
        "endpoints": {
            "health": "/health"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)