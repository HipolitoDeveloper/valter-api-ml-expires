from fastapi import FastAPI, Request
from src.api import routes
from fastapi import APIRouter

app = FastAPI(title="Valter - Expiry/Out-of-Stock", root_path="/api")

@app.get("/")
def root():
    return {"ok": True, "service": "valter-ml-expires"}

app.include_router(routes.router)
