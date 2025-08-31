from fastapi import FastAPI
from src.api import routes


app = FastAPI(title="Valter - Expiry/Out-of-Stock")
app.include_router(routes.router)