# api/index.py
import os
from fastapi import FastAPI

app = FastAPI(title="Valter - Expiry/Out-of-Stock (sanity)")

@app.get("/health")
def health():
    return {
        "ok": True,
        "python": os.getenv("PYTHON_VERSION", "unknown"),
    }

# Tente importar suas rotas *depois* de validar que a função sobe
try:
    from src.api import routes  # precisa de __init__.py nos dirs
    app.include_router(routes.router)
except Exception as e:
    # Ajuda a ver o erro nos logs sem derrubar o processo
    import sys, traceback
    print("ROUTES_IMPORT_ERROR:", e, file=sys.stderr)
    traceback.print_exc()
