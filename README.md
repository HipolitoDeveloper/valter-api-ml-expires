# Valter - ML2 Expiry/Out-of-Stock

Serviço FastAPI que treina e prediz probabilidade de um item estar **esgotado ou vencido**.

## Rodando

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8082
```

Crie um `.env`:

```
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/valter
MODEL_DIR=models
MODEL_NAME=ml2_expiry.joblib
```

Treino:
```
POST /train
```

Predição para um usuário:
```
POST /predict
{ "user_id": "USER-UUID" }
```
