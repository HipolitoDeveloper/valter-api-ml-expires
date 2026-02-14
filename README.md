# Valter ML — Expiry/Out-of-Stock Prediction Service

Serviço de Machine Learning que prediz a probabilidade de um produto estar **esgotado ou vencido** na despensa do usuário. Roda como **AWS Lambda** com deploy automatizado via **GitHub Actions**.

## Arquitetura

```
Valter API (NestJS Lambda)
        │
        │ POST /predict  { "user_id": "uuid" }
        ▼
   API Gateway HTTP API v2
        │
        ▼
   Lambda predict (Python 3.12)
        │
        ├─ Lê transações e despensa ──► PostgreSQL (Neon/Supabase)
        ├─ Carrega modelo treinado  ──► S3 (ou /tmp)
        └─ Retorna probabilidades

   EventBridge (segunda 6:00 UTC)
        │
        ▼
   Lambda train (Python 3.12)
        │
        ├─ Lê dados históricos ──► PostgreSQL
        └─ Salva modelo        ──► S3
```

## Funções Lambda

### `handle_predict` — POST /predict

Recebe um `user_id` e retorna a probabilidade de cada produto ativo estar esgotado ou vencido.

**Trigger:** API Gateway HTTP API v2

**Request:**
```json
{ "user_id": "8acbd1d0-5060-4a74-9fac-6ad241a2d598" }
```

**Response:**
```json
{
  "user_id": "8acbd1d0-5060-4a74-9fac-6ad241a2d598",
  "items": [
    {
      "product_id": "uuid",
      "probability_out_or_expired": 0.85,
      "days_since_purchase": 12.5,
      "last_notification_at": "2025-01-15T00:00:00+00:00"
    }
  ]
}
```

**Pipeline:**
1. Busca transações e dados de despensa do PostgreSQL
2. Constrói ciclos de compra por produto (`features/cycles.py`)
3. Estima taxa de consumo diário (`features/consumption.py`)
4. Calcula 7 features de inferência (`features/dataset.py`)
5. Aplica o modelo (LogisticRegression) e retorna probabilidades

### `handle_train` — CRON semanal

Retreina o modelo ML com dados atualizados do banco.

**Trigger:** EventBridge — toda segunda-feira às 6:00 UTC

**Pipeline:**
1. Busca transações, despensa e feedback humano do PostgreSQL
2. Gera dataset heurístico (ciclos rotulados automaticamente)
3. Gera dataset supervisionado (feedback do usuário, peso 2x)
4. Treina `StandardScaler + LogisticRegression` (balanced weights)
5. Salva modelo no S3

## Estrutura do Projeto

```
src/
├── handler.py              # Entry points Lambda (handle_predict, handle_train)
├── config.py               # Variáveis de ambiente (Pydantic Settings)
├── db/
│   ├── connection.py       # SQLAlchemy engine factory
│   └── queries.py          # Queries SQL (transações, despensa, feedback)
├── features/
│   ├── cycles.py           # Constrói ciclos de compra→reabastecimento
│   ├── consumption.py      # Estima consumo diário por usuário/produto
│   └── dataset.py          # Feature engineering (treino e inferência)
├── model/
│   ├── estimator.py        # Classificador (LogisticRegression pipeline)
│   └── storage.py          # Save/load modelo (local + S3)
├── pipeline/
│   ├── predict.py          # Pipeline de inferência (predict_for_user)
│   └── train.py            # Pipeline de treinamento (run_training)
└── schemas/
    ├── enums.py            # Estados de transação (IN_CART, PURCHASED, OUT, etc.)
    └── prediction.py       # Schemas Pydantic (request/response)
```

## Features do Modelo (7)

| Feature | Descrição |
|---------|-----------|
| `quantity_bought` | Quantidade comprada no ciclo |
| `valid_for_days` | Validade do produto (dias) |
| `has_validity_info` | 1.0 se validade conhecida, 0.0 caso contrário |
| `consumption_uday` | Taxa de consumo diário estimada |
| `expected_depletion_days` | Dias esperados até acabar (qty / consumo) |
| `ratio_to_validity` | days_since_purchase / valid_for_days |
| `ratio_to_depletion` | days_since_purchase / expected_depletion_days |

## Variáveis de Ambiente

| Variável | Obrigatória | Default | Descrição |
|----------|-------------|---------|-----------|
| `DATABASE_URL` | Sim | — | Connection string PostgreSQL |
| `AWS_BUCKET_NAME` | Não | `""` | Bucket S3 para modelo (vazio = filesystem local) |
| `AWS_REGION` | Não | `us-east-1` | Região AWS para deploy |
| `MODEL_NAME` | Não | `ml_expires.joblib` | Nome do arquivo do modelo |
| `MODEL_DIR` | Não | `models` | Diretório local para modelo (Lambda usa `/tmp`) |
| `HISTORY_DAYS` | Não | `720` | Dias de histórico de transações |
| `FEEDBACK_DAYS` | Não | `365` | Dias de histórico de feedback |

## Deploy

### GitHub Secrets necessários

| Secret | Descrição |
|--------|-----------|
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | Região AWS (default: us-east-1) |
| `DATABASE_URL` | PostgreSQL connection string |
| `AWS_BUCKET_NAME` | Nome do bucket S3 para modelo |

### Deploy automático

Push para `master` → GitHub Actions roda → `serverless deploy --stage prod`

### Deploy manual

```bash
npm install -g serverless@3
serverless deploy --stage dev --verbose
```

### Invocar manualmente

```bash
# Testar predict
serverless invoke -f predict --stage prod --data '{"body": "{\"user_id\": \"uuid\"}"}'

# Forçar treino
serverless invoke -f train --stage prod
```

## Dev Local

```bash
pip install -r requirements.txt

# Criar .env
cat > .env << EOF
DATABASE_URL=postgresql://user:pass@localhost:5432/core
MODEL_DIR=models
MODEL_NAME=ml_expires.joblib
EOF

# Testar predict
python -c "
from src.pipeline.predict import predict_for_user
print(predict_for_user('USER-UUID'))
"
```
