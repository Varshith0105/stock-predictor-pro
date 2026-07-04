# Backend API Spec (FastAPI target)

Base URL: `${VITE_API_BASE_URL}` (e.g. `https://api.yourdomain.com`).
All request/response bodies are JSON. Auth-protected endpoints require
`Authorization: Bearer <access_token>`.

## Stocks

### POST /api/stocks/history
Replaces the `fetch-stock-data` edge function.

Request:
```json
{ "symbol": "AAPL", "startDate": "2024-01-01", "endDate": "2024-02-01" }
```
Response `200`:
```json
{
  "data": [
    { "date": "2024-01-02", "open": 187.1, "high": 188.4, "low": 183.9, "close": 185.6, "volume": 52000000 }
  ],
  "isDemo": false,
  "message": null,
  "rateLimited": false
}
```
Errors: `400` invalid symbol/date, `429` rate limited (returns synthetic data + `rateLimited: true` for compatibility), `500` upstream failure.

## Auth

### POST /api/auth/register
Body: `{ "email": string, "password": string, "displayName"?: string }`
Response: `{ "accessToken": string, "refreshToken": string, "user": User }`

### POST /api/auth/login
Body: `{ "email": string, "password": string }`
Response: same as register.

### POST /api/auth/refresh
Body: `{ "refreshToken": string }` (or httpOnly cookie)
Response: `{ "accessToken": string, "refreshToken": string }`

### POST /api/auth/logout
Auth required. Revokes current refresh token. Response: `204`.

### GET /api/auth/me
Auth required. Response: `User`.

`User` = `{ id: uuid, email: string, displayName: string | null, createdAt: iso8601 }`.

## Predictions (optional persistence layer)

### POST /api/predictions
Auth required. Body:
```json
{ "symbol": "AAPL", "model": "lstm", "horizonDays": 7, "input": [...], "output": [...] }
```
Response `201`: `{ "id": uuid, "createdAt": iso8601 }`.

### GET /api/predictions?symbol=AAPL&limit=20
Auth required. Response: `{ "items": PredictionRun[] }`.

## Chat (OpenAI-backed)

### POST /api/chat
Auth required. Body: `{ "conversationId"?: uuid, "message": string }`.
Response: `{ "conversationId": uuid, "reply": string }`.
Server calls OpenAI with `OPENAI_API_KEY` (never exposed to client) and persists both user and assistant messages to `chat_messages`.

### GET /api/chat/:conversationId
Auth required. Response: `{ "messages": ChatMessage[] }`.

## Error format
All errors follow:
```json
{ "error": { "code": "string", "message": "string" } }
```