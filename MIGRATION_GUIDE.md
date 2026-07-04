# Migration Guide: Lovable Cloud → Independent Backend

This document inventories every Lovable Cloud dependency in the project and
describes the target architecture (FastAPI + PostgreSQL + JWT auth + OpenAI)
that will replace it after export.

> Status: **preparation only**. The current app still runs on Lovable Cloud.
> All Cloud calls have been routed through `src/services/*` so backend URLs
> can be swapped from a single config file (`src/services/config.ts`).

---

## 1. Lovable Cloud dependencies found

| Area | Location | Notes |
|---|---|---|
| SDK | `@supabase/supabase-js` in `package.json` | Client SDK. |
| Client bootstrap | `src/integrations/supabase/client.ts` | Auto-generated. Reads `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`. |
| Generated types | `src/integrations/supabase/types.ts` | Auto-generated DB types (currently empty schema). |
| Env vars | `.env`: `VITE_SUPABASE_PROJECT_ID`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY` | Injected at build time by Vite. |
| Edge function | `supabase/functions/fetch-stock-data/index.ts` | Proxies Alpha Vantage. Uses secret `ALPHA_VANTAGE_API_KEY`. |
| Edge config | `supabase/config.toml` | `verify_jwt = false` for `fetch-stock-data`. |
| Server secrets | `ALPHA_VANTAGE_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL`, `SUPABASE_PUBLISHABLE_KEY` | Only `ALPHA_VANTAGE_API_KEY` is app-owned. |
| Auth | None currently in use | No `supabase.auth` calls in codebase. |
| Database | None (empty `public` schema) | No tables today. |
| Storage | None | No buckets. |
| SDK call sites (before service layer) | `src/components/DateRangeSelector.tsx` | Now migrated to `stockService`. |

## 2. Files that must be modified/removed after migration

- **Remove:** `src/integrations/supabase/client.ts`, `src/integrations/supabase/types.ts`, `supabase/` directory, `@supabase/supabase-js` from `package.json`.
- **Modify:** `.env` — drop `VITE_SUPABASE_*`, add `VITE_API_BASE_URL`.
- **Modify:** `src/services/config.ts` — set `USE_LOVABLE_CLOUD = false`.
- **Modify:** `src/services/stockService.ts` — the `USE_LOVABLE_CLOUD` branch can be deleted once the FastAPI backend is live.
- **Add:** `backend/` (FastAPI), `database/` (SQL + migrations), `docs/` (this file, OpenAPI, spec).

## 3. APIs to be recreated

See [`BACKEND_API_SPEC.md`](./BACKEND_API_SPEC.md) and [`OPENAPI.yaml`](./OPENAPI.yaml).

Minimum surface to reach parity:

- `POST /api/stocks/history` — replaces the `fetch-stock-data` edge function.
- `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/refresh`, `POST /api/auth/logout`, `GET /api/auth/me` — new JWT auth (no equivalent exists today).
- `GET/POST /api/predictions` — persist prediction runs (new, optional).
- `GET/POST /api/chat` — OpenAI-backed chat with history (new, optional).

## 4. Database schema

See [`DATABASE_SCHEMA.sql`](./DATABASE_SCHEMA.sql). Tables: `users`, `refresh_tokens`, `prediction_runs`, `chat_messages`.

## 5. Authentication flow (target)

1. `POST /api/auth/register` → hashed password stored in `users`, returns access + refresh JWT.
2. `POST /api/auth/login` → verifies bcrypt hash, returns access (15 min) + refresh (7 day) tokens.
3. Frontend stores access token in memory, refresh token in httpOnly cookie.
4. `POST /api/auth/refresh` rotates the refresh token (row in `refresh_tokens`).
5. `POST /api/auth/logout` revokes the refresh token.
6. Protected routes require `Authorization: Bearer <access_token>`; FastAPI dependency validates signature + `exp`.

## 6. Edge functions to migrate

| Edge function | Replacement |
|---|---|
| `fetch-stock-data` | `POST /api/stocks/history` in FastAPI. Port Alpha Vantage logic + demo fallback verbatim from `supabase/functions/fetch-stock-data/index.ts`. |

## 7. Environment variables

**Frontend (`.env`):**
```
VITE_API_BASE_URL=https://api.yourdomain.com
```
(Remove all `VITE_SUPABASE_*`.)

**Backend (`backend/.env`):**
```
DATABASE_URL=postgresql://user:pass@host:5432/stockai
JWT_SECRET=<32+ char random>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
ALPHA_VANTAGE_API_KEY=<from vendor>
OPENAI_API_KEY=<from openai>
CORS_ORIGINS=https://app.yourdomain.com
```

## 8. Deployment

- **Frontend → Vercel:** import repo, framework = Vite, set `VITE_API_BASE_URL`.
- **Backend → Render/Railway:** Dockerfile with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, inject env vars above.
- **Database → Neon (Postgres):** run `DATABASE_SCHEMA.sql`, then wire `DATABASE_URL`.
- Configure CORS in FastAPI to whitelist the Vercel domain.

## 9. Local run (unchanged today)

```
npm install
npm run dev
```
After migration, additionally:
```
cd backend && uvicorn app.main:app --reload
```

## 10. Final checklist (post-GitHub export)

- [ ] Delete `supabase/` directory.
- [ ] Delete `src/integrations/supabase/`.
- [ ] Remove `@supabase/supabase-js` from `package.json`.
- [ ] Replace `VITE_SUPABASE_*` env vars with `VITE_API_BASE_URL`.
- [ ] Flip `USE_LOVABLE_CLOUD` to `false` in `src/services/config.ts`.
- [ ] Delete the `USE_LOVABLE_CLOUD` branch from every file in `src/services/`.
- [ ] Scaffold FastAPI backend (`backend/app/{routes,services,models,utils}`).
- [ ] Port `fetch-stock-data` logic into `backend/app/services/stocks.py`.
- [ ] Implement JWT auth per section 5.
- [ ] Run `DATABASE_SCHEMA.sql` on Neon/Supabase Postgres.
- [ ] Configure CORS + secrets on Render/Railway.
- [ ] Verify `npm run dev` still boots against local FastAPI.
- [ ] Remove Lovable-specific badges/branding if desired.