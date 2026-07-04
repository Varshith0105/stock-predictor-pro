// Central API configuration.
// After migration off Lovable Cloud, point API_BASE_URL to your FastAPI backend
// (e.g. https://api.yourdomain.com) and set USE_LOVABLE_CLOUD to false.

export const API_CONFIG = {
  // Toggle between Lovable Cloud edge functions and a future standalone backend.
  USE_LOVABLE_CLOUD: true,
  // Base URL for the future REST backend (FastAPI). Ignored while USE_LOVABLE_CLOUD is true.
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
} as const;