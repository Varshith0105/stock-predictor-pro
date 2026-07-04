import { supabase } from "@/integrations/supabase/client";
import { API_CONFIG } from "./config";

export interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FetchStockResponse {
  data: StockDataPoint[];
  message?: string;
  isDemo?: boolean;
  rateLimited?: boolean;
}

export interface FetchStockParams {
  symbol: string;
  startDate: string; // yyyy-MM-dd
  endDate: string;   // yyyy-MM-dd
}

/**
 * Stock data service.
 *
 * Today: proxies to the Lovable Cloud edge function `fetch-stock-data`.
 * After migration: swap the implementation to call `${API_BASE_URL}/api/stocks/history`
 * on the FastAPI backend. Consumers do not need to change.
 */
export const stockService = {
  async fetchHistory(params: FetchStockParams): Promise<FetchStockResponse> {
    if (API_CONFIG.USE_LOVABLE_CLOUD) {
      const { data, error } = await supabase.functions.invoke("fetch-stock-data", {
        body: params,
      });
      if (error) throw error;
      return data as FetchStockResponse;
    }

    const res = await fetch(`${API_CONFIG.API_BASE_URL}/api/stocks/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Stock API error: ${res.status}`);
    return (await res.json()) as FetchStockResponse;
  },
};