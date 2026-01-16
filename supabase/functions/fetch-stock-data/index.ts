import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const ALPHA_VANTAGE_API_KEY = Deno.env.get('ALPHA_VANTAGE_API_KEY');

// Map company symbols to Alpha Vantage compatible symbols
const symbolMap: Record<string, string> = {
  'AAPL': 'AAPL',
  'MSFT': 'MSFT',
  'GOOGL': 'GOOGL',
  'AMZN': 'AMZN',
  'TSLA': 'TSLA',
  'META': 'META',
  'NFLX': 'NFLX',
  'IBM': 'IBM',
  'INTC': 'INTC',
  'NVDA': 'NVDA',
  'ORCL': 'ORCL',
  'SMSN': '005930.KS', // Samsung on Korean exchange
  'SONY': 'SONY',
  'TM': 'TM',
  'WMT': 'WMT',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbol, startDate, endDate } = await req.json();
    
    console.log(`Fetching stock data for ${symbol} from ${startDate} to ${endDate}`);

    if (!ALPHA_VANTAGE_API_KEY) {
      console.error('Alpha Vantage API key not configured');
      throw new Error('Alpha Vantage API key not configured');
    }

    const mappedSymbol = symbolMap[symbol] || symbol;
    
    // Fetch daily time series data
    const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${mappedSymbol}&outputsize=full&apikey=${ALPHA_VANTAGE_API_KEY}`;
    
    console.log(`Calling Alpha Vantage API for symbol: ${mappedSymbol}`);
    
    const response = await fetch(url);
    const data = await response.json();

    if (data['Error Message']) {
      console.error('Alpha Vantage error:', data['Error Message']);
      throw new Error(`Invalid symbol: ${symbol}`);
    }

    if (data['Note']) {
      console.warn('Alpha Vantage rate limit warning:', data['Note']);
      // Return cached/demo data when rate limited
      return new Response(JSON.stringify({ 
        data: [], 
        message: 'API rate limited. Please wait a minute and try again.',
        rateLimited: true 
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const timeSeries = data['Time Series (Daily)'];
    
    if (!timeSeries) {
      console.error('No time series data received:', JSON.stringify(data));
      throw new Error('No data available for this symbol');
    }

    // Filter and format the data based on date range
    const startDateObj = new Date(startDate);
    const endDateObj = new Date(endDate);
    
    const filteredData = Object.entries(timeSeries)
      .filter(([date]) => {
        const dateObj = new Date(date);
        return dateObj >= startDateObj && dateObj <= endDateObj;
      })
      .map(([date, values]: [string, any]) => ({
        date,
        open: parseFloat(values['1. open']),
        high: parseFloat(values['2. high']),
        low: parseFloat(values['3. low']),
        close: parseFloat(values['4. close']),
        volume: parseInt(values['5. volume']),
      }))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

    console.log(`Returning ${filteredData.length} data points`);

    return new Response(JSON.stringify({ data: filteredData }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('Error in fetch-stock-data function:', errorMessage);
    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});
