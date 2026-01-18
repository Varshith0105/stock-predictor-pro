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

// Generate realistic demo data when API is unavailable
const generateDemoData = (startDate: string, endDate: string, symbol: string): any[] => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const data: any[] = [];
  
  // Base prices for different symbols
  const basePrices: Record<string, number> = {
    'AAPL': 185, 'MSFT': 420, 'GOOGL': 175, 'AMZN': 220, 'TSLA': 250,
    'META': 580, 'NFLX': 920, 'IBM': 220, 'INTC': 22, 'NVDA': 140,
    'ORCL': 175, 'SONY': 95, 'TM': 230, 'WMT': 95,
  };
  
  let price = basePrices[symbol] || 150;
  const volatility = price * 0.02; // 2% daily volatility
  
  const current = new Date(start);
  while (current <= end) {
    // Skip weekends
    if (current.getDay() !== 0 && current.getDay() !== 6) {
      // Random walk with slight upward bias
      const change = (Math.random() - 0.48) * volatility;
      price = Math.max(price + change, price * 0.9);
      
      const dayVolatility = volatility * (0.5 + Math.random());
      const open = price + (Math.random() - 0.5) * dayVolatility;
      const high = Math.max(open, price) + Math.random() * dayVolatility;
      const low = Math.min(open, price) - Math.random() * dayVolatility;
      
      data.push({
        date: current.toISOString().split('T')[0],
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(price * 100) / 100,
        volume: Math.floor(10000000 + Math.random() * 50000000),
      });
    }
    current.setDate(current.getDate() + 1);
  }
  
  return data;
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
    
    // Fetch daily time series data (compact returns last 100 data points, free tier)
    const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${mappedSymbol}&outputsize=compact&apikey=${ALPHA_VANTAGE_API_KEY}`;
    
    console.log(`Calling Alpha Vantage API for symbol: ${mappedSymbol}`);
    
    const response = await fetch(url);
    const data = await response.json();

    if (data['Error Message']) {
      console.error('Alpha Vantage error:', data['Error Message']);
      throw new Error(`Invalid symbol: ${symbol}`);
    }

    // Check for rate limiting or other API messages
    if (data['Note'] || data['Information']) {
      const message = data['Note'] || data['Information'];
      console.warn('Alpha Vantage API message:', message);
      
      // Generate demo data for the requested period
      const demoData = generateDemoData(startDate, endDate, symbol);
      return new Response(JSON.stringify({ 
        data: demoData, 
        message: 'Using demo data (API rate limited)',
        isDemo: true 
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const timeSeries = data['Time Series (Daily)'];
    
    if (!timeSeries) {
      console.error('No time series data received:', JSON.stringify(data));
      // Fallback to demo data
      const demoData = generateDemoData(startDate, endDate, symbol);
      return new Response(JSON.stringify({ 
        data: demoData, 
        message: 'Using demo data',
        isDemo: true 
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
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

