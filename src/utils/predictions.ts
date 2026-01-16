// Simple prediction model using moving average and trend analysis
export interface PredictionResult {
  date: string;
  predictedClose: number;
  confidence: number;
  isPrediction: boolean;
}

export interface HistoricalDataPoint {
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
}

// Calculate Simple Moving Average
const calculateSMA = (data: number[], period: number): number => {
  if (data.length < period) return data[data.length - 1];
  const slice = data.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
};

// Calculate Exponential Moving Average
const calculateEMA = (data: number[], period: number): number => {
  if (data.length === 0) return 0;
  const multiplier = 2 / (period + 1);
  let ema = data[0];
  for (let i = 1; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  return ema;
};

// Calculate trend direction and strength
const calculateTrend = (data: number[]): { direction: number; strength: number } => {
  if (data.length < 2) return { direction: 0, strength: 0 };
  
  const recent = data.slice(-7);
  const older = data.slice(-14, -7);
  
  const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
  const olderAvg = older.length > 0 ? older.reduce((a, b) => a + b, 0) / older.length : recentAvg;
  
  const direction = recentAvg > olderAvg ? 1 : recentAvg < olderAvg ? -1 : 0;
  const strength = Math.abs(recentAvg - olderAvg) / olderAvg;
  
  return { direction, strength };
};

// Calculate volatility (standard deviation)
const calculateVolatility = (data: number[]): number => {
  if (data.length < 2) return 0;
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const squaredDiffs = data.map(x => Math.pow(x - mean, 2));
  return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / data.length);
};

// Generate predictions using ensemble of methods
export const generatePredictions = (
  historicalData: HistoricalDataPoint[],
  daysToPredict: number = 7
): PredictionResult[] => {
  if (historicalData.length < 5) {
    return [];
  }

  const closes = historicalData.map(d => d.close);
  const lastPrice = closes[closes.length - 1];
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  
  // Calculate indicators
  const sma5 = calculateSMA(closes, 5);
  const sma20 = calculateSMA(closes, Math.min(20, closes.length));
  const ema12 = calculateEMA(closes, Math.min(12, closes.length));
  const { direction, strength } = calculateTrend(closes);
  const volatility = calculateVolatility(closes.slice(-20));
  
  // Daily return calculation
  const dailyReturns = closes.slice(1).map((price, i) => (price - closes[i]) / closes[i]);
  const avgDailyReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
  
  const predictions: PredictionResult[] = [];
  let currentPrice = lastPrice;
  
  for (let i = 1; i <= daysToPredict; i++) {
    const nextDate = new Date(lastDate);
    nextDate.setDate(nextDate.getDate() + i);
    
    // Skip weekends
    while (nextDate.getDay() === 0 || nextDate.getDay() === 6) {
      nextDate.setDate(nextDate.getDate() + 1);
    }
    
    // Ensemble prediction combining multiple signals
    const smaSignal = (sma5 + sma20) / 2;
    const emaSignal = ema12;
    const trendSignal = lastPrice * (1 + direction * strength * 0.01 * i);
    const momentumSignal = currentPrice * (1 + avgDailyReturn);
    
    // Weighted average of signals
    const predictedPrice = (
      smaSignal * 0.2 +
      emaSignal * 0.2 +
      trendSignal * 0.3 +
      momentumSignal * 0.3
    );
    
    // Add some realistic noise based on historical volatility
    const noise = (Math.random() - 0.5) * volatility * 0.5;
    const finalPrediction = predictedPrice + noise;
    
    // Confidence decreases with prediction horizon
    const baseConfidence = 0.95;
    const decayFactor = 0.05;
    const confidence = Math.max(0.5, baseConfidence - decayFactor * i);
    
    predictions.push({
      date: nextDate.toISOString().split('T')[0],
      predictedClose: Math.round(finalPrediction * 100) / 100,
      confidence,
      isPrediction: true,
    });
    
    currentPrice = finalPrediction;
  }
  
  return predictions;
};

// Combine historical data with predictions for chart display
export const combineDataWithPredictions = (
  historicalData: HistoricalDataPoint[],
  predictions: PredictionResult[]
): any[] => {
  const combined = historicalData.map(d => ({
    date: d.date,
    close: d.close,
    actual: d.close,
    predicted: null,
    predictedClose: null,
    isPrediction: false,
  }));
  
  // Add predictions
  predictions.forEach(pred => {
    combined.push({
      date: pred.date,
      close: null,
      actual: null,
      predicted: pred.predictedClose,
      predictedClose: pred.predictedClose,
      isPrediction: true,
    });
  });
  
  return combined;
};
