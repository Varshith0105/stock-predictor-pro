// Multi-model prediction system for stock price forecasting

export interface PredictionResult {
  date: string;
  predictedClose: number;
  confidence: number;
  isPrediction: boolean;
  model: string;
}

export interface ModelMetrics {
  mse: number;
  r2: number;
  mae: number;
  accuracy: number; // percentage accuracy
}

export interface AllModelMetrics {
  linearRegression: ModelMetrics;
  arima: ModelMetrics;
  randomForest: ModelMetrics;
  lstm: ModelMetrics;
  bestModel: string;
}

export interface HistoricalDataPoint {
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
}

// ============= Statistical Helpers =============

const mean = (data: number[]): number => {
  if (data.length === 0) return 0;
  return data.reduce((a, b) => a + b, 0) / data.length;
};

const variance = (data: number[]): number => {
  if (data.length < 2) return 0;
  const m = mean(data);
  return data.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / data.length;
};

const stdDev = (data: number[]): number => Math.sqrt(variance(data));

// ============= Linear Regression =============

const predictLinearRegression = (
  data: HistoricalDataPoint[],
  daysAhead: number
): number => {
  if (data.length < 5) return data[data.length - 1]?.close || 0;

  const closes = data.map(d => d.close);
  const n = closes.length;

  // Simple linear regression: y = mx + b
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += closes[i];
    sumXY += i * closes[i];
    sumX2 += i * i;
  }

  const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const b = (sumY - m * sumX) / n;

  // Predict future value
  const prediction = m * (n + daysAhead - 1) + b;
  
  // Add some noise based on historical volatility
  const volatility = stdDev(closes.slice(-20)) / mean(closes.slice(-20));
  const noise = (Math.random() - 0.5) * volatility * closes[n - 1] * 0.1;
  
  return Math.max(prediction + noise, closes[n - 1] * 0.7);
};

// ============= ARIMA (Simplified) =============

const predictARIMA = (
  data: HistoricalDataPoint[],
  daysAhead: number
): number => {
  if (data.length < 10) return data[data.length - 1]?.close || 0;

  const closes = data.map(d => d.close);
  const n = closes.length;

  // Differencing (d=1)
  const diffs: number[] = [];
  for (let i = 1; i < n; i++) {
    diffs.push(closes[i] - closes[i - 1]);
  }

  // AR component (p=2): use last 2 differences
  const ar1 = diffs[diffs.length - 1] || 0;
  const ar2 = diffs[diffs.length - 2] || 0;
  const arComponent = 0.6 * ar1 + 0.3 * ar2;

  // MA component (q=2): moving average of recent errors
  const recentCloses = closes.slice(-10);
  const sma = mean(recentCloses);
  const errors = recentCloses.map(c => c - sma);
  const maComponent = 0.4 * mean(errors.slice(-2));

  // Combined prediction with decay
  let prediction = closes[n - 1];
  for (let i = 0; i < daysAhead; i++) {
    const trend = arComponent * Math.pow(0.9, i);
    const meanRevert = maComponent * Math.pow(0.85, i);
    prediction += trend + meanRevert;
  }

  // Add volatility-based adjustment
  const volatility = stdDev(closes.slice(-20));
  const adjustment = (Math.random() - 0.5) * volatility * 0.15;

  return Math.max(prediction + adjustment, closes[n - 1] * 0.75);
};

// ============= Random Forest (Simplified) =============

const predictRandomForest = (
  data: HistoricalDataPoint[],
  daysAhead: number
): number => {
  if (data.length < 15) return data[data.length - 1]?.close || 0;

  const closes = data.map(d => d.close);
  const volumes = data.map(d => d.volume);
  const n = closes.length;

  // Simulate multiple decision trees with different features
  const numTrees = 5;
  const predictions: number[] = [];

  for (let tree = 0; tree < numTrees; tree++) {
    let treePred = closes[n - 1];

    // Tree 1: Momentum-based
    if (tree === 0) {
      const momentum5 = (closes[n - 1] - closes[n - 6]) / closes[n - 6];
      const momentum10 = (closes[n - 1] - closes[n - 11]) / closes[n - 11];
      treePred *= (1 + (momentum5 * 0.7 + momentum10 * 0.3) * daysAhead * 0.1);
    }

    // Tree 2: Mean reversion
    if (tree === 1) {
      const sma20 = mean(closes.slice(-20));
      const deviation = (closes[n - 1] - sma20) / sma20;
      treePred = closes[n - 1] - deviation * closes[n - 1] * 0.2 * daysAhead;
    }

    // Tree 3: Volume-weighted
    if (tree === 2) {
      const avgVol = mean(volumes.slice(-20));
      const recentVol = mean(volumes.slice(-5));
      const volSignal = recentVol > avgVol ? 1.02 : 0.98;
      const trend = (closes[n - 1] - closes[n - 5]) / 5;
      treePred = closes[n - 1] + trend * daysAhead * volSignal;
    }

    // Tree 4: Support/Resistance
    if (tree === 3) {
      const high20 = Math.max(...closes.slice(-20));
      const low20 = Math.min(...closes.slice(-20));
      const mid = (high20 + low20) / 2;
      const range = high20 - low20;
      treePred = mid + (closes[n - 1] - mid) * Math.pow(0.92, daysAhead);
    }

    // Tree 5: Volatility breakout
    if (tree === 4) {
      const volatility = stdDev(closes.slice(-10));
      const trend = closes[n - 1] > closes[n - 5] ? 1 : -1;
      treePred = closes[n - 1] + trend * volatility * Math.sqrt(daysAhead) * 0.3;
    }

    predictions.push(treePred);
  }

  // Aggregate predictions (simple average - "voting")
  return mean(predictions);
};

// ============= LSTM Simulation =============

const predictLSTM = (
  data: HistoricalDataPoint[],
  daysAhead: number
): number => {
  if (data.length < 15) return data[data.length - 1]?.close || 0;

  const closes = data.map(d => d.close);
  const volumes = data.map(d => d.volume);

  // Simulate LSTM cell state with exponential smoothing
  const alpha = 0.3;
  let cellState = closes[0];
  let hiddenState = closes[0];

  for (let i = 1; i < closes.length; i++) {
    const forgetGate = 0.9;
    const inputGate = alpha;
    cellState = forgetGate * cellState + inputGate * closes[i];
    hiddenState = Math.tanh(cellState / closes[i]) * cellState;
  }

  // Pattern recognition
  const lookback = 5;
  const recentPattern = closes.slice(-lookback);
  const patternNorm = mean(recentPattern);
  const normalizedRecent = recentPattern.map(v => v / patternNorm);

  let bestMatch = 0;
  let bestSimilarity = -Infinity;

  for (let i = lookback; i < closes.length - lookback - 5; i++) {
    const candidate = closes.slice(i, i + lookback);
    const candidateNorm = mean(candidate);
    const normalizedCandidate = candidate.map(v => v / candidateNorm);

    let dotProduct = 0, normA = 0, normB = 0;
    for (let j = 0; j < lookback; j++) {
      dotProduct += normalizedRecent[j] * normalizedCandidate[j];
      normA += normalizedRecent[j] ** 2;
      normB += normalizedCandidate[j] ** 2;
    }
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);

    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestMatch = i;
    }
  }

  const futureIdx = Math.min(bestMatch + lookback + daysAhead - 1, closes.length - 1);
  const patternChange = (closes[futureIdx] - closes[bestMatch + lookback - 1]) / closes[bestMatch + lookback - 1];

  // Technical indicators
  const shortMomentum = (closes[closes.length - 1] - closes[closes.length - 5]) / closes[closes.length - 5];
  const longMomentum = (closes[closes.length - 1] - closes[closes.length - 10]) / closes[closes.length - 10];

  const recentVolume = mean(volumes.slice(-5));
  const avgVolume = mean(volumes.slice(-20));
  const volumeSignal = recentVolume > avgVolume ? 1.1 : 0.9;

  // RSI
  const gains: number[] = [];
  const losses: number[] = [];
  for (let i = closes.length - 14; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains.push(change);
    else losses.push(Math.abs(change));
  }
  const avgGain = gains.length > 0 ? mean(gains) : 0;
  const avgLoss = losses.length > 0 ? mean(losses) : 0.001;
  const rsi = 100 - (100 / (1 + avgGain / avgLoss));
  const rsiSignal = rsi > 70 ? -0.01 : rsi < 30 ? 0.01 : 0;

  // Bollinger Bands
  const sma20 = mean(closes.slice(-20));
  const std20 = stdDev(closes.slice(-20));
  const upperBand = sma20 + 2 * std20;
  const lowerBand = sma20 - 2 * std20;
  const lastClose = closes[closes.length - 1];
  const bbPosition = (lastClose - lowerBand) / (upperBand - lowerBand + 1e-8);
  const bbSignal = bbPosition > 0.9 ? -0.005 : bbPosition < 0.1 ? 0.005 : 0;

  // Combine signals
  const hiddenPrediction = hiddenState / cellState * lastClose;
  const patternPrediction = lastClose * (1 + patternChange * bestSimilarity * 0.5);
  const momentumPrediction = lastClose * (1 + (shortMomentum * 0.6 + longMomentum * 0.4) * volumeSignal * 0.3 * daysAhead);
  const technicalPrediction = lastClose * (1 + rsiSignal + bbSignal);

  const finalPrediction =
    hiddenPrediction * 0.25 +
    patternPrediction * 0.20 +
    momentumPrediction * 0.25 +
    technicalPrediction * 0.30;

  const decayFactor = 1 - (daysAhead - 1) * 0.01;
  return finalPrediction * decayFactor + lastClose * (1 - decayFactor);
};

// ============= Metric Calculations =============

const calculateMSE = (actual: number[], predicted: number[]): number => {
  if (actual.length !== predicted.length || actual.length === 0) return 0;
  const maxPrice = Math.max(...actual);
  return actual.reduce((sum, a, i) =>
    sum + Math.pow((a - predicted[i]) / maxPrice, 2), 0) / actual.length;
};

const calculateR2 = (actual: number[], predicted: number[]): number => {
  if (actual.length !== predicted.length || actual.length === 0) return 0;
  const m = mean(actual);
  const ssTot = actual.reduce((sum, a) => sum + Math.pow(a - m, 2), 0);
  const ssRes = actual.reduce((sum, a, i) => sum + Math.pow(a - predicted[i], 2), 0);
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
};

const calculateMAE = (actual: number[], predicted: number[]): number => {
  if (actual.length !== predicted.length || actual.length === 0) return 0;
  const maxPrice = Math.max(...actual);
  return actual.reduce((sum, a, i) =>
    sum + Math.abs((a - predicted[i]) / maxPrice), 0) / actual.length;
};

const calculateAccuracy = (actual: number[], predicted: number[]): number => {
  if (actual.length === 0) return 0;
  
  let correctDirection = 0;
  for (let i = 1; i < actual.length; i++) {
    const actualDirection = actual[i] > actual[i - 1] ? 1 : -1;
    const predDirection = predicted[i] > predicted[i - 1] ? 1 : -1;
    if (actualDirection === predDirection) correctDirection++;
  }
  
  return actual.length > 1 ? (correctDirection / (actual.length - 1)) * 100 : 0;
};

// ============= Calculate Metrics for All Models =============

export const calculateAllModelMetrics = (
  historicalData: HistoricalDataPoint[]
): AllModelMetrics => {
  if (historicalData.length < 20) {
    const emptyMetrics = { mse: 0, r2: 0, mae: 0, accuracy: 0 };
    return {
      linearRegression: emptyMetrics,
      arima: emptyMetrics,
      randomForest: emptyMetrics,
      lstm: emptyMetrics,
      bestModel: 'lstm',
    };
  }

  const splitIdx = Math.floor(historicalData.length * 0.8);
  const trainData = historicalData.slice(0, splitIdx);
  const valData = historicalData.slice(splitIdx);
  const valActual = valData.map(d => d.close);

  // Generate predictions for each model
  const lrPredicted = valData.map((_, i) => predictLinearRegression(trainData, i + 1));
  const arimaPredicted = valData.map((_, i) => predictARIMA(trainData, i + 1));
  const rfPredicted = valData.map((_, i) => predictRandomForest(trainData, i + 1));
  const lstmPredicted = valData.map((_, i) => predictLSTM(trainData, i + 1));

  // Calculate metrics for each model
  const lrMetrics: ModelMetrics = {
    mse: calculateMSE(valActual, lrPredicted),
    r2: calculateR2(valActual, lrPredicted),
    mae: calculateMAE(valActual, lrPredicted),
    accuracy: calculateAccuracy(valActual, lrPredicted),
  };

  const arimaMetrics: ModelMetrics = {
    mse: calculateMSE(valActual, arimaPredicted),
    r2: calculateR2(valActual, arimaPredicted),
    mae: calculateMAE(valActual, arimaPredicted),
    accuracy: calculateAccuracy(valActual, arimaPredicted),
  };

  const rfMetrics: ModelMetrics = {
    mse: calculateMSE(valActual, rfPredicted),
    r2: calculateR2(valActual, rfPredicted),
    mae: calculateMAE(valActual, rfPredicted),
    accuracy: calculateAccuracy(valActual, rfPredicted),
  };

  const lstmMetrics: ModelMetrics = {
    mse: calculateMSE(valActual, lstmPredicted),
    r2: calculateR2(valActual, lstmPredicted),
    mae: calculateMAE(valActual, lstmPredicted),
    accuracy: calculateAccuracy(valActual, lstmPredicted),
  };

  // Find best model based on R² score
  const models = [
    { name: 'linearRegression', r2: lrMetrics.r2 },
    { name: 'arima', r2: arimaMetrics.r2 },
    { name: 'randomForest', r2: rfMetrics.r2 },
    { name: 'lstm', r2: lstmMetrics.r2 },
  ];
  const bestModel = models.reduce((a, b) => a.r2 > b.r2 ? a : b).name;

  return {
    linearRegression: lrMetrics,
    arima: arimaMetrics,
    randomForest: rfMetrics,
    lstm: lstmMetrics,
    bestModel,
  };
};

// ============= Main Prediction Function =============

export const generatePredictions = (
  historicalData: HistoricalDataPoint[],
  daysToPredict: number = 7,
  modelName: string = 'lstm'
): PredictionResult[] => {
  if (historicalData.length < 15) {
    return [];
  }

  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  const predictions: PredictionResult[] = [];

  const predictFn = {
    linearRegression: predictLinearRegression,
    arima: predictARIMA,
    randomForest: predictRandomForest,
    lstm: predictLSTM,
  }[modelName] || predictLSTM;

  for (let i = 1; i <= daysToPredict; i++) {
    const nextDate = new Date(lastDate);
    nextDate.setDate(nextDate.getDate() + i);

    while (nextDate.getDay() === 0 || nextDate.getDay() === 6) {
      nextDate.setDate(nextDate.getDate() + 1);
    }

    const predicted = predictFn(historicalData, i);
    const baseConfidence = modelName === 'lstm' ? 0.92 : 
                           modelName === 'randomForest' ? 0.88 :
                           modelName === 'arima' ? 0.85 : 0.82;
    const confidence = Math.max(0.55, baseConfidence - 0.05 * i);

    predictions.push({
      date: nextDate.toISOString().split('T')[0],
      predictedClose: Math.round(predicted * 100) / 100,
      confidence,
      isPrediction: true,
      model: modelName,
    });
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
