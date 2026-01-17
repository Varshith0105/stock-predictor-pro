import { motion } from 'framer-motion';
import { Brain, TrendingUp, BarChart3, Activity, Award } from 'lucide-react';
import { cn } from '@/lib/utils';
import { AllModelMetrics, ModelMetrics } from '@/utils/predictions';

interface ModelComparisonProps {
  metrics?: AllModelMetrics | null;
  stockSymbol?: string;
}

const ModelComparison = ({ metrics, stockSymbol }: ModelComparisonProps) => {
  const hasData = metrics !== null && metrics !== undefined;

  const models = hasData ? [
    {
      id: 'linearRegression',
      name: 'Linear Regression',
      icon: TrendingUp,
      description: 'Trend-based statistical model',
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      metrics: metrics.linearRegression,
    },
    {
      id: 'arima',
      name: 'ARIMA',
      icon: Activity,
      description: 'Auto-regressive integrated moving average',
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      metrics: metrics.arima,
    },
    {
      id: 'randomForest',
      name: 'Random Forest',
      icon: BarChart3,
      description: 'Ensemble decision tree model',
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      metrics: metrics.randomForest,
    },
    {
      id: 'lstm',
      name: 'LSTM Neural Network',
      icon: Brain,
      description: 'Long Short-Term Memory deep learning',
      color: 'text-primary',
      bgColor: 'bg-primary/10',
      metrics: metrics.lstm,
    },
  ] : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="bg-card border border-border rounded-2xl p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <Brain className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Model Performance Comparison</h3>
          <p className="text-sm text-muted-foreground">
            Comparing 4 prediction models for {stockSymbol || 'stock'}
          </p>
        </div>
      </div>

      {!hasData && (
        <div className="text-center py-8 text-muted-foreground">
          <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Fetch stock data to see model performance comparison</p>
        </div>
      )}

      {hasData && (
        <div className="space-y-4">
          {models.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 * index }}
              className={cn(
                "relative border rounded-xl p-4 transition-all",
                metrics.bestModel === model.id
                  ? "border-primary bg-primary/5"
                  : "border-border bg-muted/20"
              )}
            >
              {/* Best Model Badge */}
              {metrics.bestModel === model.id && (
                <div className="absolute -top-2 -right-2 bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full flex items-center gap-1">
                  <Award className="w-3 h-3" />
                  Best
                </div>
              )}

              {/* Model Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", model.bgColor)}>
                  <model.icon className={cn("w-4 h-4", model.color)} />
                </div>
                <div>
                  <h4 className="font-semibold text-sm">{model.name}</h4>
                  <p className="text-xs text-muted-foreground">{model.description}</p>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-4 gap-3">
                <MetricBox
                  label="MSE"
                  value={model.metrics.mse.toFixed(4)}
                  isGood={model.metrics.mse < 0.05}
                  tooltip="Mean Squared Error"
                />
                <MetricBox
                  label="R² Score"
                  value={model.metrics.r2.toFixed(3)}
                  isGood={model.metrics.r2 > 0.5}
                  tooltip="Coefficient of Determination"
                />
                <MetricBox
                  label="MAE"
                  value={model.metrics.mae.toFixed(4)}
                  isGood={model.metrics.mae < 0.03}
                  tooltip="Mean Absolute Error"
                />
                <MetricBox
                  label="Accuracy"
                  value={`${model.metrics.accuracy.toFixed(1)}%`}
                  isGood={model.metrics.accuracy > 55}
                  tooltip="Direction Prediction Accuracy"
                  highlight
                />
              </div>
            </motion.div>
          ))}

          {/* Legend */}
          <div className="mt-4 p-3 bg-muted/30 rounded-lg">
            <p className="text-xs text-muted-foreground">
              <strong>Metrics Guide:</strong> MSE & MAE (lower is better) • R² Score & Accuracy (higher is better) • 
              Best model selected based on R² score
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
};

interface MetricBoxProps {
  label: string;
  value: string;
  isGood: boolean;
  tooltip: string;
  highlight?: boolean;
}

const MetricBox = ({ label, value, isGood, tooltip, highlight }: MetricBoxProps) => (
  <div
    className={cn(
      "text-center p-2 rounded-lg",
      highlight ? "bg-background border border-border" : "bg-muted/30"
    )}
    title={tooltip}
  >
    <p className="text-xs text-muted-foreground mb-0.5">{label}</p>
    <p
      className={cn(
        'text-sm font-mono font-bold',
        isGood ? 'text-success' : 'text-warning'
      )}
    >
      {value}
    </p>
  </div>
);

export default ModelComparison;
