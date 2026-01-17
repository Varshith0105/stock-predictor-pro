import { motion } from 'framer-motion';
import { Brain, Activity, Target, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ModelMetrics } from '@/utils/predictions';

interface ModelComparisonProps {
  metrics?: ModelMetrics | null;
  stockSymbol?: string;
}

const ModelComparison = ({ metrics, stockSymbol }: ModelComparisonProps) => {
  const displayMetrics = metrics || { mse: 0, r2: 0, mae: 0 };
  const hasData = metrics !== null && metrics !== undefined;

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
          <h3 className="text-lg font-semibold">LSTM Neural Network</h3>
          <p className="text-sm text-muted-foreground">
            Deep learning model for {stockSymbol || 'stock'} prediction
          </p>
        </div>
      </div>

      {!hasData && (
        <div className="text-center py-8 text-muted-foreground">
          <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Fetch stock data to see LSTM model performance</p>
        </div>
      )}

      {hasData && (
        <>
          {/* LSTM Architecture Info */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-muted/30 rounded-xl p-4 text-center">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center mx-auto mb-2">
                <Activity className="w-4 h-4 text-primary" />
              </div>
              <p className="text-xs text-muted-foreground mb-1">Hidden State</p>
              <p className="text-sm font-semibold">Pattern Memory</p>
            </div>
            <div className="bg-muted/30 rounded-xl p-4 text-center">
              <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center mx-auto mb-2">
                <Target className="w-4 h-4 text-accent" />
              </div>
              <p className="text-xs text-muted-foreground mb-1">Cell State</p>
              <p className="text-sm font-semibold">Long-term Trends</p>
            </div>
            <div className="bg-muted/30 rounded-xl p-4 text-center">
              <div className="w-8 h-8 rounded-lg bg-success/20 flex items-center justify-center mx-auto mb-2">
                <TrendingUp className="w-4 h-4 text-success" />
              </div>
              <p className="text-xs text-muted-foreground mb-1">Indicators</p>
              <p className="text-sm font-semibold">RSI, BB, Volume</p>
            </div>
          </div>

          {/* Model Metrics */}
          <div className="bg-primary/5 border border-primary/20 rounded-xl p-4">
            <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
              <Brain className="w-4 h-4 text-primary" />
              Model Performance Metrics
            </h4>
            <div className="grid grid-cols-3 gap-4">
              <MetricCard
                label="MSE"
                value={displayMetrics.mse.toFixed(4)}
                description="Mean Squared Error"
                isGood={displayMetrics.mse < 0.05}
              />
              <MetricCard
                label="R² Score"
                value={displayMetrics.r2.toFixed(3)}
                description="Variance Explained"
                isGood={displayMetrics.r2 > 0.5}
              />
              <MetricCard
                label="MAE"
                value={displayMetrics.mae.toFixed(4)}
                description="Mean Absolute Error"
                isGood={displayMetrics.mae < 0.03}
              />
            </div>
          </div>

          {/* LSTM Features */}
          <div className="mt-4 p-4 bg-muted/20 rounded-xl">
            <p className="text-xs text-muted-foreground">
              <strong>LSTM Features:</strong> Forget Gate, Input Gate, Output Gate, 
              Pattern Recognition, Momentum Analysis, Volume Signals, RSI, Bollinger Bands
            </p>
          </div>
        </>
      )}
    </motion.div>
  );
};

interface MetricCardProps {
  label: string;
  value: string;
  description: string;
  isGood: boolean;
}

const MetricCard = ({ label, value, description, isGood }: MetricCardProps) => (
  <div className="text-center">
    <p className="text-xs text-muted-foreground mb-1">{label}</p>
    <p
      className={cn(
        'text-xl font-mono font-bold',
        isGood ? 'text-success' : 'text-warning'
      )}
    >
      {value}
    </p>
    <p className="text-xs text-muted-foreground mt-1">{description}</p>
  </div>
);

export default ModelComparison;
