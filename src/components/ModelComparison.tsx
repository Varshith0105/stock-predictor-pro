import { motion } from 'framer-motion';
import { Brain, TreeDeciduous, TrendingUp, Layers, Award } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ModelMetrics } from '@/utils/predictions';

interface ModelComparisonProps {
  metrics?: {
    linearRegression: ModelMetrics;
    arima: ModelMetrics;
    randomForest: ModelMetrics;
    lstm: ModelMetrics;
  };
  bestModel?: 'linearRegression' | 'arima' | 'randomForest' | 'lstm';
  stockSymbol?: string;
}

const ModelComparison = ({ metrics, bestModel, stockSymbol }: ModelComparisonProps) => {
  // Default metrics if not provided
  const defaultMetrics: ModelMetrics = { mse: 0.05, r2: 0.7, mae: 0.04 };
  const displayMetrics = metrics || {
    linearRegression: defaultMetrics,
    arima: defaultMetrics,
    randomForest: defaultMetrics,
    lstm: defaultMetrics,
  };
  const displayBestModel = bestModel || 'linearRegression';

  const models = [
    {
      id: 'linearRegression' as const,
      title: 'Linear Regression',
      shortTitle: 'Linear Reg.',
      icon: <TrendingUp className="w-5 h-5" />,
      description: 'Trend-based forecasting',
      metrics: displayMetrics.linearRegression,
    },
    {
      id: 'arima' as const,
      title: 'ARIMA',
      shortTitle: 'ARIMA',
      icon: <Layers className="w-5 h-5" />,
      description: 'Time series analysis',
      metrics: displayMetrics.arima,
    },
    {
      id: 'randomForest' as const,
      title: 'Random Forest',
      shortTitle: 'Random Forest',
      icon: <TreeDeciduous className="w-5 h-5" />,
      description: 'Ensemble decision trees',
      metrics: displayMetrics.randomForest,
    },
    {
      id: 'lstm' as const,
      title: 'LSTM Network',
      shortTitle: 'LSTM',
      icon: <Brain className="w-5 h-5" />,
      description: 'Deep learning patterns',
      metrics: displayMetrics.lstm,
    },
  ];

  const bestModelData = models.find(m => m.id === displayBestModel);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="bg-card border border-border rounded-2xl p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
          <Brain className="w-5 h-5 text-accent" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Model Comparison</h3>
          <p className="text-sm text-muted-foreground">
            4 algorithms competing for {stockSymbol || 'stock'} prediction
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-6">
        {models.map((model) => (
          <ModelCard
            key={model.id}
            title={model.shortTitle}
            icon={model.icon}
            description={model.description}
            mse={model.metrics.mse}
            r2={model.metrics.r2}
            mae={model.metrics.mae}
            isRecommended={model.id === displayBestModel}
          />
        ))}
      </div>

      <div className="flex items-center gap-3 p-4 rounded-xl bg-primary/10 border border-primary/20">
        <Award className="w-5 h-5 text-primary" />
        <div>
          <p className="text-sm font-medium">
            Winner: {bestModelData?.title || 'Linear Regression'}
          </p>
          <p className="text-xs text-muted-foreground">
            Lowest MSE ({displayMetrics[displayBestModel]?.mse.toFixed(4)}) for current data
          </p>
        </div>
      </div>
    </motion.div>
  );
};

interface ModelCardProps {
  title: string;
  icon: React.ReactNode;
  description: string;
  mse: number;
  r2: number;
  mae: number;
  isRecommended: boolean;
}

const ModelCard = ({ title, icon, description, mse, r2, mae, isRecommended }: ModelCardProps) => (
  <div
    className={cn(
      'p-3 rounded-xl border transition-all',
      isRecommended
        ? 'bg-primary/5 border-primary/30 ring-1 ring-primary/20'
        : 'bg-muted/30 border-border'
    )}
  >
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2">
        <div
          className={cn(
            'w-7 h-7 rounded-lg flex items-center justify-center',
            isRecommended ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
          )}
        >
          {icon}
        </div>
        <div>
          <span className="text-sm font-medium">{title}</span>
          {isRecommended && (
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-primary text-primary-foreground font-medium">
              Best
            </span>
          )}
        </div>
      </div>
    </div>
    
    <p className="text-xs text-muted-foreground mb-2">{description}</p>

    <div className="space-y-1.5">
      <MetricRow
        label="MSE"
        value={mse.toFixed(4)}
        isGood={mse < 0.05}
      />
      <MetricRow
        label="R²"
        value={r2.toFixed(3)}
        isGood={r2 > 0.5}
      />
      <MetricRow
        label="MAE"
        value={mae.toFixed(4)}
        isGood={mae < 0.03}
      />
    </div>
  </div>
);

interface MetricRowProps {
  label: string;
  value: string;
  isGood: boolean;
}

const MetricRow = ({ label, value, isGood }: MetricRowProps) => (
  <div className="flex items-center justify-between">
    <span className="text-xs text-muted-foreground">{label}</span>
    <span
      className={cn(
        'text-xs font-mono font-medium',
        isGood ? 'text-success' : 'text-warning'
      )}
    >
      {value}
    </span>
  </div>
);

export default ModelComparison;
