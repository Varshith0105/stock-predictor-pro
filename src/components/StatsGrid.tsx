import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, Target, BarChart3 } from 'lucide-react';
import { CompanyData } from '@/data/stockData';
import { cn } from '@/lib/utils';

interface StatsGridProps {
  company: CompanyData;
  selectedModel?: string;
}

const modelDisplayNames: Record<string, string> = {
  lstm: 'LSTM Neural Network',
  randomForest: 'Random Forest',
  arima: 'ARIMA',
  linearRegression: 'Linear Regression',
};

const StatsGrid = ({ company, selectedModel = 'lstm' }: StatsGridProps) => {
  const latestData = company.data[company.data.length - 1];
  const firstData = company.data[0];
  const priceChange = latestData.actual - firstData.actual;
  const priceChangePercent = (priceChange / firstData.actual) * 100;
  
  // Calculate prediction accuracy
  const predictedData = company.data.filter((d) => d.predicted);
  const avgAccuracy =
    predictedData.length > 0
      ? predictedData.reduce((acc, d) => {
          const error = Math.abs((d.predicted! - d.actual) / d.actual) * 100;
          return acc + (100 - error);
        }, 0) / predictedData.length
      : 0;

  const stats = [
    {
      label: 'Current Price',
      value: `$${company.currentPrice.toFixed(2)}`,
      change: company.changePercent,
      icon: <Activity className="w-5 h-5" />,
      color: 'accent',
    },
    {
      label: 'Period Change',
      value: `${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(2)}%`,
      subValue: `$${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}`,
      icon: priceChange >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />,
      color: priceChange >= 0 ? 'success' : 'destructive',
    },
    {
      label: 'Model Accuracy',
      value: `${avgAccuracy.toFixed(1)}%`,
      subValue: 'Avg prediction',
      icon: <Target className="w-5 h-5" />,
      color: avgAccuracy > 95 ? 'success' : avgAccuracy > 90 ? 'warning' : 'destructive',
    },
    {
      label: 'Selected Model',
      value: modelDisplayNames[selectedModel] || selectedModel,
      subValue: 'Currently active',
      icon: <BarChart3 className="w-5 h-5" />,
      color: 'primary',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-card border border-border rounded-xl p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <div
              className={cn(
                'w-10 h-10 rounded-lg flex items-center justify-center',
                stat.color === 'success' && 'bg-success/10 text-success',
                stat.color === 'destructive' && 'bg-destructive/10 text-destructive',
                stat.color === 'warning' && 'bg-warning/10 text-warning',
                stat.color === 'accent' && 'bg-accent/10 text-accent',
                stat.color === 'primary' && 'bg-primary/10 text-primary'
              )}
            >
              {stat.icon}
            </div>
            {'change' in stat && stat.change !== undefined && (
              <span
                className={cn(
                  'text-xs font-medium px-2 py-1 rounded-full',
                  stat.change >= 0
                    ? 'bg-success/10 text-success'
                    : 'bg-destructive/10 text-destructive'
                )}
              >
                {stat.change >= 0 ? '+' : ''}
                {stat.change.toFixed(2)}%
              </span>
            )}
          </div>
          <p className="text-2xl font-bold mono">{stat.value}</p>
          <p className="text-sm text-muted-foreground mt-1">
            {stat.subValue || stat.label}
          </p>
        </motion.div>
      ))}
    </div>
  );
};

export default StatsGrid;
