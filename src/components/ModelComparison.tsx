import { motion } from 'framer-motion';
import { Brain, TreeDeciduous, Award, AlertTriangle } from 'lucide-react';
import { CompanyData } from '@/data/stockData';
import { cn } from '@/lib/utils';

interface ModelComparisonProps {
  company: CompanyData;
}

const ModelComparison = ({ company }: ModelComparisonProps) => {
  const { metrics } = company;
  const rfWins = metrics.recommendation === 'rf';

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
            LSTM vs Random Forest performance
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <ModelCard
          title="LSTM Neural Network"
          icon={<Brain className="w-5 h-5" />}
          mse={metrics.lstmMSE}
          r2={metrics.lstmR2}
          isRecommended={!rfWins}
        />
        <ModelCard
          title="Random Forest"
          icon={<TreeDeciduous className="w-5 h-5" />}
          mse={metrics.rfMSE}
          r2={metrics.rfR2}
          isRecommended={rfWins}
        />
      </div>

      <div
        className={cn(
          'flex items-center gap-3 p-4 rounded-xl',
          rfWins ? 'bg-primary/10 border border-primary/20' : 'bg-accent/10 border border-accent/20'
        )}
      >
        <Award className={cn('w-5 h-5', rfWins ? 'text-primary' : 'text-accent')} />
        <div>
          <p className="text-sm font-medium">
            Recommendation: {rfWins ? 'Random Forest' : 'LSTM'} performs better
          </p>
          <p className="text-xs text-muted-foreground">
            Based on lower Mean Squared Error for {company.name}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

interface ModelCardProps {
  title: string;
  icon: React.ReactNode;
  mse: number;
  r2: number;
  isRecommended: boolean;
}

const ModelCard = ({ title, icon, mse, r2, isRecommended }: ModelCardProps) => (
  <div
    className={cn(
      'p-4 rounded-xl border transition-all',
      isRecommended
        ? 'bg-primary/5 border-primary/30'
        : 'bg-muted/30 border-border'
    )}
  >
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <div
          className={cn(
            'w-8 h-8 rounded-lg flex items-center justify-center',
            isRecommended ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
          )}
        >
          {icon}
        </div>
        <span className="text-sm font-medium">{title}</span>
      </div>
      {isRecommended && (
        <span className="text-xs px-2 py-1 rounded-full bg-primary text-primary-foreground font-medium">
          Best
        </span>
      )}
    </div>

    <div className="space-y-3">
      <MetricRow
        label="MSE"
        value={mse.toFixed(4)}
        isGood={mse < 0.1}
        tooltip="Mean Squared Error - Lower is better"
      />
      <MetricRow
        label="R² Score"
        value={r2.toFixed(4)}
        isGood={r2 > 0}
        tooltip="Coefficient of determination - Closer to 1 is better"
      />
    </div>
  </div>
);

interface MetricRowProps {
  label: string;
  value: string;
  isGood: boolean;
  tooltip: string;
}

const MetricRow = ({ label, value, isGood, tooltip }: MetricRowProps) => (
  <div className="flex items-center justify-between group relative">
    <span className="text-xs text-muted-foreground">{label}</span>
    <span
      className={cn(
        'text-sm font-mono font-medium',
        isGood ? 'text-success' : 'text-warning'
      )}
    >
      {value}
    </span>
  </div>
);

export default ModelComparison;
