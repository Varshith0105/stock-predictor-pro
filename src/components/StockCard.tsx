import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { CompanyData } from '@/data/stockData';
import { cn } from '@/lib/utils';

interface StockCardProps {
  company: CompanyData;
  isSelected: boolean;
  onClick: () => void;
  index: number;
}

const StockCard = ({ company, isSelected, onClick, index }: StockCardProps) => {
  const isPositive = company.changePercent >= 0;

  return (
    <motion.button
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={cn(
        'w-full p-4 rounded-xl border transition-all text-left',
        isSelected
          ? 'bg-primary/10 border-primary/50 shadow-lg shadow-primary/10'
          : 'bg-card border-border hover:border-primary/30 hover:bg-card/80'
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{company.logo}</span>
          <div>
            <p className="font-semibold text-sm">{company.symbol}</p>
            <p className="text-xs text-muted-foreground truncate max-w-[100px]">
              {company.name}
            </p>
          </div>
        </div>
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
            isPositive
              ? 'bg-success/10 text-success'
              : 'bg-destructive/10 text-destructive'
          )}
        >
          {isPositive ? (
            <TrendingUp className="w-3 h-3" />
          ) : (
            <TrendingDown className="w-3 h-3" />
          )}
          {isPositive ? '+' : ''}
          {company.changePercent.toFixed(2)}%
        </div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <p className="text-lg font-bold mono">${company.currentPrice.toFixed(2)}</p>
          <p
            className={cn(
              'text-xs mono',
              isPositive ? 'text-success' : 'text-destructive'
            )}
          >
            {isPositive ? '+' : ''}${company.change.toFixed(2)}
          </p>
        </div>
        <MiniChart isPositive={isPositive} data={company.data.slice(-7)} />
      </div>
    </motion.button>
  );
};

interface MiniChartProps {
  isPositive: boolean;
  data: { actual: number }[];
}

const MiniChart = ({ isPositive, data }: MiniChartProps) => {
  const values = data.map((d) => d.actual);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 60;
      const y = 20 - ((v - min) / range) * 20;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width="60" height="24" className="overflow-visible">
      <polyline
        points={points}
        fill="none"
        stroke={isPositive ? 'hsl(var(--success))' : 'hsl(var(--destructive))'}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default StockCard;
