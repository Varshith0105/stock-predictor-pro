import { motion } from 'framer-motion';
import { companies } from '@/data/stockData';
import { cn } from '@/lib/utils';

const MarketTicker = () => {
  const tickerItems = [...companies, ...companies]; // Duplicate for seamless loop

  return (
    <div className="bg-card/50 border-b border-border overflow-hidden py-2">
      <div className="flex animate-ticker">
        {tickerItems.map((company, index) => (
          <div
            key={`${company.id}-${index}`}
            className="flex items-center gap-3 px-6 whitespace-nowrap"
          >
            <span className="text-lg">{company.logo}</span>
            <span className="font-medium text-sm">{company.symbol}</span>
            <span className="mono text-sm">${company.currentPrice.toFixed(2)}</span>
            <span
              className={cn(
                'text-xs font-medium',
                company.changePercent >= 0 ? 'text-success' : 'text-destructive'
              )}
            >
              {company.changePercent >= 0 ? '+' : ''}
              {company.changePercent.toFixed(2)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MarketTicker;
