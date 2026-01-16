import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Legend,
} from 'recharts';
import { CompanyData } from '@/data/stockData';

interface PriceChartProps {
  company: CompanyData;
}

const PriceChart = ({ company }: PriceChartProps) => {
  const chartData = company.data.map((point) => ({
    date: new Date(point.date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
    actual: point.actual,
    predicted: point.predicted,
    rfPredicted: point.rfPredicted,
    lstmPredicted: point.lstmPredicted,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-card border border-border rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">Price History & Predictions</h3>
          <p className="text-sm text-muted-foreground">
            Actual vs AI-predicted prices
          </p>
        </div>
        <div className="flex items-center gap-4">
          <ChartLegendItem color="hsl(var(--chart-line))" label="Actual" />
          <ChartLegendItem color="hsl(var(--chart-predicted))" label="Predicted" />
        </div>
      </div>

      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--chart-line))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--chart-line))" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--chart-predicted))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--chart-predicted))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--chart-grid))"
              vertical={false}
            />
            <XAxis
              dataKey="date"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${value}`}
              domain={['dataMin - 5', 'dataMax + 5']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="actual"
              stroke="hsl(var(--chart-line))"
              strokeWidth={2}
              fill="url(#actualGradient)"
              dot={false}
              activeDot={{ r: 6, stroke: 'hsl(var(--chart-line))', strokeWidth: 2, fill: 'hsl(var(--background))' }}
            />
            <Area
              type="monotone"
              dataKey="predicted"
              stroke="hsl(var(--chart-predicted))"
              strokeWidth={2}
              strokeDasharray="5 5"
              fill="url(#predictedGradient)"
              dot={false}
              activeDot={{ r: 6, stroke: 'hsl(var(--chart-predicted))', strokeWidth: 2, fill: 'hsl(var(--background))' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};

interface ChartLegendItemProps {
  color: string;
  label: string;
}

const ChartLegendItem = ({ color, label }: ChartLegendItemProps) => (
  <div className="flex items-center gap-2">
    <div
      className="w-3 h-3 rounded-full"
      style={{ backgroundColor: color }}
    />
    <span className="text-sm text-muted-foreground">{label}</span>
  </div>
);

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-popover border border-border rounded-lg p-3 shadow-xl">
        <p className="text-sm font-medium mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 text-sm">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-muted-foreground capitalize">{entry.name}:</span>
            <span className="font-mono font-medium">
              ${entry.value?.toFixed(2) || 'N/A'}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default PriceChart;
