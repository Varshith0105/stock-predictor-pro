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
  ReferenceLine,
} from 'recharts';
import { CompanyData, StockDataPoint } from '@/data/stockData';

interface PriceChartProps {
  company: CompanyData;
  liveData?: any[];
  showPredictions?: boolean;
}

const PriceChart = ({ company, liveData, showPredictions = true }: PriceChartProps) => {
  // Use live data if available, otherwise use static data
  const chartData = liveData && liveData.length > 0
    ? liveData.map((point) => ({
        date: new Date(point.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        actual: point.close,
        // Generate mock predictions based on historical trend
        predicted: point.predictedClose,
        isPrediction: point.isPrediction,
      }))
    : company.data.map((point) => ({
        date: new Date(point.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        actual: point.actual,
        predicted: point.predicted,
      }));

  // Find the index where predictions start
  const predictionStartIndex = chartData.findIndex((d: any) => d.isPrediction);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-card border border-border rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">
            {liveData && liveData.length > 0 ? 'Live Price Data' : 'Price History'} & Predictions
          </h3>
          <p className="text-sm text-muted-foreground">
            {liveData && liveData.length > 0 
              ? `Showing ${liveData.filter((d: any) => !d.isPrediction).length} historical + ${liveData.filter((d: any) => d.isPrediction).length} predicted days`
              : 'Actual vs AI-predicted prices'
            }
          </p>
        </div>
        <div className="flex items-center gap-4">
          <ChartLegendItem color="hsl(var(--chart-line))" label="Actual" />
          {showPredictions && (
            <ChartLegendItem color="hsl(var(--chart-predicted))" label="Predicted" dashed />
          )}
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
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={['dataMin - 5', 'dataMax + 5']}
            />
            <Tooltip content={<CustomTooltip />} />
            {predictionStartIndex > 0 && (
              <ReferenceLine
                x={chartData[predictionStartIndex]?.date}
                stroke="hsl(var(--warning))"
                strokeDasharray="5 5"
                label={{ value: 'Predictions', fill: 'hsl(var(--warning))', fontSize: 10 }}
              />
            )}
            <Area
              type="monotone"
              dataKey="actual"
              stroke="hsl(var(--chart-line))"
              strokeWidth={2}
              fill="url(#actualGradient)"
              dot={false}
              activeDot={{ r: 6, stroke: 'hsl(var(--chart-line))', strokeWidth: 2, fill: 'hsl(var(--background))' }}
              connectNulls
            />
            {showPredictions && (
              <Area
                type="monotone"
                dataKey="predicted"
                stroke="hsl(var(--chart-predicted))"
                strokeWidth={2}
                strokeDasharray="5 5"
                fill="url(#predictedGradient)"
                dot={false}
                activeDot={{ r: 6, stroke: 'hsl(var(--chart-predicted))', strokeWidth: 2, fill: 'hsl(var(--background))' }}
                connectNulls
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};

interface ChartLegendItemProps {
  color: string;
  label: string;
  dashed?: boolean;
}

const ChartLegendItem = ({ color, label, dashed }: ChartLegendItemProps) => (
  <div className="flex items-center gap-2">
    <div
      className="w-6 h-0.5"
      style={{ 
        backgroundColor: color,
        backgroundImage: dashed ? 'repeating-linear-gradient(90deg, transparent, transparent 2px, currentColor 2px, currentColor 4px)' : undefined
      }}
    />
    <span className="text-sm text-muted-foreground">{label}</span>
  </div>
);

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const isPrediction = payload[0]?.payload?.isPrediction;
    return (
      <div className="bg-popover border border-border rounded-lg p-3 shadow-xl">
        <p className="text-sm font-medium mb-2">
          {label}
          {isPrediction && (
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-warning/20 text-warning">
              Prediction
            </span>
          )}
        </p>
        {payload.map((entry: any, index: number) => (
          entry.value && (
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
          )
        ))}
      </div>
    );
  }
  return null;
};

export default PriceChart;
