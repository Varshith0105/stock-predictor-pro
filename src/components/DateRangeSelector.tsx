import { useState } from 'react';
import { motion } from 'framer-motion';
import { Calendar, TrendingUp, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Calendar as CalendarComponent } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { format, subDays } from 'date-fns';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';

interface DateRangeSelectorProps {
  symbol: string;
  onDataFetched: (data: any[]) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const DateRangeSelector = ({ symbol, onDataFetched, isLoading, setIsLoading }: DateRangeSelectorProps) => {
  const [startDate, setStartDate] = useState<Date>(subDays(new Date(), 30));
  const [endDate, setEndDate] = useState<Date>(new Date());
  const [predictionDays, setPredictionDays] = useState<number>(7);
  const { toast } = useToast();

  const handleFetchData = async () => {
    if (!startDate || !endDate) {
      toast({
        title: 'Invalid Date Range',
        description: 'Please select both start and end dates.',
        variant: 'destructive',
      });
      return;
    }

    if (startDate >= endDate) {
      toast({
        title: 'Invalid Date Range',
        description: 'Start date must be before end date.',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);

    try {
      const { data, error } = await supabase.functions.invoke('fetch-stock-data', {
        body: {
          symbol,
          startDate: format(startDate, 'yyyy-MM-dd'),
          endDate: format(endDate, 'yyyy-MM-dd'),
        },
      });

      if (error) {
        throw error;
      }

      if (data.rateLimited) {
        toast({
          title: 'Rate Limited',
          description: data.message,
          variant: 'destructive',
        });
        return;
      }

      if (data.data && data.data.length > 0) {
        onDataFetched(data.data);
        toast({
          title: 'Data Fetched',
          description: `Loaded ${data.data.length} days of stock data for ${symbol}.`,
        });
      } else {
        toast({
          title: 'No Data',
          description: 'No stock data available for the selected date range.',
          variant: 'destructive',
        });
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch stock data';
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const presetRanges = [
    { label: '7D', days: 7 },
    { label: '1M', days: 30 },
    { label: '3M', days: 90 },
    { label: '6M', days: 180 },
    { label: '1Y', days: 365 },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border border-border rounded-2xl p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <Calendar className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Date Range & Prediction</h3>
          <p className="text-sm text-muted-foreground">
            Fetch real-time data and set prediction window
          </p>
        </div>
      </div>

      {/* Quick Presets */}
      <div className="flex flex-wrap gap-2 mb-4">
        {presetRanges.map((preset) => (
          <Button
            key={preset.label}
            variant="outline"
            size="sm"
            onClick={() => {
              setStartDate(subDays(new Date(), preset.days));
              setEndDate(new Date());
            }}
            className="text-xs"
          >
            {preset.label}
          </Button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Start Date */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-muted-foreground">Start Date</label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  'w-full justify-start text-left font-normal',
                  !startDate && 'text-muted-foreground'
                )}
              >
                <Calendar className="mr-2 h-4 w-4" />
                {startDate ? format(startDate, 'MMM dd, yyyy') : 'Pick a date'}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <CalendarComponent
                mode="single"
                selected={startDate}
                onSelect={(date) => date && setStartDate(date)}
                initialFocus
                className="p-3 pointer-events-auto"
              />
            </PopoverContent>
          </Popover>
        </div>

        {/* End Date */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-muted-foreground">End Date</label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  'w-full justify-start text-left font-normal',
                  !endDate && 'text-muted-foreground'
                )}
              >
                <Calendar className="mr-2 h-4 w-4" />
                {endDate ? format(endDate, 'MMM dd, yyyy') : 'Pick a date'}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <CalendarComponent
                mode="single"
                selected={endDate}
                onSelect={(date) => date && setEndDate(date)}
                initialFocus
                className="p-3 pointer-events-auto"
              />
            </PopoverContent>
          </Popover>
        </div>

        {/* Prediction Days */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-muted-foreground">Predict Next</label>
          <div className="flex gap-2">
            {[3, 7, 14, 30].map((days) => (
              <Button
                key={days}
                variant={predictionDays === days ? 'default' : 'outline'}
                size="sm"
                onClick={() => setPredictionDays(days)}
                className="flex-1"
              >
                {days}D
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Action Button */}
      <Button
        onClick={handleFetchData}
        disabled={isLoading}
        className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90"
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Fetching Data...
          </>
        ) : (
          <>
            <RefreshCw className="mr-2 h-4 w-4" />
            Fetch Real-Time Data & Predict
          </>
        )}
      </Button>

      <p className="text-xs text-muted-foreground text-center mt-3">
        Data provided by Alpha Vantage API • Predictions based on ML models
      </p>
    </motion.div>
  );
};

export default DateRangeSelector;
