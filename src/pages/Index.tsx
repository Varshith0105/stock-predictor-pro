import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Header from '@/components/Header';
import MarketTicker from '@/components/MarketTicker';
import StockCard from '@/components/StockCard';
import PriceChart from '@/components/PriceChart';
import StatsGrid from '@/components/StatsGrid';
import DateRangeSelector from '@/components/DateRangeSelector';
import { companies, CompanyData } from '@/data/stockData';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { generatePredictions, combineDataWithPredictions, HistoricalDataPoint } from '@/utils/predictions';

const modelOptions = [
  { value: 'lstm', label: 'LSTM Neural Network' },
  { value: 'randomForest', label: 'Random Forest' },
  { value: 'arima', label: 'ARIMA' },
  { value: 'linearRegression', label: 'Linear Regression' },
];

const Index = () => {
  const [selectedCompany, setSelectedCompany] = useState<CompanyData>(companies[0]);
  const [liveData, setLiveData] = useState<any[] | null>(null);
  const [rawData, setRawData] = useState<HistoricalDataPoint[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('lstm');

  const handleDataFetched = (data: HistoricalDataPoint[]) => {
    setRawData(data);
    // Generate predictions using selected model
    const predictions = generatePredictions(data, 7, selectedModel);
    const combinedData = combineDataWithPredictions(data, predictions);
    setLiveData(combinedData);
  };

  // Regenerate predictions when model changes
  useEffect(() => {
    if (rawData && rawData.length > 0) {
      const predictions = generatePredictions(rawData, 7, selectedModel);
      const combinedData = combineDataWithPredictions(rawData, predictions);
      setLiveData(combinedData);
    }
  }, [selectedModel, rawData]);

  const handleCompanySelect = (company: CompanyData) => {
    setSelectedCompany(company);
    setLiveData(null);
    setRawData(null);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <MarketTicker />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-gradient">AI-Powered</span> Stock Predictions
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Fetch live market data and predict future prices using 
            advanced machine learning models
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-[320px_1fr] gap-8">
          {/* Stock List Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-card/30 border border-border rounded-2xl p-4"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">Select Stock</h3>
              <span className="text-xs text-muted-foreground">
                {companies.length} companies
              </span>
            </div>
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-3">
                {companies.map((company, index) => (
                  <StockCard
                    key={company.id}
                    company={company}
                    isSelected={selectedCompany.id === company.id}
                    onClick={() => handleCompanySelect(company)}
                    index={index}
                  />
                ))}
              </div>
            </ScrollArea>
          </motion.div>

          {/* Main Content */}
          <div className="space-y-6">
            {/* Selected Stock Header */}
            <motion.div
              key={selectedCompany.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
            >
              <div className="flex items-center gap-4">
                <span className="text-5xl">{selectedCompany.logo}</span>
                <div>
                  <h2 className="text-3xl font-bold">{selectedCompany.name}</h2>
                  <p className="text-muted-foreground">
                    {selectedCompany.symbol} • {liveData ? `${modelOptions.find(m => m.value === selectedModel)?.label} Mode` : 'Select dates to fetch data'}
                  </p>
                </div>
              </div>
              
              {/* Model Selector */}
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">Prediction Model:</span>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-[200px] bg-card border-border">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border z-50">
                    {modelOptions.map((model) => (
                      <SelectItem 
                        key={model.value} 
                        value={model.value}
                        className="cursor-pointer"
                      >
                        {model.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </motion.div>

            {/* Date Range Selector */}
            <DateRangeSelector 
              symbol={selectedCompany.symbol}
              onDataFetched={handleDataFetched}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />

            {/* Stats Grid */}
            <StatsGrid company={selectedCompany} selectedModel={selectedModel} />

            {/* Price Chart */}
            <PriceChart 
              company={selectedCompany} 
              liveData={liveData || undefined}
              showPredictions={true}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            StockAI - AI-Powered Stock Predictions • Real-time data from Alpha Vantage
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
