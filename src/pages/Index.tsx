import { useState } from 'react';
import { motion } from 'framer-motion';
import Header from '@/components/Header';
import MarketTicker from '@/components/MarketTicker';
import StockCard from '@/components/StockCard';
import PriceChart from '@/components/PriceChart';
import ModelComparison from '@/components/ModelComparison';
import StatsGrid from '@/components/StatsGrid';
import { companies, CompanyData } from '@/data/stockData';
import { ScrollArea } from '@/components/ui/scroll-area';

const Index = () => {
  const [selectedCompany, setSelectedCompany] = useState<CompanyData>(companies[0]);

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
            Analyzing 15 major companies using LSTM Neural Networks and Random Forest 
            algorithms for accurate price forecasting
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
                    onClick={() => setSelectedCompany(company)}
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
              className="flex items-center gap-4"
            >
              <span className="text-5xl">{selectedCompany.logo}</span>
              <div>
                <h2 className="text-3xl font-bold">{selectedCompany.name}</h2>
                <p className="text-muted-foreground">
                  {selectedCompany.symbol} • Prediction Analysis
                </p>
              </div>
            </motion.div>

            {/* Stats Grid */}
            <StatsGrid company={selectedCompany} />

            {/* Price Chart */}
            <PriceChart company={selectedCompany} />

            {/* Model Comparison */}
            <ModelComparison company={selectedCompany} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            StockAI - Powered by Machine Learning • Data from Oct 2023 - Oct 2024
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
