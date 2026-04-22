import { create } from 'zustand';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';
const HISTORY_LIMIT = 48;

const initialState = {
  is_running: false,
  current_step: 0,
  agents: {},
  portfolio: { value: 100000.0, cash: 100000.0, positions: {} },
  metrics: { reward: 0.0, grade: 0.0, drawdown: 0.0, sharpe: 0.0 },
  chart: { price: 50000.0, trade: null, price_change: 0.0 },
  trade: {
    pulse: 0,
    side: 'HOLD',
    size: 0.0,
    price: 50000.0,
    sl: 0.0,
    tp: 0.0,
    portfolio_delta: 0.0,
    notional: 0.0,
    reason: 'Waiting for the first coordinated decision.',
    override: false,
  },
  flow: [],
  engine: {
    name: 'Desk Policy',
    mode: 'Rule Fallback',
    policy_active: false,
    note: 'Enable USE_LOCAL_POLICY=true to run the fine-tuned local model.',
  },
};

const withPoint = (history, point) => {
  if (!point) {
    return history;
  }
  return [...history, point].slice(-HISTORY_LIMIT);
};

export const useSimulationStore = create((set, get) => ({
  simState: initialState,
  portfolioHistory: [{ step: 0, value: initialState.portfolio.value }],
  priceHistory: [{ step: 0, price: initialState.chart.price }],
  lastPortfolioDelta: 0,
  lastPriceDelta: 0,
  fetchState: async () => {
    try {
      const res = await fetch(`${API_BASE}/state`);
      const data = await res.json();

      set((state) => {
        const stepChanged = data.current_step !== state.simState.current_step;
        return {
          simState: data,
          lastPortfolioDelta: data.portfolio.value - state.simState.portfolio.value,
          lastPriceDelta: data.chart.price - state.simState.chart.price,
          portfolioHistory: stepChanged
            ? withPoint(state.portfolioHistory, {
                step: data.current_step,
                value: data.portfolio.value,
              })
            : state.portfolioHistory,
          priceHistory: stepChanged
            ? withPoint(state.priceHistory, {
                step: data.current_step,
                price: data.chart.price,
              })
            : state.priceHistory,
        };
      });
    } catch (error) {
      console.error('Error fetching state:', error);
    }
  },
  toggleSimulation: async (isRunning) => {
    const endpoint = isRunning ? '/stop' : '/start';
    try {
      await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
      get().fetchState();
    } catch (error) {
      console.error(`Error toggling simulation to ${!isRunning}:`, error);
    }
  },
  stepSimulation: async () => {
    try {
      await fetch(`${API_BASE}/step`, { method: 'POST' });
      get().fetchState();
    } catch (error) {
      console.error('Error stepping simulation:', error);
    }
  },
}));
