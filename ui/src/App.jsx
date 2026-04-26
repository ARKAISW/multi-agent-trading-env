import React from 'react';
import { Play, SkipForward, Square } from 'lucide-react';

import { BalancePanel } from './components/BalancePanel';
import { MarketPanel } from './components/MarketPanel';
import { MetricsPanel } from './components/MetricsPanel';
import { OfficeScene } from './components/OfficeScene';
import { useSimulationStore } from './store';

function App() {
  const {
    simState,
    portfolioHistory,
    priceHistory,
    lastPortfolioDelta,
    lastPriceDelta,
    lastError,
    toggleSimulation,
    fetchState,
    stepSimulation,
  } = useSimulationStore();

  React.useEffect(() => {
    fetchState();
    const interval = window.setInterval(() => {
      fetchState();
    }, 600);
    return () => window.clearInterval(interval);
  }, [fetchState]);

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f9eadb_0%,#f6ddc6_42%,#ecd0bb_100%)] px-6 py-8 text-stone-800 md:px-8">
      <div className="mx-auto max-w-[1500px]">
        <div className="mb-8 flex flex-col gap-5 rounded-[2.2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/78 px-6 py-6 shadow-[0_24px_55px_rgba(77,44,26,0.12)] backdrop-blur md:flex-row md:items-end md:justify-between">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.32em] text-stone-500">
              <span className={`h-2.5 w-2.5 rounded-full ${simState.is_running ? 'bg-emerald-500' : 'bg-stone-400'}`} />
              Wild Card Demo
            </div>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight text-stone-900 md:text-5xl">
              Indie quant floor with live agent choreography
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-stone-600 md:text-base">
              The scene updates off the running trading environment: signals travel desk to desk, the trader fires orders, and the balance reacts to the trades in real time.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {lastError ? (
              <div className="rounded-full border border-rose-200 bg-rose-50 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-rose-600 shadow-sm">
                Demo error: {lastError}
              </div>
            ) : null}
            <div className="rounded-full bg-white/85 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500 shadow-sm">
              Step {simState.current_step}
            </div>
            <div className="rounded-full bg-white/85 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500 shadow-sm">
              {simState.engine?.mode}
            </div>
            <button
              onClick={() => stepSimulation()}
              className="inline-flex items-center gap-2 rounded-full border border-[#7d5a4f]/15 bg-white px-5 py-3 text-sm font-semibold text-stone-700 shadow-sm transition hover:bg-stone-50"
            >
              <SkipForward size={16} />
              Step
            </button>
            <button
              onClick={() => toggleSimulation(simState.is_running)}
              className={`inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-semibold shadow-sm transition ${
                simState.is_running
                  ? 'bg-rose-500 text-white hover:bg-rose-600'
                  : 'bg-emerald-500 text-white hover:bg-emerald-600'
              }`}
            >
              {simState.is_running ? <Square size={16} /> : <Play size={16} />}
              {simState.is_running ? 'Stop Demo' : 'Run Demo'}
            </button>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <OfficeScene
            agents={simState.agents}
            flow={simState.flow}
            trade={simState.trade}
            currentStep={simState.current_step}
            isRunning={simState.is_running}
            engine={simState.engine}
          />

          <div className="flex flex-col gap-6">
            <BalancePanel
              portfolio={simState.portfolio}
              history={portfolioHistory}
              lastDelta={lastPortfolioDelta}
              trade={simState.trade}
            />
            <MarketPanel
              chart={simState.chart}
              history={priceHistory}
              trade={simState.trade}
              engine={simState.engine}
              lastPriceDelta={lastPriceDelta}
            />
          </div>
        </div>

        <div className="mt-6">
          <MetricsPanel metrics={simState.metrics} trade={simState.trade} agents={simState.agents} />
        </div>
      </div>
    </div>
  );
}

export default App;
