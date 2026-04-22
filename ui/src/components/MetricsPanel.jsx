import React from 'react';
import { motion } from 'framer-motion';

const clamp = (value) => Math.min(Math.max(value || 0, 0), 1);

const MetricCard = ({ label, value, tone, hint }) => (
  <div className="rounded-[1.6rem] border border-white/65 bg-white/80 p-4 shadow-[0_16px_30px_rgba(77,44,26,0.08)]">
    <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500">
      <span>{label}</span>
      <span>{clamp(value).toFixed(2)}</span>
    </div>
    <div className="mt-4 h-3 rounded-full bg-stone-200/80">
      <motion.div
        className="h-3 rounded-full"
        style={{ background: tone }}
        initial={{ width: 0 }}
        animate={{ width: `${clamp(value) * 100}%` }}
        transition={{ type: 'spring', stiffness: 130, damping: 24 }}
      />
    </div>
    <div className="mt-3 text-sm text-stone-600">{hint}</div>
  </div>
);

export const MetricsPanel = ({ metrics, trade, agents }) => {
  const reward = clamp(metrics?.reward);
  const grade = clamp(metrics?.grade);
  const drawdown = clamp(1 - (metrics?.drawdown || 0));
  const sharpe = clamp(metrics?.sharpe);
  const confidences = Object.values(agents || {}).map((agent) => agent?.confidence || 0);
  const deskSync = confidences.length
    ? confidences.reduce((sum, confidence) => sum + confidence, 0) / confidences.length
    : 0;

  return (
    <div className="rounded-[2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/92 p-6 shadow-[0_28px_50px_rgba(77,44,26,0.12)]">
      <div className="mb-5 flex items-end justify-between gap-6">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.34em] text-stone-500">Training Pulse</div>
          <h3 className="mt-2 text-2xl font-semibold text-stone-800">Desk metrics in motion</h3>
        </div>
        <div className="rounded-full bg-white/80 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500">
          {trade?.override ? 'PM override live' : 'Autonomy running'}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <MetricCard
          label="Reward"
          value={reward}
          tone="linear-gradient(90deg, #4cc9f0, #90e0ef)"
          hint="Immediate step reward from the latest action."
        />
        <MetricCard
          label="Grade"
          value={grade}
          tone="linear-gradient(90deg, #52b788, #95d5b2)"
          hint="Overall normalized performance quality."
        />
        <MetricCard
          label="Drawdown"
          value={drawdown}
          tone="linear-gradient(90deg, #f4a261, #f6bd60)"
          hint="Higher means the desk is protecting capital."
        />
        <MetricCard
          label="Sharpe"
          value={sharpe}
          tone="linear-gradient(90deg, #7f8ce6, #9ec5fe)"
          hint="Risk-adjusted return trend for the episode."
        />
        <MetricCard
          label="Desk Sync"
          value={deskSync}
          tone="linear-gradient(90deg, #f28c6f, #f6bd60)"
          hint="Average conviction across all five agents."
        />
      </div>
    </div>
  );
};
