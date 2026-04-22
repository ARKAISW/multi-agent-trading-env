import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

const buildPath = (points, width, height) => {
  if (!points.length) {
    return '';
  }

  const prices = points.map((point) => point.price);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;

  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - ((point.price - min) / range) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
};

export const MarketPanel = ({ chart, history, trade, engine, lastPriceDelta }) => {
  const path = useMemo(() => buildPath(history || [], 250, 90), [history]);
  const priceTone = lastPriceDelta < 0 ? '#d95d5d' : '#52b788';

  return (
    <div className="rounded-[2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/92 p-6 shadow-[0_28px_50px_rgba(77,44,26,0.12)]">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.34em] text-stone-500">Live Execution</div>
          <h3 className="mt-2 text-2xl font-semibold text-stone-800">Market tape</h3>
        </div>
        <div className="rounded-full bg-white/80 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-stone-500">
          {engine?.policy_active ? 'SLM live' : 'Fallback'}
        </div>
      </div>

      <div className="mt-5 rounded-[1.6rem] border border-white/65 bg-white/75 p-4">
        <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500">
          <span>Price</span>
          <span>
            {lastPriceDelta >= 0 ? '+' : ''}
            {lastPriceDelta.toFixed(2)}
          </span>
        </div>

        <div className="mt-3 text-4xl font-semibold tracking-tight text-stone-800">
          ${chart?.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>

        <svg viewBox="0 0 250 90" className="mt-4 h-[90px] w-full overflow-visible">
          <path d={path} fill="none" stroke="#edd5c4" strokeWidth="10" strokeLinecap="round" />
          <motion.path
            d={path}
            fill="none"
            stroke={priceTone}
            strokeWidth="5"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.55 }}
          />
        </svg>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-[1.4rem] bg-white/75 px-4 py-4">
          <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Last side</div>
          <div className="mt-2 text-lg font-semibold text-stone-800">{trade?.side || 'HOLD'}</div>
        </div>
        <div className="rounded-[1.4rem] bg-white/75 px-4 py-4">
          <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Notional</div>
          <div className="mt-2 text-lg font-semibold text-stone-800">
            ${trade?.notional?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
        </div>
      </div>

      <div className="mt-5 rounded-[1.6rem] border border-white/65 bg-white/80 px-4 py-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-stone-500">Decision reason</div>
        <div className="mt-2 text-sm leading-6 text-stone-600">{trade?.reason || 'Waiting for the desk to move.'}</div>
      </div>
    </div>
  );
};
