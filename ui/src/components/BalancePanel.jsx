import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const buildPath = (points, width, height) => {
  if (!points.length) {
    return '';
  }

  const values = points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - ((point.value - min) / range) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
};

export const BalancePanel = ({ portfolio, history, lastDelta, trade }) => {
  const { value = 100000, cash = 100000, positions = {} } = portfolio || {};
  const [displayValue, setDisplayValue] = useState(value);
  const direction = lastDelta > 0 ? 'up' : lastDelta < 0 ? 'down' : 'flat';

  useEffect(() => {
    const start = displayValue;
    const end = value;
    let frameId;
    let startTime;

    const tick = (timestamp) => {
      if (!startTime) {
        startTime = timestamp;
      }
      const progress = Math.min((timestamp - startTime) / 500, 1);
      setDisplayValue(start + (end - start) * progress);
      if (progress < 1) {
        frameId = window.requestAnimationFrame(tick);
      }
    };

    frameId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frameId);
  }, [value]);

  const path = useMemo(() => buildPath(history || [], 260, 86), [history]);
  const invested = Math.max(value - cash, 0);
  const exposure = value > 0 ? (invested / value) * 100 : 0;

  return (
    <div className="relative overflow-hidden rounded-[2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/92 p-6 shadow-[0_28px_50px_rgba(77,44,26,0.12)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(123,192,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(242,140,111,0.12),transparent_30%)]" />

      <div className="relative z-10">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.34em] text-stone-500">Balance Tape</div>
            <h3 className="mt-2 text-2xl font-semibold text-stone-800">Portfolio Value</h3>
          </div>
          <div
            className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.26em] ${
              direction === 'up'
                ? 'bg-emerald-100 text-emerald-700'
                : direction === 'down'
                  ? 'bg-rose-100 text-rose-700'
                  : 'bg-stone-200 text-stone-600'
            }`}
          >
            {trade?.side || 'HOLD'}
          </div>
        </div>

        <motion.div
          key={value}
          animate={direction === 'flat' ? { scale: 1 } : { scale: [1, 1.04, 1] }}
          transition={{ duration: 0.45 }}
          className={`mt-5 text-5xl font-semibold tracking-tight tabular-nums ${
            direction === 'up'
              ? 'text-emerald-600'
              : direction === 'down'
                ? 'text-rose-600'
                : 'text-stone-800'
          }`}
        >
          ${displayValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </motion.div>

        <div className="mt-3 flex items-center gap-3 text-sm">
          <span className="rounded-full bg-white/85 px-3 py-1 text-stone-600 shadow-sm">
            Cash ${cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="rounded-full bg-white/85 px-3 py-1 text-stone-600 shadow-sm">
            Exposure {exposure.toFixed(1)}%
          </span>
        </div>

        <div className="mt-6 rounded-[1.6rem] border border-white/70 bg-white/65 p-4">
          <div className="mb-3 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.24em] text-stone-500">
            <span>Live equity trail</span>
            <span>
              {lastDelta >= 0 ? '+' : ''}
              {lastDelta.toFixed(2)}
            </span>
          </div>

          <svg viewBox="0 0 260 86" className="h-[86px] w-full overflow-visible">
            <path d={path} fill="none" stroke="#f0d8c7" strokeWidth="10" strokeLinecap="round" />
            <motion.path
              d={path}
              fill="none"
              stroke={direction === 'down' ? '#d95d5d' : '#52b788'}
              strokeWidth="5"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.55 }}
            />
          </svg>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3">
          <div className="rounded-[1.4rem] bg-white/70 px-4 py-4">
            <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Invested</div>
            <div className="mt-2 text-lg font-semibold text-stone-800">
              ${invested.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="rounded-[1.4rem] bg-white/70 px-4 py-4">
            <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Open books</div>
            <div className="mt-2 text-lg font-semibold text-stone-800">{Object.keys(positions || {}).length}</div>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {direction !== 'flat' && (
          <>
            {[0, 1, 2].map((index) => (
              <motion.div
                // eslint-disable-next-line react/no-array-index-key
                key={`${trade?.pulse || 0}-${direction}-${index}`}
                className={`pointer-events-none absolute bottom-10 left-[12%] h-4 w-4 rounded-full ${
                  direction === 'up' ? 'bg-emerald-400/60' : 'bg-rose-400/55'
                }`}
                initial={{ y: 0, x: index * 24, opacity: 0.8, scale: 0.8 }}
                animate={{ y: direction === 'up' ? -88 : 52, x: index * 56 + 24, opacity: 0, scale: 1.45 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.15, delay: index * 0.08, ease: 'easeOut' }}
              />
            ))}
          </>
        )}
      </AnimatePresence>
    </div>
  );
};
