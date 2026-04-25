import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import officeBg from '../assets/office_bg.png';

/* ──────────────────────── theme & layout ──────────────────────── */

const AGENT_THEMES = {
  Researcher: {
    accent: '#f28c6f',
    bubble: 'from-rose-100/95 via-orange-50/95 to-amber-50/90',
    icon: '🔍',
    label: 'Signal Lab',
  },
  'Risk Manager': {
    accent: '#5f8b7e',
    bubble: 'from-emerald-100/95 via-teal-50/95 to-lime-50/90',
    icon: '🛡️',
    label: 'Guard Rail',
  },
  'Portfolio Manager': {
    accent: '#e2a63b',
    bubble: 'from-amber-100/95 via-yellow-50/95 to-orange-50/90',
    icon: '💼',
    label: 'Capital Desk',
  },
  'Fundamental Analyst': {
    accent: '#7f8ce6',
    bubble: 'from-sky-100/95 via-indigo-50/95 to-cyan-50/90',
    icon: '📊',
    label: 'Macro Wire',
  },
  Trader: {
    accent: '#e36b5f',
    bubble: 'from-red-100/95 via-orange-50/95 to-yellow-50/90',
    icon: '⚔️',
    label: 'Execution',
  },
};

/* Desk positions (% of scene) — tuned to sit on the pixel-art desks */
const STATIONS = {
  Researcher:           { left: '4%',  top: '25%',  anchor: [100, 140] },
  'Risk Manager':       { left: '26%', top: '22%',  anchor: [420, 120] },
  'Portfolio Manager':  { left: '55%', top: '25%',  anchor: [730, 140] },
  'Fundamental Analyst':{ left: '10%', top: '56%', anchor: [230, 430] },
  Trader:               { left: '48%', top: '56%', anchor: [700, 430] },
};

const MARKET_ANCHOR = [950, 280];

const LINK_TONES = {
  signal: '#f28c6f',
  research: '#f0a95b',
  macro: '#7f8ce6',
  risk: '#5f8b7e',
  approval: '#e2a63b',
  buy: '#52b788',
  sell: '#d95d5d',
  hold: '#8d99ae',
};

const trimMessage = (m = '') => (m.length <= 80 ? m : `${m.slice(0, 77)}…`);

/* ──────────────────────── data-link SVG ──────────────────────── */

const DataLink = ({ flow, isRunning }) => {
  const start = flow.from === 'Market' ? MARKET_ANCHOR : STATIONS[flow.from]?.anchor;
  const end   = flow.to   === 'Market' ? MARKET_ANCHOR : STATIONS[flow.to]?.anchor;
  if (!start || !end) return null;
  const color = LINK_TONES[flow.tone] || '#8d99ae';

  return (
    <>
      <line
        x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]}
        stroke={color} strokeOpacity={flow.active ? 0.85 : 0.25}
        strokeWidth="5" strokeLinecap="round"
        strokeDasharray={flow.active ? '8 14' : '5 15'}
      />
      {flow.active && (
        <motion.circle
          r="7" fill={color}
          cx={start[0]} cy={start[1]}
          animate={{ cx: [start[0], end[0]], cy: [start[1], end[1]], opacity: [0, 1, 1, 0] }}
          transition={{
            duration: isRunning ? Math.max(0.7, 1.6 - flow.strength) : 0,
            repeat: isRunning ? Infinity : 0, ease: 'linear',
          }}
        />
      )}
    </>
  );
};

/* ──────────────────────── agent speech bubble ──────────────────────── */

const AgentBubble = ({ name, agent, isRunning }) => {
  const theme   = AGENT_THEMES[name];
  const station = STATIONS[name];
  const conf    = agent?.confidence || 0;
  const active  = agent?.status === 'active';

  return (
    <motion.div
      className="absolute z-10 w-[230px]"
      style={{ left: station.left, top: station.top }}
      animate={isRunning ? { y: [0, -3, 0] } : { y: 0 }}
      transition={{ duration: 3, repeat: isRunning ? Infinity : 0, ease: 'easeInOut', delay: conf * 2 }}
    >
      {/* speech bubble */}
      <AnimatePresence>
        {agent?.message && (
          <motion.div
            key={`${name}-${agent.message}`}
            initial={{ opacity: 0, y: 6, scale: 0.93 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className={`rounded-2xl border border-white/80 bg-gradient-to-br ${theme.bubble} px-4 py-3 text-[11px] leading-4 text-stone-700 shadow-lg backdrop-blur-sm`}
          >
            <div className="mb-1.5 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.28em] text-stone-500">
                <span className="text-sm">{theme.icon}</span>
                {theme.label}
              </span>
              <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${
                active ? 'bg-emerald-500/20 text-emerald-700' : 'bg-stone-200 text-stone-500'
              }`}>
                {active ? 'LIVE' : 'IDLE'}
              </span>
            </div>
            <div className="font-medium">{trimMessage(agent.message)}</div>

            {/* confidence bar */}
            <div className="mt-2 h-1.5 rounded-full bg-stone-200/60">
              <motion.div
                className="h-1.5 rounded-full"
                style={{ backgroundColor: theme.accent }}
                animate={{ width: `${Math.max(6, conf * 100)}%` }}
                transition={{ type: 'spring', stiffness: 140, damping: 22 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* role tag — always visible */}
      <div className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-stone-900/75 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.22em] text-[#fff8ef] shadow-md backdrop-blur">
        <span className="text-xs">{theme.icon}</span>
        {name}
      </div>
    </motion.div>
  );
};

/* ──────────────────────── trade flash overlay ──────────────────────── */

const TradeFlash = ({ trade, isRunning }) => {
  if (!trade || trade.side === 'HOLD') return null;
  const isBuy = trade.side === 'BUY';
  return (
    <motion.div
      className={`pointer-events-none absolute inset-0 z-20 rounded-[2.4rem] ${
        isBuy ? 'bg-emerald-400/8' : 'bg-rose-400/8'
      }`}
      animate={{ opacity: [0, 0.6, 0] }}
      transition={{ duration: 0.6, repeat: isRunning ? 0 : 0 }}
    />
  );
};

/* ──────────────────────── main scene ──────────────────────── */

export const OfficeScene = ({ agents, flow, trade, currentStep, isRunning, engine }) => {
  const tradeTone =
    trade?.side === 'BUY' ? 'text-emerald-600' : trade?.side === 'SELL' ? 'text-rose-600' : 'text-stone-500';

    <div className="relative min-h-[680px] w-full overflow-hidden rounded-[2.4rem] border border-[#7d5a4f]/20 shadow-[0_34px_90px_rgba(82,49,31,0.18)] bg-[#d2a382]">
      {/* pixel-art background */}
      <img
        src={officeBg}
        alt="Quant Office"
        className="absolute inset-0 h-full w-full"
        style={{ imageRendering: 'pixelated' }}
      />

      {/* slight gradient overlay so text is readeable */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/10 via-transparent to-black/20" />

      {/* trade flash */}
      <TradeFlash trade={trade} isRunning={isRunning} />

      {/* header bar */}
      <div className="relative z-10 flex items-start justify-between gap-4 p-6">
        <div>
          <div className="mb-1 inline-flex items-center gap-2 rounded-full bg-black/50 px-4 py-1 text-[11px] font-bold uppercase tracking-[0.34em] text-white/90 backdrop-blur">
            <span className={`h-2.5 w-2.5 rounded-full ${isRunning ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]' : 'bg-stone-400'}`} />
            Wild Card Floor
          </div>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-white drop-shadow-[0_2px_6px_rgba(0,0,0,0.5)]">
            🏢 Cutesy Quant Office
          </h2>
          <p className="mt-1 max-w-xl text-xs leading-5 text-white/80 drop-shadow">
            Five desks, one coordinated loop. Watch signals travel desk-to-desk in real time.
          </p>
        </div>

        {/* engine + trade info card */}
        <div className="max-w-[320px] rounded-2xl border border-white/20 bg-black/50 px-5 py-4 text-right backdrop-blur-md">
          <div className="text-[10px] font-bold uppercase tracking-[0.3em] text-white/60">Engine</div>
          <div className="mt-1 text-base leading-tight font-bold text-white break-words">{engine?.name || 'Desk Policy'}</div>
          <div className="text-sm text-white/70">{engine?.mode || 'Rule Fallback'}</div>
          <div className={`mt-2 text-xs font-bold uppercase tracking-[0.26em] ${
            trade?.side === 'BUY' ? 'text-emerald-400' : trade?.side === 'SELL' ? 'text-rose-400' : 'text-white/50'
          }`}>
            {trade?.side || 'HOLD'} · ${trade?.price?.toFixed(0) || '—'} · Step {currentStep}
          </div>
        </div>
      </div>

      {/* data-link SVG layer */}
      <svg viewBox="0 0 1200 620" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 z-[5] h-full w-full">
        {flow?.map((item) => (
          <DataLink key={`${item.from}-${item.to}-${item.tone}`} flow={item} isRunning={isRunning} />
        ))}
      </svg>

      {/* market board overlay (right side) */}
      <div 
        className="absolute z-[6] w-[220px] rounded-2xl border border-white/20 bg-black/55 px-4 py-3 backdrop-blur-md"
        style={{ left: '76%', top: '35%' }}
      >
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-bold text-white">📈 Market Board</div>
          <div className={`rounded-full px-2 py-0.5 text-[9px] font-bold uppercase ${
            trade?.side === 'BUY' ? 'bg-emerald-500/30 text-emerald-300'
            : trade?.side === 'SELL' ? 'bg-rose-500/30 text-rose-300'
            : 'bg-white/10 text-white/50'
          }`}>
            {trade?.side || 'HOLD'}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-white">
          <div className="rounded-xl bg-white/10 px-3 py-2">
            <div className="text-[9px] uppercase tracking-wider text-white/50">Size</div>
            <div className="text-sm font-bold">{trade?.size?.toFixed(2) || '0.00'}</div>
          </div>
          <div className="rounded-xl bg-white/10 px-3 py-2">
            <div className="text-[9px] uppercase tracking-wider text-white/50">Price</div>
            <div className="text-sm font-bold">${trade?.price?.toFixed(0) || '—'}</div>
          </div>
          <div className="rounded-xl bg-white/10 px-3 py-2">
            <div className="text-[9px] uppercase tracking-wider text-white/50">SL</div>
            <div className="text-sm font-bold">${trade?.sl?.toFixed(0) || '—'}</div>
          </div>
          <div className="rounded-xl bg-white/10 px-3 py-2">
            <div className="text-[9px] uppercase tracking-wider text-white/50">TP</div>
            <div className="text-sm font-bold">${trade?.tp?.toFixed(0) || '—'}</div>
          </div>
        </div>
      </div>

      {/* agent speech bubbles */}
      {Object.entries(STATIONS).map(([name]) => (
        <AgentBubble key={name} name={name} agent={agents?.[name]} isRunning={isRunning} />
      ))}
    </div>
  );
};
