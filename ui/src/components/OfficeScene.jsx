import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const AGENT_THEMES = {
  Researcher: {
    accent: '#f28c6f',
    soft: 'from-rose-100 via-orange-50 to-amber-50',
    desk: '#f5d7c4',
    hair: '#47312c',
    outfit: '#f28c6f',
    label: 'Signal Lab',
  },
  'Risk Manager': {
    accent: '#5f8b7e',
    soft: 'from-emerald-100 via-teal-50 to-lime-50',
    desk: '#d3e9dc',
    hair: '#39514b',
    outfit: '#67a18f',
    label: 'Guard Rail',
  },
  'Portfolio Manager': {
    accent: '#e2a63b',
    soft: 'from-amber-100 via-yellow-50 to-orange-50',
    desk: '#f6e5bb',
    hair: '#594430',
    outfit: '#e2a63b',
    label: 'Capital Desk',
  },
  'Fundamental Analyst': {
    accent: '#7f8ce6',
    soft: 'from-sky-100 via-indigo-50 to-cyan-50',
    desk: '#dce3ff',
    hair: '#36406f',
    outfit: '#7f8ce6',
    label: 'Macro Wire',
  },
  Trader: {
    accent: '#e36b5f',
    soft: 'from-red-100 via-orange-50 to-yellow-50',
    desk: '#f7d0c7',
    hair: '#402825',
    outfit: '#db715e',
    label: 'Execution',
  },
};

const STATIONS = {
  Researcher: { left: '5%', top: '16%', anchor: [170, 180] },
  'Risk Manager': { left: '27%', top: '10%', anchor: [425, 145] },
  'Portfolio Manager': { left: '50%', top: '15%', anchor: [690, 170] },
  'Fundamental Analyst': { left: '18%', top: '56%', anchor: [340, 470] },
  Trader: { left: '57%', top: '58%', anchor: [775, 490] },
};

const MARKET_ANCHOR = [1045, 320];

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

const trimMessage = (message = '') => {
  if (message.length <= 94) {
    return message;
  }
  return `${message.slice(0, 91)}...`;
};

const DataLink = ({ flow, isRunning }) => {
  const start = flow.from === 'Market' ? MARKET_ANCHOR : STATIONS[flow.from]?.anchor;
  const end = flow.to === 'Market' ? MARKET_ANCHOR : STATIONS[flow.to]?.anchor;

  if (!start || !end) {
    return null;
  }

  const color = LINK_TONES[flow.tone] || '#8d99ae';
  const opacity = flow.active ? 0.95 : 0.35;

  return (
    <>
      <line
        x1={start[0]}
        y1={start[1]}
        x2={end[0]}
        y2={end[1]}
        stroke={color}
        strokeOpacity={opacity}
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={flow.active ? '9 15' : '6 16'}
      />
      {flow.active && (
        <>
          <motion.circle
            r="8"
            fill={color}
            cx={start[0]}
            cy={start[1]}
            animate={{
              cx: [start[0], end[0]],
              cy: [start[1], end[1]],
              opacity: [0, 1, 1, 0],
              scale: [0.7, 1, 1, 0.8],
            }}
            transition={{
              duration: isRunning ? Math.max(0.8, 1.8 - flow.strength) : 0,
              repeat: isRunning ? Infinity : 0,
              ease: 'linear',
            }}
          />
          <motion.circle
            r="4"
            fill="#fff8ef"
            cx={start[0]}
            cy={start[1]}
            animate={{
              cx: [start[0], end[0]],
              cy: [start[1], end[1]],
              opacity: [0, 1, 1, 0],
            }}
            transition={{
              duration: isRunning ? Math.max(0.8, 1.8 - flow.strength) : 0,
              repeat: isRunning ? Infinity : 0,
              ease: 'linear',
              delay: 0.18,
            }}
          />
        </>
      )}
    </>
  );
};

const AgentDesk = ({ name, agent, isRunning }) => {
  const theme = AGENT_THEMES[name];
  const station = STATIONS[name];
  const confidence = agent?.confidence || 0;
  const isActive = agent?.status === 'active';

  return (
    <motion.div
      className="absolute w-[250px]"
      style={{ left: station.left, top: station.top }}
      animate={isRunning ? { y: [0, -4, 0] } : { y: 0 }}
      transition={{
        duration: 3.2,
        repeat: isRunning ? Infinity : 0,
        ease: 'easeInOut',
        delay: confidence,
      }}
    >
      <AnimatePresence>
        {agent?.message && (
          <motion.div
            key={`${name}-${agent.message}`}
            initial={{ opacity: 0, y: 8, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.35 }}
            className={`mb-3 rounded-[1.25rem] border border-white/70 bg-gradient-to-br ${theme.soft} px-4 py-3 text-[11px] leading-4 text-stone-700 shadow-[0_14px_30px_rgba(83,55,43,0.12)]`}
          >
            <div className="mb-1 flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.28em] text-stone-500">
              <span>{theme.label}</span>
              <span>{isActive ? 'Live' : 'Standby'}</span>
            </div>
            {trimMessage(agent.message)}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="relative rounded-[2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/90 p-4 shadow-[0_24px_45px_rgba(77,44,26,0.14)] backdrop-blur">
        <div
          className="absolute inset-x-5 bottom-3 h-5 rounded-full blur-md"
          style={{ backgroundColor: `${theme.accent}55` }}
        />

        <div className="relative flex items-end gap-4">
          <div className="flex w-[72px] flex-col items-center">
            <motion.div
              className="relative h-[96px] w-[64px]"
              animate={isActive ? { rotate: [-1, 1, -1] } : { rotate: 0 }}
              transition={{ duration: 2.4, repeat: isActive ? Infinity : 0, ease: 'easeInOut' }}
            >
              <div
                className="absolute left-2 top-0 h-10 w-10 rounded-full border-2 border-[#3e2b25]/10"
                style={{ backgroundColor: '#ffd5bd' }}
              />
              <div
                className="absolute left-[5px] top-[-3px] h-5 w-11 rounded-t-full"
                style={{ backgroundColor: theme.hair }}
              />
              <div className="absolute left-[18px] top-[16px] h-[4px] w-[4px] rounded-full bg-stone-700" />
              <div className="absolute left-[28px] top-[16px] h-[4px] w-[4px] rounded-full bg-stone-700" />
              <div className="absolute left-[21px] top-[24px] h-[2px] w-[12px] rounded-full bg-rose-300" />
              <div
                className="absolute bottom-0 left-[8px] h-[58px] w-12 rounded-t-[22px]"
                style={{ backgroundColor: theme.outfit }}
              />
              <div className="absolute bottom-[20px] left-0 h-[10px] w-3 rounded-full bg-[#ffd5bd]" />
              <div className="absolute bottom-[20px] right-0 h-[10px] w-3 rounded-full bg-[#ffd5bd]" />
            </motion.div>

            <div className="mt-2 rounded-full bg-stone-900/75 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-[#fff8ef]">
              {isActive ? 'Active' : 'Calm'}
            </div>
          </div>

          <div className="flex-1">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-stone-800">{name}</div>
                <div className="text-[11px] uppercase tracking-[0.26em] text-stone-500">{theme.label}</div>
              </div>
              <div
                className="rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.22em]"
                style={{
                  backgroundColor: `${theme.accent}22`,
                  color: theme.accent,
                }}
              >
                {(confidence * 100).toFixed(0)}%
              </div>
            </div>

            <div
              className="rounded-[1.4rem] border border-white/60 p-3"
              style={{ backgroundColor: theme.desk }}
            >
              <div className="mb-2 flex items-end gap-[3px]">
                {[18, 28, 21, 34, 26, 39].map((height, index) => (
                  <motion.div
                    // eslint-disable-next-line react/no-array-index-key
                    key={`${name}-bar-${index}`}
                    className="w-2 rounded-full"
                    style={{ height, backgroundColor: `${theme.accent}bb` }}
                    animate={isRunning ? { scaleY: [0.88, 1.1, 0.92] } : { scaleY: 1 }}
                    transition={{
                      duration: 1.2,
                      repeat: isRunning ? Infinity : 0,
                      delay: index * 0.08,
                      ease: 'easeInOut',
                    }}
                  />
                ))}
              </div>

              <div className="mb-2 h-2 rounded-full bg-white/65">
                <motion.div
                  className="h-2 rounded-full"
                  style={{ backgroundColor: theme.accent }}
                  animate={{ width: `${Math.max(8, confidence * 100)}%` }}
                  transition={{ type: 'spring', stiffness: 140, damping: 22 }}
                />
              </div>

              <div className="min-h-[34px] overflow-hidden text-[11px] leading-4 text-stone-700">
                {trimMessage(agent?.message || 'Desk waiting for the next signal.')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export const OfficeScene = ({ agents, flow, trade, currentStep, isRunning, engine }) => {
  const tradeTone =
    trade?.side === 'BUY' ? 'text-emerald-600' : trade?.side === 'SELL' ? 'text-rose-600' : 'text-stone-500';

  return (
    <div className="relative min-h-[680px] overflow-hidden rounded-[2.4rem] border border-[#7d5a4f]/20 bg-[linear-gradient(180deg,#fff8ef_0%,#f9e6d7_50%,#f3dbc7_100%)] p-8 shadow-[0_34px_90px_rgba(82,49,31,0.18)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.95),transparent_38%),radial-gradient(circle_at_bottom_right,rgba(242,140,111,0.18),transparent_32%)]" />
      <div className="absolute inset-x-0 top-0 h-32 bg-[linear-gradient(180deg,rgba(187,216,240,0.85),rgba(255,255,255,0))]" />
      <div className="absolute left-[4%] top-[10%] h-32 w-40 rounded-[2rem] border border-white/70 bg-white/45" />
      <div className="absolute left-[20%] top-[8%] h-40 w-48 rounded-[2rem] border border-white/70 bg-white/35" />
      <div className="absolute right-[6%] top-[9%] h-36 w-44 rounded-[2rem] border border-white/70 bg-white/40" />
      <div className="absolute inset-x-8 bottom-8 h-24 rounded-[2rem] bg-[#e5cbb5]/65 blur-sm" />
      <div className="absolute inset-x-10 bottom-10 h-16 rounded-[1.75rem] border border-white/40 bg-[#f5dfca]/70" />

      <div className="relative z-10 mb-6 flex items-start justify-between gap-6">
        <div>
          <div className="mb-2 inline-flex items-center gap-2 rounded-full bg-white/75 px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.34em] text-stone-500">
            <span className={`h-2.5 w-2.5 rounded-full ${isRunning ? 'bg-emerald-500' : 'bg-stone-400'}`} />
            Wild Card Floor
          </div>
          <h2 className="text-3xl font-semibold tracking-tight text-stone-800">Cutesy Quant Office</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-stone-600">
            Five desks, one coordinated loop. Signals, risk checks, approvals, and trades are all pushed across the room in real time.
          </p>
        </div>

        <div className="min-w-[300px] rounded-[1.6rem] border border-[#7d5a4f]/15 bg-white/70 px-5 py-4 text-right shadow-[0_16px_34px_rgba(77,44,26,0.10)]">
          <div className="text-[11px] font-semibold uppercase tracking-[0.3em] text-stone-500">Engine</div>
          <div className="mt-2 text-lg font-semibold text-stone-800">{engine?.name || 'Desk Policy'}</div>
          <div className="text-sm text-stone-600">{engine?.mode || 'Rule Fallback'}</div>
          <div className={`mt-3 text-xs font-semibold uppercase tracking-[0.26em] ${tradeTone}`}>
            {trade?.side || 'HOLD'} desk pulse
          </div>
          <div className="mt-2 text-xs text-stone-500">Step {currentStep}</div>
        </div>
      </div>

      <svg viewBox="0 0 1200 620" className="pointer-events-none absolute inset-0 z-[2] h-full w-full">
        {flow?.map((item) => (
          <DataLink key={`${item.from}-${item.to}-${item.tone}`} flow={item} isRunning={isRunning} />
        ))}
      </svg>

      <div className="absolute right-[3.5%] top-[28%] z-[3] w-[250px] rounded-[2rem] border border-[#7d5a4f]/15 bg-[#fff8ef]/88 p-5 shadow-[0_24px_40px_rgba(77,44,26,0.14)]">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-stone-800">Market Board</div>
            <div className="text-[11px] uppercase tracking-[0.26em] text-stone-500">Execution rail</div>
          </div>
          <div className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] ${tradeTone}`}>
            {trade?.side || 'HOLD'}
          </div>
        </div>

        <div className="relative h-28 overflow-hidden rounded-[1.6rem] border border-white/60 bg-[#c6dff4] p-4">
          <div className="absolute inset-x-3 bottom-4 h-10 rounded-full bg-white/30 blur-md" />
          <div className="absolute left-4 top-4 h-12 w-16 rounded-[1rem] bg-white/55" />
          <div className="absolute right-5 top-5 h-7 w-10 rounded-full bg-white/40" />
          <div className="absolute bottom-4 left-4 right-4 h-[2px] bg-sky-900/15" />
          <motion.div
            className={`absolute left-5 top-12 h-3 w-3 rounded-full ${trade?.side === 'SELL' ? 'bg-rose-500' : 'bg-emerald-500'}`}
            animate={trade?.side === 'HOLD' ? { opacity: 0.45 } : { x: [0, 150], opacity: [0.25, 1, 0.3] }}
            transition={{ duration: 0.95, repeat: trade?.side === 'HOLD' || !isRunning ? 0 : Infinity, ease: 'easeInOut' }}
          />
          <motion.div
            className="absolute bottom-5 left-5 h-10 w-10 rounded-full bg-white/60"
            animate={trade?.side === 'HOLD' ? { scale: 1 } : { scale: [1, 1.28, 1] }}
            transition={{ duration: 1.1, repeat: trade?.side === 'HOLD' || !isRunning ? 0 : Infinity }}
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-stone-700">
          <div className="rounded-[1.25rem] bg-white/75 px-3 py-3">
            <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Ticket</div>
            <div className="mt-1 font-semibold">{trade?.size?.toFixed(2)} size</div>
          </div>
          <div className="rounded-[1.25rem] bg-white/75 px-3 py-3">
            <div className="text-[10px] uppercase tracking-[0.24em] text-stone-500">Price</div>
            <div className="mt-1 font-semibold">${trade?.price?.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {Object.entries(STATIONS).map(([name]) => (
        <AgentDesk key={name} name={name} agent={agents?.[name]} isRunning={isRunning} />
      ))}
    </div>
  );
};
