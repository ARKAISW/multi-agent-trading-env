import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const getRoleEmoji = (role) => {
  const map = {
    "Researcher": "👨‍💻",
    "Fundamental Analyst": "🌍",
    "Risk Manager": "🛡️",
    "Trader": "⚡",
    "Portfolio Manager": "🧭"
  };
  return map[role] || "🤖";
};

const getStatusColor = (status) => {
  return status === "active" ? "ring-2 ring-emerald-400" : "ring-1 ring-slate-700 opacity-80";
};

export const AgentCard = ({ name, agent }) => {
  const { message, confidence, status } = agent || {};
  
  return (
    <div className={`relative flex flex-col items-center bg-slate-800 p-4 rounded-xl shadow-lg transition-all ${getStatusColor(status)} w-48`}>
      <AnimatePresence>
        {status === 'active' && message && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="absolute -top-12 bg-white text-slate-900 px-3 py-1.5 rounded-lg text-xs font-bold shadow-xl whitespace-nowrap z-10 before:content-[''] before:absolute before:-bottom-2 before:left-1/2 before:-translate-x-1/2 before:border-l-[6px] before:border-r-[6px] before:border-t-[8px] before:border-l-transparent before:border-r-transparent before:border-t-white"
          >
            {message}
          </motion.div>
        )}
      </AnimatePresence>
      
      <div className="text-4xl mb-2">{getRoleEmoji(name)}</div>
      <h3 className="text-sm font-bold text-slate-200 text-center">{name}</h3>
      
      <div className="w-full mt-3 bg-slate-700 h-1.5 rounded-full overflow-hidden">
        <motion.div 
          className="h-full bg-blue-500" 
          animate={{ width: `${(confidence || 0) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
      <div className="w-full flex justify-between mt-1 text-[10px] text-slate-400">
        <span>Conf: {(confidence || 0).toFixed(2)}</span>
        <span className="uppercase text-[9px] tracking-wider">{status}</span>
      </div>
    </div>
  );
};
