import React from 'react';
import { Activity, Clock, Sparkles, CheckCircle2, Mic } from 'lucide-react';

interface HeaderProps {
  activeFile: string;
  processingTime?: number;
  isAnalyzing: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeFile, processingTime, isAnalyzing }) => {
  return (
    <header className="sticky top-0 z-50 glass-header border-b border-white/10 px-6 py-3.5 flex items-center justify-between shadow-lg">
      {/* Left Brand */}
      <div className="flex items-center gap-3.5">
        <div className="relative group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 via-purple-500 to-indigo-600 flex items-center justify-center text-white font-display font-extrabold text-xl shadow-glow-cyan">
            S
          </div>
          <div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-emerald-500 border-2 border-dark-900 rounded-full animate-pulse" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display font-bold text-lg tracking-tight text-white">
              SVAR <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">Intelligence</span>
            </h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Enterprise v2.4
            </span>
          </div>
          <p className="text-xs text-slate-400 flex items-center gap-1.5 mt-0.5">
            <Sparkles className="w-3 h-3 text-purple-400" />
            End-to-End Hindi Speech & Conversational QA Pipeline
          </p>
        </div>
      </div>

      {/* Right Stats & Badges */}
      <div className="flex items-center gap-3">
        {activeFile && (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-slate-300">
            <Mic className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono font-medium text-slate-200">{activeFile}</span>
          </div>
        )}

        {isAnalyzing && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs font-semibold animate-pulse">
            <Activity className="w-3.5 h-3.5 animate-spin" />
            Pipeline Executing...
          </div>
        )}

        {processingTime !== undefined && !isAnalyzing && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
            <Clock className="w-3.5 h-3.5" />
            Processed in {processingTime}s
          </div>
        )}

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs font-semibold">
          <CheckCircle2 className="w-3.5 h-3.5" />
          Backend Ready
        </div>
      </div>
    </header>
  );
};
