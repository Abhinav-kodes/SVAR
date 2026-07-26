import React from 'react';
import { PhoneCall, CheckCircle2, Clock } from 'lucide-react';
import type { CallData } from '../types/dashboard';

interface HeaderProps {
  activeFile: string;
  callData?: CallData;
}

export const Header: React.FC<HeaderProps> = ({ activeFile, callData }) => {
  const duration = callData?.duration_s ? `${callData.duration_s.toFixed(1)}s` : '--';
  const processTime = callData?.processing_time_s ? `${callData.processing_time_s.toFixed(1)}s` : '--';

  return (
    <header className="bg-[#121a26] border-b border-[#263245] px-6 py-3.5 flex items-center justify-between sticky top-0 z-30">
      {/* Title & Metadata */}
      <div className="flex items-center gap-4">
        <div className="w-9 h-9 rounded-md bg-sky-600/20 border border-sky-500/30 flex items-center justify-center text-sky-400 font-bold font-display">
          <PhoneCall className="w-5 h-5" />
        </div>

        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display font-semibold text-base text-slate-100 tracking-tight">
              SVAR Call Analytics
            </h1>
            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
              v2.4
            </span>
          </div>
          <p className="text-xs text-slate-400 flex items-center gap-2 mt-0.5">
            <span>Call: <strong className="text-slate-200 font-mono font-normal">{activeFile}</strong></span>
            <span className="text-slate-600">•</span>
            <span>Duration: {duration}</span>
          </p>
        </div>
      </div>

      {/* Status & Processing Metrics */}
      <div className="flex items-center gap-4 text-xs">
        <div className="hidden sm:flex items-center gap-2 text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-md border border-slate-800">
          <Clock className="w-3.5 h-3.5 text-slate-500" />
          <span>Processing time: <strong className="text-slate-200 font-medium">{processTime}</strong></span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Pipeline ready</span>
        </div>
      </div>
    </header>
  );
};
