import React from 'react';
import type { CallData } from '../../types/dashboard';
import { ShieldCheck, Play, AlertOctagon, AlertTriangle } from 'lucide-react';

interface ComplianceTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
}

export const ComplianceTab: React.FC<ComplianceTabProps> = ({ data, onSeekAudio }) => {
  const compliance = data?.compliance;

  if (!compliance) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No compliance evaluation data available.
      </div>
    );
  }

  const { compliant, total_violations = 0, agent_violations = 0, customer_violations = 0, segment_results = [] } = compliance;

  return (
    <div className="space-y-6">
      {/* Risk Summary Header */}
      <div className="panel p-5 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white">Compliance & policy review</h2>
            <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${total_violations > 0 ? 'badge-danger' : 'badge-success'}`}>
              {compliant ? 'Passed' : 'Action required'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            RBI, IRDAI regulatory rules, debt collection conduct, and abusive language inspection.
          </p>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-400">
          <div>
            Total flags: <strong className="text-slate-100">{total_violations}</strong>
          </div>
          <div>
            Agent: <strong className="text-amber-400">{agent_violations}</strong>
          </div>
          <div>
            Customer: <strong className="text-purple-400">{customer_violations}</strong>
          </div>
        </div>
      </div>

      {/* Flagged Policy Evidence List */}
      <div className="panel p-5 space-y-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Evidence & Flagged Statements
        </h3>

        {total_violations === 0 ? (
          <div className="p-4 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 flex-shrink-0" />
            <div>
              <span className="font-semibold block">Clean call record</span>
              <span className="text-emerald-300/80">No RBI, IRDAI, or conduct policy violations were triggered.</span>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {segment_results.map((sr, sIdx) => {
              if (!sr || sr.violation_count === 0 || !Array.isArray(sr.flags)) return null;

              const startTime = (sr as any).start ?? (sr as any).start_time_s ?? 0;

              return sr.flags.map((flag, fIdx) => {
                const parts = flag.split(':');
                const cat = parts[0].toUpperCase();
                const detail = parts.slice(1).join(':') || flag;
                const isCritical = cat === 'RBI' || cat === 'THREAT';

                return (
                  <div
                    key={`${sIdx}-${fIdx}`}
                    className="p-4 rounded-md bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-colors flex items-start justify-between gap-4"
                  >
                    <div className="flex items-start gap-3">
                      {isCritical ? (
                        <AlertOctagon className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                      )}

                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${isCritical ? 'badge-danger' : 'badge-warning'}`}>
                            {cat}
                          </span>
                          <span className="text-xs font-semibold text-slate-200 capitalize">
                            {sr.speaker || 'Speaker'}
                          </span>
                          <span className="text-xs text-slate-500">@ {typeof startTime === 'number' ? startTime.toFixed(1) : startTime}s</span>
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed font-mono bg-slate-950/40 p-2 rounded border border-slate-800/80">
                          "{detail}"
                        </p>
                      </div>
                    </div>

                    {/* Listen to Evidence Action */}
                    <button
                      onClick={() => onSeekAudio?.(startTime)}
                      className="btn-secondary px-2.5 py-1.5 text-xs flex items-center gap-1.5 flex-shrink-0"
                      title="Play audio from this timestamp"
                    >
                      <Play className="w-3 h-3 fill-current text-sky-400" />
                      <span>Listen</span>
                    </button>
                  </div>
                );
              });
            })}
          </div>
        )}
      </div>
    </div>
  );
};
