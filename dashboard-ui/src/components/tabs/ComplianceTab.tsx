import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Scale, ShieldCheck, ShieldAlert } from 'lucide-react';

interface ComplianceTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
}

export const ComplianceTab: React.FC<ComplianceTabProps> = ({ data, onSeekAudio }) => {
  const compliance = data?.compliance;

  if (!compliance) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No compliance engine output available.
      </div>
    );
  }

  const { compliant, total_violations = 0, agent_violations = 0, customer_violations = 0, segment_results = [] } = compliance;

  const violationTypeStyles: Record<string, string> = {
    rbi: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    irdai: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    abusive: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    threat: 'bg-rose-600/20 text-rose-300 border-rose-500/40',
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Compliance Overview Grid */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center gap-2 border-b border-white/10 pb-3">
          <Scale className="w-5 h-5 text-cyan-400" />
          <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
            Regulatory Compliance & Risk Assessment
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-xs font-semibold text-slate-400 uppercase">Compliance Status</div>
            <div className="pt-1">
              {compliant ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  <ShieldCheck className="w-4 h-4" /> Compliant
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30">
                  <ShieldAlert className="w-4 h-4" /> Non-Compliant
                </span>
              )}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-xs font-semibold text-slate-400 uppercase">Total Violations</div>
            <div className={`font-display text-2xl font-bold ${total_violations > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {total_violations}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-xs font-semibold text-slate-400 uppercase">Agent Violations</div>
            <div className="font-display text-2xl font-bold text-amber-400">
              {agent_violations}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-xs font-semibold text-slate-400 uppercase">Customer Violations</div>
            <div className="font-display text-2xl font-bold text-purple-400">
              {customer_violations}
            </div>
          </div>
        </div>
      </div>

      {/* Flagged Violations Detailed List */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
          Detected Policy & Regulatory Flags
        </h3>

        {total_violations === 0 ? (
          <div className="p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 flex-shrink-0" />
            <div>
              <div className="font-semibold text-sm">Clean Call Recording</div>
              <div>No RBI, IRDAI, debt collection, or abusive language policy violations were triggered.</div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {segment_results.map((sr, sIdx) => {
              if (!sr || sr.violation_count === 0 || !Array.isArray(sr.flags)) return null;

              const startTime = (sr as any).start ?? (sr as any).start_time_s ?? 0;

              return sr.flags.map((flag, fIdx) => {
                const parts = flag.split(':');
                const typeKey = parts[0].toLowerCase();
                const styleClass = violationTypeStyles[typeKey] || violationTypeStyles.rbi;
                const message = parts.slice(1).join(':') || flag;

                return (
                  <div
                    key={`${sIdx}-${fIdx}`}
                    onClick={() => onSeekAudio?.(startTime)}
                    className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-rose-500/40 transition-all cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                  >
                    <div className="flex items-center gap-3">
                      <span className={`px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider border ${styleClass}`}>
                        {typeKey}
                      </span>
                      <p className="text-xs text-slate-200 font-medium">{message}</p>
                    </div>

                    <div className="text-[11px] text-slate-400 font-mono flex items-center gap-2">
                      <span className="capitalize">{sr.speaker || 'unknown'}</span>
                      <span>@ {typeof startTime === 'number' ? startTime.toFixed(1) : startTime}s</span>
                    </div>
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
