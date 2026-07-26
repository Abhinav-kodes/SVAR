import React from 'react';
import type { CallData } from '../../types/dashboard';
import { AlertTriangle } from 'lucide-react';

interface ComplianceTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const ComplianceTab: React.FC<ComplianceTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const compliance = data?.compliance;

  if (!compliance) {
    return (
      <div className="card text-center text-[13px]" style={{ color: 'var(--text-secondary)', padding: '40px' }}>
        No compliance evaluation data available.
      </div>
    );
  }

  const { total_violations = 0, agent_violations = 0, customer_violations = 0, segment_results = [] } = compliance;

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between gap-5">
          <div>
            <h2 className="text-[15.5px] font-semibold">
              Compliance & policy review
              {total_violations > 0 && (
                <span className="badge badge-danger ml-2">Action required</span>
              )}
            </h2>
            <div className="text-[12.5px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              RBI & IRDAI regulatory rules, debt collection conduct, and abusive language inspection.
            </div>
          </div>

          <div className="flex gap-5.5 flex-shrink-0">
            <div className="text-right">
              <span className="num text-[18px] font-semibold block">{total_violations}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>TOTAL FLAGS</span>
            </div>
            <div className="text-right">
              <span className="num text-[18px] font-semibold block" style={{ color: 'var(--amber)' }}>{agent_violations}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>AGENT</span>
            </div>
            <div className="text-right">
              <span className="num text-[18px] font-semibold block" style={{ color: 'var(--red-strong)' }}>{customer_violations}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>CUSTOMER</span>
            </div>
          </div>
        </div>
        <div className="tick-divider" />

        {/* Evidence List */}
        {total_violations === 0 ? (
          <div className="p-4 rounded-lg text-[13px]" style={{ background: 'var(--green-dim)', color: '#A9D0AF' }}>
            No RBI, IRDAI, or conduct policy violations were triggered.
          </div>
        ) : (
          <div className="flex flex-col gap-2.5">
            {segment_results.map((sr, sIdx) => {
              if (!sr || sr.violation_count === 0 || !Array.isArray(sr.flags)) return null;

              const segIdx = (sr as any).index ?? sIdx;
              const matchingSeg = data?.segments?.[segIdx] || data?.segments?.[sIdx];
              const rawStart = (sr as any).start ?? (sr as any).start_time_s ?? matchingSeg?.start_time_s ?? 0;
              const startTime = typeof rawStart === 'number' ? rawStart : parseFloat(rawStart) || 0;

              return sr.flags.map((flag, fIdx) => {
                const parts = flag.split(':');
                const cat = parts[0].toUpperCase();
                const detail = parts.slice(1).join(':') || flag;
                const isCritical = cat === 'THREAT';
                const isWarn = cat === 'ABUSIVE';

                const tierClass = isCritical ? 'tier-critical' : isWarn ? 'tier-warn' : '';
                const isActive = currentTime >= startTime && currentTime < (startTime + 4);

                return (
                  <div
                    key={`${sIdx}-${fIdx}`}
                    onClick={() => onSeekAudio?.(startTime)}
                    className={`evidence cursor-pointer transition-all duration-150 hover:bg-[#211C15]/80 ${tierClass}`}
                    style={isActive ? { outline: '2px solid var(--amber)', outlineOffset: '1px', background: 'var(--amber-dim)' } : undefined}
                    title="Click to play audio from this timestamp"
                  >
                    <div className="flex items-start gap-3 min-w-0">
                      <div
                        className="w-[26px] h-[26px] rounded-[6px] flex items-center justify-center flex-shrink-0 mt-px"
                        style={{
                          background: isCritical ? 'rgba(193,71,58,0.2)' : isWarn ? 'var(--amber-dim)' : 'var(--surface-3)',
                          color: isCritical ? 'var(--red-strong)' : isWarn ? 'var(--amber-strong)' : 'var(--text-tertiary)',
                        }}
                      >
                        <AlertTriangle className="w-[14px] h-[14px]" strokeWidth={1.6} />
                      </div>

                      <div>
                        <div className="text-xs mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                          <span className={`ev-tag ${tierClass ? '' : ''}`}>{cat}</span>
                          <b style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{sr.speaker || 'Speaker'}</b>
                          <span> · </span>
                          <span className="num">{startTime.toFixed(1)}s</span>
                        </div>
                        <div
                          className="text-[12.5px] leading-relaxed"
                          style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}
                        >
                          "{detail}"
                        </div>
                      </div>
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
