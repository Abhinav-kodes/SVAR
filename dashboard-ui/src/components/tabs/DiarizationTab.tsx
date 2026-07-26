import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Clock } from 'lucide-react';

interface DiarizationTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const DiarizationTab: React.FC<DiarizationTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const segments = data?.segments || [];
  const talkRatio = data?.talk_ratio;
  const roleRes = data?.role_resolution;

  if (segments.length === 0) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No speaker identification data available.
      </div>
    );
  }

  const agentPct = ((talkRatio?.agent_ratio ?? 0.5) * 100).toFixed(0);
  const customerPct = ((talkRatio?.customer_ratio ?? 0.5) * 100).toFixed(0);

  return (
    <div className="space-y-6">
      {/* Overview Header */}
      <div className="panel p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white">Speaker identification</h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
              {segments.length} speaker turns
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Diarization turn mapping resolved via Gemini role inference.
          </p>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-400">
          <div>
            Agent talk: <strong className="text-slate-100">{agentPct}%</strong>
          </div>
          <div>
            Customer talk: <strong className="text-slate-100">{customerPct}%</strong>
          </div>
          {roleRes?.confidence && (
            <div>
              Confidence: <strong className="text-emerald-400">{(roleRes.confidence * 100).toFixed(0)}%</strong>
            </div>
          )}
        </div>
      </div>

      {/* Speaker Turns Table */}
      <div className="panel p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Speaker turn breakdown</h3>

        <div className="space-y-2 max-h-[calc(100vh-280px)] overflow-y-auto pr-1">
          {segments.map((seg, idx) => {
            const isAgent = (seg.speaker || '').toLowerCase().includes('agent');
            const isActive = currentTime >= seg.start_time_s && currentTime <= seg.end_time_s;

            return (
              <div
                key={idx}
                onClick={() => onSeekAudio?.(seg.start_time_s)}
                className={`p-3 rounded-md border transition-colors cursor-pointer flex items-center justify-between gap-4 text-xs ${
                  isActive
                    ? 'bg-sky-500/10 border-sky-500/40 text-slate-100'
                    : 'bg-slate-900/40 border-slate-800 hover:bg-slate-800/50 text-slate-300'
                }`}
              >
                <div className="flex items-center gap-3 truncate">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${isAgent ? 'badge-neutral' : 'badge-neutral'}`}>
                    {isAgent ? 'Agent' : 'Customer'}
                  </span>
                  <span className="truncate text-slate-200">{seg.text || '(Speech turn)'}</span>
                </div>

                <div className="flex items-center gap-3 text-slate-500 font-mono text-[11px] flex-shrink-0">
                  <Clock className="w-3 h-3 text-slate-500" />
                  <span>{seg.start_time_s.toFixed(1)}s – {seg.end_time_s.toFixed(1)}s</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
