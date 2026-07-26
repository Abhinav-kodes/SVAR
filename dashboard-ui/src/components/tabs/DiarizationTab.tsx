import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Users, Cpu, ShieldCheck, Play, AlertTriangle } from 'lucide-react';

interface DiarizationTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const DiarizationTab: React.FC<DiarizationTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const segments = data?.segments || [];
  const duration = data?.duration_s || 1;
  const talk = data?.talk_ratio || {};
  const roleRes = data?.role_resolution || {};

  if (segments.length === 0) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No diarization segments available.
      </div>
    );
  }

  const roleMethod = roleRes.method || 'heuristic';
  const roleConf = roleRes.confidence != null ? `${(roleRes.confidence * 100).toFixed(0)}%` : '--';
  const roleStatus = roleRes.status || 'resolved';
  const roleMap = roleRes.mapping || {};

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Interactive Timeline Bar */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-cyan-400" />
            <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
              Diarized Speaker Timeline
            </h3>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-cyan-400">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block shadow-glow-cyan" />
              Agent ({talk.agent_duration_s || 0}s)
            </span>
            <span className="flex items-center gap-1.5 text-amber-400">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block shadow-glow-amber" />
              Customer ({talk.customer_duration_s || 0}s)
            </span>
          </div>
        </div>

        {/* Timeline Bar Scrubber */}
        <div
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickRatio = (e.clientX - rect.left) / rect.width;
            onSeekAudio?.(clickRatio * duration);
          }}
          className="relative h-10 w-full bg-dark-900/80 rounded-xl overflow-hidden border border-white/10 cursor-pointer group"
        >
          {segments.map((seg, idx) => {
            const leftPct = (seg.start_time_s / duration) * 100;
            const widthPct = Math.max((seg.duration_s / duration) * 100, 0.4);
            const isAgent = seg.speaker === 'agent' || seg.speaker === 'spk_0';
            const isActive = currentTime >= seg.start_time_s && currentTime <= seg.end_time_s;

            return (
              <div
                key={idx}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeekAudio?.(seg.start_time_s);
                }}
                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                title={`${isAgent ? 'Agent' : 'Customer'} (${seg.start_time_s.toFixed(1)}s - ${seg.end_time_s.toFixed(1)}s)`}
                className={`absolute top-1 bottom-1 rounded transition-all duration-150 ${
                  isAgent
                    ? 'bg-cyan-500 hover:bg-cyan-400 shadow-glow-cyan'
                    : 'bg-amber-500 hover:bg-amber-400 shadow-glow-amber'
                } ${isActive ? 'ring-2 ring-white z-10 scale-y-110' : 'opacity-85'}`}
              />
            );
          })}

          {/* Current Playhead Vertical Indicator */}
          <div
            style={{ left: `${Math.min((currentTime / duration) * 100, 100)}%` }}
            className="absolute top-0 bottom-0 w-0.5 bg-white z-20 shadow-[0_0_8px_#ffffff] pointer-events-none"
          />
        </div>
      </div>

      {/* Role Resolution Overview Card */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center gap-2 border-b border-white/10 pb-3">
          <Cpu className="w-5 h-5 text-purple-400" />
          <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
            AI Speaker Role Resolution Engine
          </h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Resolution Method</div>
            <div className="text-sm font-bold text-cyan-400 capitalize">{roleMethod}</div>
          </div>

          <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Confidence Score</div>
            <div className="text-sm font-bold text-emerald-400">{roleConf}</div>
          </div>

          <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Status</div>
            <div className="text-sm font-bold text-slate-200 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span className="capitalize">{roleStatus}</span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-white/5 border border-white/10 space-y-1">
            <div className="text-[11px] font-semibold text-slate-400 uppercase">Speaker Mapping</div>
            <div className="text-xs font-mono text-slate-300 truncate">
              {Object.entries(roleMap)
                .map(([k, v]) => `${k}→${v}`)
                .join(', ') || 'Resolved automatically'}
            </div>
          </div>
        </div>
      </div>

      {/* Segment Details Table */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
          Turn-by-Turn Segment Details
        </h3>

        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-left text-xs">
            <thead className="bg-white/5 text-slate-400 uppercase font-semibold text-[11px]">
              <tr>
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">Speaker</th>
                <th className="px-4 py-3">Start</th>
                <th className="px-4 py-3">End</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {segments.map((seg, idx) => {
                const isAgent = seg.speaker === 'agent' || seg.speaker === 'spk_0';
                const isActive = currentTime >= seg.start_time_s && currentTime <= seg.end_time_s;

                return (
                  <tr
                    key={idx}
                    onClick={() => onSeekAudio?.(seg.start_time_s)}
                    className={`cursor-pointer transition-colors duration-150 ${
                      isActive ? 'bg-cyan-500/15 font-semibold text-white' : 'hover:bg-white/5 text-slate-300'
                    }`}
                  >
                    <td className="px-4 py-3 text-slate-500 font-mono">{idx + 1}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wide ${
                          isAgent
                            ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                            : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                        }`}
                      >
                        {isAgent ? 'Agent' : 'Customer'}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono">{seg.start_time_s.toFixed(2)}s</td>
                    <td className="px-4 py-3 font-mono">{seg.end_time_s.toFixed(2)}s</td>
                    <td className="px-4 py-3 font-mono">{seg.duration_s.toFixed(2)}s</td>
                    <td className="px-4 py-3">
                      <span className="flex items-center gap-1">
                        {seg.confidence != null ? `${(seg.confidence * 100).toFixed(0)}%` : '--'}
                        {seg.uncertain && <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button className="p-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 transition-colors">
                        <Play className="w-3.5 h-3.5 fill-current" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
