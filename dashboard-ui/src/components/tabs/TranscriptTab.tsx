import React, { useState, useEffect, useRef } from 'react';
import type { CallData } from '../../types/dashboard';
import { FileText, Search, Play, Volume2 } from 'lucide-react';

interface TranscriptTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const TranscriptTab: React.FC<TranscriptTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const activeTurnRef = useRef<HTMLDivElement | null>(null);

  const segments = (data?.segments || []).filter((s) => s.text && s.text.trim().length > 0);

  useEffect(() => {
    if (activeTurnRef.current) {
      activeTurnRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [currentTime]);

  if (segments.length === 0) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No transcript content available for this call.
      </div>
    );
  }

  const filteredSegments = segments.filter((s) =>
    s.text?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Transcript Header & Search Bar */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
              Hindi Conversational Speech Transcript
            </h3>
          </div>

          {/* Search Filter */}
          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search transcript text..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-dark-900/80 border border-white/10 rounded-xl pl-9 pr-4 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
            />
          </div>
        </div>

        {/* Transcript Conversation Flow */}
        <div className="space-y-3 max-h-[550px] overflow-y-auto pr-2 pt-2">
          {filteredSegments.map((seg, idx) => {
            const isAgent = seg.speaker === 'agent' || seg.speaker === 'spk_0';
            const isActive = currentTime >= seg.start_time_s && currentTime <= seg.end_time_s;

            return (
              <div
                key={idx}
                ref={isActive ? activeTurnRef : null}
                onClick={() => onSeekAudio?.(seg.start_time_s)}
                className={`p-4 rounded-xl transition-all duration-200 cursor-pointer border flex gap-3.5 items-start ${
                  isActive
                    ? 'bg-cyan-500/15 border-cyan-500/60 shadow-glow-cyan'
                    : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10'
                }`}
              >
                {/* Speaker Badge */}
                <div className="flex flex-col items-center gap-1.5 flex-shrink-0 pt-0.5">
                  <span
                    className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider ${
                      isAgent
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                    }`}
                  >
                    {isAgent ? 'Agent' : 'Customer'}
                  </span>
                  {isActive && (
                    <span className="flex items-center text-cyan-400 animate-pulse">
                      <Volume2 className="w-3.5 h-3.5" />
                    </span>
                  )}
                </div>

                {/* Text & Timestamp Content */}
                <div className="flex-1 space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>
                      {seg.start_time_s.toFixed(1)}s – {seg.end_time_s.toFixed(1)}s
                    </span>
                    <button className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 opacity-0 hover:opacity-100 transition-opacity">
                      <Play className="w-3 h-3 fill-current" />
                      Listen
                    </button>
                  </div>
                  <p className="text-sm text-slate-100 leading-relaxed font-sans">
                    {seg.text}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
