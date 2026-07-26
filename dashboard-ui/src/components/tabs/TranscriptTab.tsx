import React, { useState, useRef, useEffect } from 'react';
import type { CallData } from '../../types/dashboard';
import { Search, Play } from 'lucide-react';

interface TranscriptTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const TranscriptTab: React.FC<TranscriptTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const segments = data?.segments || [];
  const [searchTerm, setSearchTerm] = useState('');
  const activeTurnRef = useRef<HTMLDivElement | null>(null);

  const filteredSegments = segments.filter((seg) => {
    if (!searchTerm) return true;
    const text = seg.text?.toLowerCase() || '';
    const speaker = seg.speaker?.toLowerCase() || '';
    return text.includes(searchTerm.toLowerCase()) || speaker.includes(searchTerm.toLowerCase());
  });

  // Auto-scroll active turn into view
  useEffect(() => {
    if (activeTurnRef.current) {
      activeTurnRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [currentTime]);

  if (segments.length === 0) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No transcript segments available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Header */}
      <div className="panel p-4 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-white">Transcript</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Synchronized speaker turns ({segments.length} turns).
          </p>
        </div>

        <div className="relative w-64">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search transcript..."
            className="w-full pl-9 pr-3 py-1.5 bg-slate-900 border border-slate-800 rounded-md text-xs text-slate-200 focus:outline-none focus:border-sky-500"
          />
        </div>
      </div>

      {/* Transcript Turns List */}
      <div className="panel p-4 space-y-2.5 max-h-[calc(100vh-250px)] overflow-y-auto">
        {filteredSegments.map((seg, idx) => {
          const isAgent = (seg.speaker || '').toLowerCase().includes('agent');
          const isActive = currentTime >= seg.start_time_s && currentTime <= seg.end_time_s;

          return (
            <div
              key={idx}
              ref={isActive ? activeTurnRef : null}
              onClick={() => onSeekAudio?.(seg.start_time_s)}
              className={`p-3 rounded-md transition-colors cursor-pointer border ${
                isActive
                  ? 'bg-sky-500/10 border-sky-500/40 text-slate-100'
                  : 'bg-slate-900/40 border-slate-800/80 hover:bg-slate-800/50 text-slate-300'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${isAgent ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30' : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'}`}>
                    {isAgent ? 'Agent' : 'Customer'}
                  </span>
                  <span className="text-[11px] font-mono text-slate-500">
                    {seg.start_time_s.toFixed(1)}s – {seg.end_time_s.toFixed(1)}s
                  </span>
                </div>

                <button className="text-slate-400 hover:text-sky-400 flex items-center gap-1 text-[11px]">
                  <Play className="w-3 h-3 fill-current" />
                  <span>Seek</span>
                </button>
              </div>

              <p className="text-xs leading-relaxed font-sans font-normal text-slate-200">
                {seg.text || '(No speech transcript)'}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
