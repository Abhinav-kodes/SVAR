import React, { useState, useMemo } from 'react';
import type { CallData } from '../../types/dashboard';
import { Search } from 'lucide-react';

interface TranscriptTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const TranscriptTab: React.FC<TranscriptTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const segments = data?.segments || [];
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    if (!query.trim()) return segments;
    const q = query.toLowerCase();
    return segments.filter((seg) => (seg.text || '').toLowerCase().includes(q));
  }, [segments, query]);

  if (segments.length === 0) {
    return (
      <div className="card text-center text-[13px]" style={{ color: 'var(--text-secondary)', padding: '40px' }}>
        No transcript data available.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between gap-5">
          <div>
            <h2 className="text-[15.5px] font-semibold">Transcript</h2>
            <div className="text-[12.5px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              Synchronized speaker turns ({segments.length} turns).
            </div>
          </div>

          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-[7px] min-w-[220px]"
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border-strong)',
              color: 'var(--text-tertiary)',
            }}
          >
            <Search className="w-[14px] h-[14px]" strokeWidth={1.7} />
            <input
              type="text"
              placeholder="Search transcript…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="bg-transparent border-none outline-none text-[12.5px] w-full"
              style={{ color: 'var(--text-primary)' }}
            />
          </div>
        </div>
        <div className="tick-divider" />

        {/* Turn rows */}
        <div>
          {filtered.map((seg, idx) => {
            const isAgent = (seg.speaker || '').toLowerCase().includes('agent');
            const isActive = currentTime >= seg.start_time_s && currentTime < (seg.end_time_s || seg.start_time_s + 5);

            return (
              <div
                key={idx}
                onClick={() => onSeekAudio?.(seg.start_time_s)}
                className="trow cursor-pointer transition-colors duration-100 hover:bg-[#211C15]/60"
                style={isActive ? { background: 'var(--amber-dim)', margin: '0 -14px', padding: '13px 14px', borderRadius: '8px', borderBottomColor: 'transparent' } : undefined}
                title="Click to play audio from this turn"
              >
                <span className={`tag ${isAgent ? 'tag-agent' : 'tag-customer'}`}>
                  {isAgent ? 'Agent' : 'Customer'}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="num text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                      {seg.start_time_s.toFixed(1)}s – {(seg.end_time_s || seg.start_time_s + 2).toFixed(1)}s
                    </span>
                  </div>
                  <div className="text-[13.5px] leading-relaxed">{seg.text || '(silence)'}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
