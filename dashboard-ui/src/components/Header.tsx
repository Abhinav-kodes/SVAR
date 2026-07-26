import React from 'react';
import { Clock } from 'lucide-react';
import type { CallData } from '../types/dashboard';

interface HeaderProps {
  activeFile: string;
  callData?: CallData;
}

export const Header: React.FC<HeaderProps> = ({ activeFile, callData }) => {
  const duration = callData?.duration_s ? `${callData.duration_s.toFixed(1)}s` : '--';
  const processTime = callData?.processing_time_s ? `${callData.processing_time_s.toFixed(1)}s` : '--';

  return (
    <header
      className="flex flex-col sm:flex-row items-start sm:items-center justify-between px-4 sm:px-7 py-3 sm:py-4 gap-2.5 sm:gap-0 border-b"
      style={{ background: 'var(--surface-1)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center gap-2 text-[13px]" style={{ color: 'var(--text-secondary)' }}>
        <span>Recording <b className="font-medium" style={{ color: 'var(--text-primary)' }}>{activeFile}</b></span>
        <span style={{ color: 'var(--text-tertiary)' }}>·</span>
        <span>Duration <b className="num font-medium" style={{ color: 'var(--text-primary)' }}>{duration}</b></span>
      </div>

      <div className="flex items-center gap-2.5">
        <div
          className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-full"
          style={{
            color: 'var(--text-secondary)',
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
          }}
        >
          <Clock className="w-[13px] h-[13px]" strokeWidth={1.6} />
          <span>Processing</span>
          <span className="num" style={{ color: 'var(--text-primary)' }}>{processTime}</span>
        </div>

        <div
          className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-full"
          style={{
            color: 'var(--green)',
            background: 'var(--surface-2)',
            border: '1px solid rgba(127,169,135,0.25)',
          }}
        >
          <span
            className="w-[7px] h-[7px] rounded-full flex-shrink-0"
            style={{ background: 'var(--green)' }}
          />
          <span>Pipeline ready</span>
        </div>
      </div>
    </header>
  );
};
