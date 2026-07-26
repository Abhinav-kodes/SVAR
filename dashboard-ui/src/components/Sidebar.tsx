import React from 'react';
import {
  FileAudio,
  Play,
  Loader2,
  LayoutDashboard,
  Volume2,
  FileText,
  Smile,
  ShieldAlert,
  Award,
  FileSpreadsheet,
  PhoneCall,
} from 'lucide-react';
import type { TabId, ProgressState } from '../types/dashboard';

interface SidebarProps {
  files: string[];
  activeFile: string;
  onSelectFile: (file: string) => void;
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  onRunAnalysis: (file: string) => void;
  progress: ProgressState | null;
}

const TABS: Array<{ id: TabId; label: string; icon: React.FC<{ className?: string; strokeWidth?: number }> }> = [
  { id: 'summary', label: 'Call overview', icon: LayoutDashboard },
  { id: 'compliance', label: 'Compliance & policy', icon: ShieldAlert },
  { id: 'qascore', label: 'Quality review', icon: Award },
  { id: 'crm', label: 'CRM note', icon: FileSpreadsheet },
  { id: 'transcript', label: 'Transcript', icon: FileText },
  { id: 'emotion', label: 'Emotion & sentiment', icon: Smile },
  { id: 'denoising', label: 'Audio quality', icon: Volume2 },
];

export const Sidebar: React.FC<SidebarProps> = ({
  files,
  activeFile,
  onSelectFile,
  activeTab,
  onSelectTab,
  onRunAnalysis,
  progress,
}) => {
  const isRunning = progress?.status === 'running';

  return (
    <aside
      className="w-[250px] flex-shrink-0 flex flex-col overflow-y-auto"
      style={{
        background: 'var(--surface-1)',
        borderRight: '1px solid var(--border)',
        padding: '20px 16px 76px',
        gap: '22px',
        display: 'flex',
      }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-0.5 pb-1">
        <div
          className="w-[30px] h-[30px] rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--amber-dim)', color: 'var(--amber-strong)' }}
        >
          <PhoneCall className="w-4 h-4" strokeWidth={1.7} />
        </div>
        <div className="flex items-center">
          <span className="text-[14.5px] font-semibold tracking-tight" style={{ letterSpacing: '-0.01em' }}>
            SVAR Call Analytics
          </span>
          <span
            className="ml-1.5 text-[10px] px-1.5 py-px rounded"
            style={{
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-tertiary)',
              background: 'var(--surface-3)',
            }}
          >
            v2.4
          </span>
        </div>
      </div>

      {/* Recording Selector */}
      <div className="flex flex-col gap-1.5">
        <div className="eyebrow px-2.5 pb-0.5">Select recording</div>

        {files.map((file) => {
          const isSelected = file === activeFile;
          return (
            <button
              key={file}
              onClick={() => onSelectFile(file)}
              className="flex items-center gap-2 px-2.5 py-2 rounded-[7px] text-[13px] cursor-pointer transition-colors duration-100 text-left w-full"
              style={{
                color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: isSelected ? 'var(--surface-3)' : 'transparent',
                borderLeft: isSelected ? '2px solid var(--amber)' : '2px solid transparent',
                border: 'none',
                borderLeftWidth: '2px',
                borderLeftStyle: 'solid',
                borderLeftColor: isSelected ? 'var(--amber)' : 'transparent',
              }}
            >
              <FileAudio
                className="w-[15px] h-[15px] flex-shrink-0"
                strokeWidth={1.6}
                style={{ opacity: isSelected ? 1 : 0.75, color: isSelected ? 'var(--amber-strong)' : 'inherit' }}
              />
              <span className="truncate flex-1">{file}</span>
              {isSelected && (
                <span
                  className="w-[5px] h-[5px] rounded-full flex-shrink-0"
                  style={{ background: 'var(--amber)' }}
                />
              )}
            </button>
          );
        })}

        <button
          onClick={() => onRunAnalysis(activeFile)}
          disabled={isRunning}
          className="btn-analyze mt-1.5"
        >
          {isRunning ? (
            <>
              <Loader2 className="w-[13px] h-[13px] animate-spin" />
              <span>Analyzing…</span>
            </>
          ) : (
            <>
              <Play className="w-[13px] h-[13px] fill-current" />
              <span>Analyze call</span>
            </>
          )}
        </button>

        {/* Progress */}
        {isRunning && progress && (
          <div className="space-y-1 pt-1 px-1">
            <div className="flex justify-between text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              <span className="capitalize">{progress.current_stage || 'Processing'}</span>
              <span className="num">{progress.percent}%</span>
            </div>
            <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--surface-3)' }}>
              <div
                className="h-full transition-all duration-300"
                style={{ width: `${progress.percent}%`, background: 'var(--amber)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex flex-col gap-1.5">
        <div className="eyebrow px-2.5 pb-0.5">Analysis views</div>

        <div className="flex flex-col gap-0.5">
          {TABS.map(({ id, label, icon: Icon }) => {
            const isActive = activeTab === id;
            return (
              <button
                key={id}
                onClick={() => onSelectTab(id)}
                className="flex items-center gap-2.5 px-2.5 py-2 rounded-[7px] text-[13px] cursor-pointer transition-colors duration-100 text-left w-full"
                style={{
                  color: isActive ? 'var(--amber-strong)' : 'var(--text-secondary)',
                  background: isActive ? 'var(--amber-dim)' : 'transparent',
                  fontWeight: isActive ? 500 : 400,
                  border: 'none',
                }}
              >
                <Icon className="w-4 h-4 flex-shrink-0" strokeWidth={1.6} />
                <span>{label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
};
