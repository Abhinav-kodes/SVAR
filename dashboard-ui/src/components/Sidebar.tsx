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

const TABS: Array<{ id: TabId; label: string; icon: React.FC<{ className?: string }> }> = [
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
    <aside className="w-64 bg-[#121a26] border-r border-[#263245] flex flex-col h-[calc(100vh-53px)] overflow-y-auto">
      {/* Call Selector Section */}
      <div className="p-4 border-b border-[#263245] space-y-3">
        <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
          Select Recording
        </label>

        <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
          {files.map((file) => {
            const isSelected = file === activeFile;
            return (
              <button
                key={file}
                onClick={() => onSelectFile(file)}
                className={`w-full text-left px-3 py-2 rounded-md text-xs font-medium flex items-center justify-between transition-colors ${
                  isSelected
                    ? 'bg-sky-600/20 text-sky-300 border border-sky-500/30'
                    : 'text-slate-300 hover:bg-slate-800/60'
                }`}
              >
                <span className="truncate flex items-center gap-2">
                  <FileAudio className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                  <span className="truncate">{file}</span>
                </span>
                {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />}
              </button>
            );
          })}
        </div>

        {/* Primary Action Button */}
        <button
          onClick={() => onRunAnalysis(activeFile)}
          disabled={isRunning}
          className="w-full btn-primary py-2 px-3 text-xs flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed mt-2"
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing call...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Analyze call</span>
            </>
          )}
        </button>

        {/* Pipeline Progress Feedback */}
        {isRunning && progress && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-[11px] text-slate-400">
              <span className="capitalize">{progress.current_stage || 'Processing'}</span>
              <span>{progress.percent}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-500 transition-all duration-300"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Operator Navigation Tabs */}
      <nav className="p-3 space-y-1 flex-1">
        <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Analysis Views
        </div>

        {TABS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onSelectTab(id)}
              className={`w-full text-left px-3 py-2 rounded-md text-xs font-medium flex items-center gap-2.5 transition-colors ${
                isActive
                  ? 'bg-sky-500/15 text-sky-300 font-semibold border-l-2 border-sky-400'
                  : 'text-slate-300 hover:bg-slate-800/40 hover:text-slate-100'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-400'}`} />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
};
