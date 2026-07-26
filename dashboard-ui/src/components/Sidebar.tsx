import React from 'react';
import { 
  BarChart3, VolumeX, Users, FileText, MessageSquare, 
  Scale, Trophy, ClipboardList, Play, Loader2, CheckCircle2, FileAudio
} from 'lucide-react';
import type { TabId, ProgressState } from '../types/dashboard';

interface SidebarProps {
  files: string[];
  activeFile: string;
  onSelectFile: (filename: string) => void;
  onStartAnalysis: () => void;
  isAnalyzing: boolean;
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  progress: ProgressState | null;
  hasDataForTab: (tabId: TabId) => boolean;
}

const TABS: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
  { id: 'summary', label: 'Summary', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'denoising', label: 'Denoising', icon: <VolumeX className="w-4 h-4" /> },
  { id: 'diarization', label: 'Diarization', icon: <Users className="w-4 h-4" /> },
  { id: 'transcript', label: 'Transcript', icon: <FileText className="w-4 h-4" /> },
  { id: 'emotion', label: 'Emotion & Sentiment', icon: <MessageSquare className="w-4 h-4" /> },
  { id: 'compliance', label: 'Compliance', icon: <Scale className="w-4 h-4" /> },
  { id: 'qascore', label: 'QA Scorecard', icon: <Trophy className="w-4 h-4" /> },
  { id: 'crm', label: 'CRM Note', icon: <ClipboardList className="w-4 h-4" /> },
];

const STAGE_LABELS: Record<string, string> = {
  denoise: 'Audio Denoising',
  diarize: 'Diarization',
  stt: 'Speech-to-Text',
  acoustic: 'Acoustic Emotion',
  text_emo: 'Text Emotion',
  compliance: 'Compliance Check',
  fusion: 'Multimodal Fusion',
  qa: 'QA Scoring',
  crm: 'CRM Generation',
};

export const Sidebar: React.FC<SidebarProps> = ({
  files,
  activeFile,
  onSelectFile,
  onStartAnalysis,
  isAnalyzing,
  activeTab,
  onSelectTab,
  progress,
  hasDataForTab,
}) => {
  return (
    <aside className="w-72 min-w-[18rem] border-r border-white/10 bg-dark-800/80 backdrop-blur-xl flex flex-col h-[calc(100vh-65px)] overflow-y-auto">
      {/* Sample Calls Section */}
      <div className="p-4 border-b border-white/10 space-y-3">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>Sample Audio Recordings</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-mono">
            {files.length} Files
          </span>
        </h2>

        <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
          {files.length === 0 ? (
            <div className="text-xs text-slate-500 py-2">Loading call files...</div>
          ) : (
            files.map((file) => {
              const ext = file.split('.').pop()?.toUpperCase() || 'WAV';
              const isActive = file === activeFile;
              return (
                <button
                  key={file}
                  onClick={() => onSelectFile(file)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs flex items-center justify-between transition-all duration-150 border ${
                    isActive
                      ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300 shadow-glow-cyan font-medium'
                      : 'bg-white/5 border-transparent text-slate-300 hover:bg-white/10 hover:border-white/10'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate pr-2">
                    <FileAudio className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <span className="truncate">{file}</span>
                  </div>
                  <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-white/10 text-slate-400 flex-shrink-0">
                    {ext}
                  </span>
                </button>
              );
            })
          )}
        </div>

        {/* Analyze Button */}
        <button
          onClick={onStartAnalysis}
          disabled={!activeFile || isAnalyzing}
          className="w-full py-2.5 px-4 rounded-xl font-semibold text-xs text-white bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 shadow-glow-cyan disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none transition-all duration-200 flex items-center justify-center gap-2"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Executing Pipeline...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              <span>Run AI Analysis</span>
            </>
          )}
        </button>

        {/* Live Progress Bar */}
        {progress && (progress.status === 'running' || isAnalyzing) && (
          <div className="mt-3 p-3 rounded-xl bg-white/5 border border-white/10 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium truncate">
                {STAGE_LABELS[progress.current_stage] || progress.current_stage || 'Processing...'}
              </span>
              <span className="text-cyan-400 font-mono font-bold">{progress.percent}%</span>
            </div>
            <div className="w-full h-1.5 bg-dark-900 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 transition-all duration-300 rounded-full"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Pipeline Navigation Tabs */}
      <div className="p-3 flex-1 flex flex-col space-y-1">
        <h2 className="px-2 py-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Analysis Views
        </h2>
        <nav className="space-y-1">
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            const hasData = hasDataForTab(tab.id);

            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                disabled={isAnalyzing && tab.id !== 'summary'}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent'
                } ${isAnalyzing && tab.id !== 'summary' ? 'opacity-40 cursor-not-allowed' : ''}`}
              >
                <div className="flex items-center gap-2.5">
                  <span className={isActive ? 'text-cyan-400' : 'text-slate-400'}>
                    {tab.icon}
                  </span>
                  <span>{tab.label}</span>
                </div>

                {hasData && (
                  <span className="flex items-center text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </aside>
  );
};
