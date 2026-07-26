import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { AudioPlayerBar } from './components/AudioPlayerBar';
import { SummaryTab } from './components/tabs/SummaryTab';
import { DenoisingTab } from './components/tabs/DenoisingTab';
import { DiarizationTab } from './components/tabs/DiarizationTab';
import { TranscriptTab } from './components/tabs/TranscriptTab';
import { EmotionTab } from './components/tabs/EmotionTab';
import { ComplianceTab } from './components/tabs/ComplianceTab';
import { QAScoreTab } from './components/tabs/QAScoreTab';
import { CRMTab } from './components/tabs/CRMTab';
import { ErrorBoundary } from './components/ErrorBoundary';
import type { TabId, CallData, ProgressState } from './types/dashboard';


const TAB_KEYS: Record<TabId, (keyof CallData)[]> = {
  summary: ['duration_s', 'processing_time_s', 'segments', 'talk_ratio', 'denoise_metrics', 'qa', 'compliance', 'fusion'],
  denoising: ['denoise_metrics'],
  diarization: ['segments', 'talk_ratio', 'role_resolution'],
  transcript: ['segments'],
  emotion: ['segments', 'fusion'],
  compliance: ['compliance'],
  qascore: ['qa'],
  crm: ['crm_note', 'qa', 'compliance', 'fusion'],
};

export const App: React.FC = () => {
  const [files, setFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState<string>('');
  const [activeTab, setActiveTab] = useState<TabId>('summary');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [callData, setCallData] = useState<CallData>({});
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [seekTime, setSeekTime] = useState<number | null>(null);

  // Fetch list of available sample call files
  useEffect(() => {
    fetch('/api/sample_calls')
      .then((res) => res.json())
      .then((data: string[]) => {
        setFiles(data);
        if (data.length > 0) {
          setActiveFile(data[0]);
        }
      })
      .catch((err) => console.error('Failed to load sample calls:', err));
  }, []);

  // Fetch complete call analysis in a single unified API request
  const fetchAllResults = useCallback(async (filename: string) => {
    try {
      const res = await fetch('/api/results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (res.ok) {
        const data: CallData = await res.json();
        setCallData(data);
      }
    } catch (err) {
      console.error('Error fetching call results:', err);
    }
  }, []);


  // Poll progress
  const pollProgress = useCallback(
    async (filename: string) => {
      try {
        const res = await fetch(`/api/progress?file=${encodeURIComponent(filename)}`);
        if (!res.ok) return;
        const p: ProgressState = await res.json();
        setProgress(p);

        if (p.status === 'completed' || p.status === 'error') {
          setIsAnalyzing(false);
          if (p.status === 'completed') {
            await fetchAllResults(filename);
          }
        }
      } catch (err) {
        console.error('Error polling progress:', err);
      }
    },
    [fetchAllResults]
  );

  // Poll loop effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (isAnalyzing && activeFile) {
      interval = setInterval(() => {
        pollProgress(activeFile);
      }, 1500);
      pollProgress(activeFile);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAnalyzing, activeFile, pollProgress]);

  // Handle call selection change
  const handleSelectFile = (filename: string) => {
    setActiveFile(filename);
    setCallData({});
    setProgress(null);
    setIsAnalyzing(false);
    setActiveTab('summary');
    setCurrentTime(0);
    setSeekTime(null);
  };

  // Start analysis
  const handleStartAnalysis = async () => {
    if (!activeFile || isAnalyzing) return;
    setIsAnalyzing(true);
    setCallData({});
    setProgress({ status: 'running', current_stage: 'Starting...', percent: 0, stages: {} });
    setActiveTab('summary');

    try {
      await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: activeFile }),
      });
    } catch (err) {
      console.error('Failed to trigger analysis:', err);
      setIsAnalyzing(false);
    }
  };

  const hasDataForTab = (tabId: TabId) => {
    const keys = TAB_KEYS[tabId] || [];
    return keys.some((k) => {
      const val = callData[k];
      if (val == null) return false;
      if (Array.isArray(val)) return val.length > 0;
      if (typeof val === 'object') return Object.keys(val).length > 0;
      return true;
    });
  };

  const handleSeekAudio = (timeSec: number) => {
    setSeekTime(timeSec);
  };

  const audioUrl = activeFile ? `/audio/${activeFile}` : '';

  return (
    <div className="min-h-screen bg-dark-900 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Header */}
      <Header
        activeFile={activeFile}
        processingTime={callData.processing_time_s}
        isAnalyzing={isAnalyzing}
      />

      {/* Main Layout Container */}
      <div className="flex flex-1 overflow-hidden pb-16">
        {/* Sidebar */}
        <Sidebar
          files={files}
          activeFile={activeFile}
          onSelectFile={handleSelectFile}
          onStartAnalysis={handleStartAnalysis}
          isAnalyzing={isAnalyzing}
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          progress={progress}
          hasDataForTab={hasDataForTab}
        />

        {/* Main Content Workspace */}
        <main className="flex-1 p-6 overflow-y-auto max-w-7xl mx-auto w-full">
          <ErrorBoundary key={activeTab}>
            {activeTab === 'summary' && <SummaryTab data={callData} />}
            {activeTab === 'denoising' && <DenoisingTab data={callData} activeFile={activeFile} />}
            {activeTab === 'diarization' && (
              <DiarizationTab data={callData} onSeekAudio={handleSeekAudio} currentTime={currentTime} />
            )}
            {activeTab === 'transcript' && (
              <TranscriptTab data={callData} onSeekAudio={handleSeekAudio} currentTime={currentTime} />
            )}
            {activeTab === 'emotion' && <EmotionTab data={callData} />}
            {activeTab === 'compliance' && (
              <ComplianceTab data={callData} onSeekAudio={handleSeekAudio} />
            )}
            {activeTab === 'qascore' && <QAScoreTab data={callData} />}
            {activeTab === 'crm' && <CRMTab data={callData} />}
          </ErrorBoundary>
        </main>
      </div>

      {/* Floating Bottom Audio Player */}
      <AudioPlayerBar
        audioUrl={audioUrl}
        onTimeUpdate={setCurrentTime}
        seekTime={seekTime}
      />
    </div>
  );
};
