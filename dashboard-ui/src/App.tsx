import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { AudioPlayerBar } from './components/AudioPlayerBar';
import { SummaryTab } from './components/tabs/SummaryTab';
import { DenoisingTab } from './components/tabs/DenoisingTab';
import { TranscriptTab } from './components/tabs/TranscriptTab';

import { EmotionTab } from './components/tabs/EmotionTab';
import { ComplianceTab } from './components/tabs/ComplianceTab';
import { QAScoreTab } from './components/tabs/QAScoreTab';
import { CRMTab } from './components/tabs/CRMTab';
import { ErrorBoundary } from './components/ErrorBoundary';
import type { TabId, CallData, ProgressState } from './types/dashboard';


import type { SeekRequest } from './components/AudioPlayerBar';

export const App: React.FC = () => {
  const [files, setFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState<string>('');
  const [activeTab, setActiveTab] = useState<TabId>('summary');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [callData, setCallData] = useState<CallData>({});
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [seekTime, setSeekTime] = useState<SeekRequest | null>(null);

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



  const handleSeekAudio = (timeSec: number) => {
    setSeekTime({ time: timeSec, id: Date.now() });
  };

  const audioUrl = activeFile ? `/audio/${activeFile}` : '';

  return (
    <div className="flex min-h-screen" style={{ background: 'var(--bg)', color: 'var(--text-primary)', fontFamily: 'var(--font-ui)' }}>
      {/* Sidebar */}
      <Sidebar
        files={files}
        activeFile={activeFile}
        onSelectFile={handleSelectFile}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onRunAnalysis={handleStartAnalysis}
        progress={progress}
      />

      {/* Main Area */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Topbar */}
        <Header activeFile={activeFile} callData={callData} />

        {/* Content */}
        <main className="flex-1 overflow-y-auto" style={{ padding: '26px 28px 90px', maxWidth: '1240px' }}>
          <ErrorBoundary key={activeTab}>
            {activeTab === 'summary' && <SummaryTab data={callData} />}
            {activeTab === 'denoising' && <DenoisingTab data={callData} activeFile={activeFile} />}
            {activeTab === 'transcript' && (
              <TranscriptTab data={callData} onSeekAudio={handleSeekAudio} currentTime={currentTime} />
            )}
            {activeTab === 'emotion' && (
              <EmotionTab data={callData} onSeekAudio={handleSeekAudio} currentTime={currentTime} />
            )}
            {activeTab === 'compliance' && (
              <ComplianceTab data={callData} onSeekAudio={handleSeekAudio} currentTime={currentTime} />
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
