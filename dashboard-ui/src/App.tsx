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


  // Live progress over WebSocket (replaces 1.5s polling)
  useEffect(() => {
    if (!isAnalyzing || !activeFile) return;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let tornDown = false;
    let receivedTerminal = false;

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${proto}//${location.host}/ws/progress?file=${encodeURIComponent(activeFile)}`);
      ws.onmessage = (ev) => {
        try {
          const p: ProgressState = JSON.parse(ev.data);
          setProgress(p);
          if (p.status === 'completed' || p.status === 'error') {
            receivedTerminal = true;
            setIsAnalyzing(false);
            if (p.status === 'completed') {
              fetchAllResults(activeFile);
            }
          }
        } catch (err) {
          console.error('Error parsing progress message:', err);
        }
      };
      ws.onclose = () => {
        if (!tornDown && !receivedTerminal) {
          timer = setTimeout(connect, 1000);
        }
      };
    };
    connect();

    return () => {
      tornDown = true;
      if (timer) clearTimeout(timer);
      if (ws) ws.close();
    };
  }, [isAnalyzing, activeFile, fetchAllResults]);

  // Handle call selection change
  const handleSelectFile = (filename: string) => {
    setActiveFile(filename);
    setCallData({});
    setProgress(null);
    setIsAnalyzing(false);
    setActiveTab('summary');
    setCurrentTime(0);
    setSeekTime(null);
    fetchAllResults(filename);
  };

  // Start analysis
  const handleStartAnalysis = async () => {
    if (!activeFile || isAnalyzing) return;
    setIsAnalyzing(true);
    setCallData({});
    setProgress({ status: 'running', current_stage: 'Starting...', percent: 0, stages: {} });
    setActiveTab('summary');

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: activeFile }),
      });
      const body = await res.json();
      if (body.status === 'completed') {
        setIsAnalyzing(false);
        fetchAllResults(activeFile);
      }
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
        <main className="flex-1 overflow-y-auto w-full max-w-[1240px] mx-auto px-4 sm:px-7 py-6 pb-24">
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
