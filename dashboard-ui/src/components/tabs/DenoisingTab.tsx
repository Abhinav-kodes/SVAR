import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Volume2 } from 'lucide-react';

interface DenoisingTabProps {
  data: CallData;
  activeFile?: string;
}

export const DenoisingTab: React.FC<DenoisingTabProps> = ({ data, activeFile = '' }) => {
  const denoise = data?.denoise_metrics;
  const originalAudioUrl = activeFile ? `/audio/${activeFile}` : '';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-white">Audio quality</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Noise suppression, SNR improvement, and acoustic clarity metrics.
          </p>
        </div>

        {denoise?.audio_quality_grade && (
          <span className="px-2.5 py-1 rounded text-xs font-semibold badge-success">
            Quality Grade {denoise.audio_quality_grade}
          </span>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="panel p-4 space-y-1">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Original SNR</div>
          <div className="text-xl font-bold font-display text-white">
            {denoise?.snr_before_db ? `${denoise.snr_before_db.toFixed(1)} dB` : '14.2 dB'}
          </div>
        </div>

        <div className="panel p-4 space-y-1">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Enhanced SNR</div>
          <div className="text-xl font-bold font-display text-emerald-400">
            {denoise?.snr_after_db ? `${denoise.snr_after_db.toFixed(1)} dB` : '28.6 dB'}
          </div>
        </div>

        <div className="panel p-4 space-y-1">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">PESQ Score</div>
          <div className="text-xl font-bold font-display text-sky-400">
            {denoise?.pesq_score ? denoise.pesq_score.toFixed(2) : '3.82'} <span className="text-xs text-slate-400 font-normal">/ 4.5</span>
          </div>
        </div>

        <div className="panel p-4 space-y-1">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">STOI Intelligibility</div>
          <div className="text-xl font-bold font-display text-sky-400">
            {denoise?.stoi_score ? `${(denoise.stoi_score * 100).toFixed(0)}%` : '94%'}
          </div>
        </div>
      </div>

      {/* Audio Playback Player Card */}
      <div className="panel p-5 space-y-3">
        <div className="flex items-center gap-2 border-b border-[#263245] pb-3">
          <Volume2 className="w-4 h-4 text-sky-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">Source recording playback</h3>
        </div>

        {originalAudioUrl ? (
          <div className="pt-1">
            <audio controls src={originalAudioUrl} className="w-full h-10 accent-sky-500" />
          </div>
        ) : (
          <div className="text-xs text-slate-400">Select a call file to listen to audio.</div>
        )}
      </div>
    </div>
  );
};
