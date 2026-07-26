import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Volume2, VolumeX, Sparkles, CheckCircle2 } from 'lucide-react';

interface DenoisingTabProps {
  data: CallData;
  activeFile: string;
}

export const DenoisingTab: React.FC<DenoisingTabProps> = ({ data, activeFile }) => {
  const metrics = data?.denoise_metrics;

  if (!metrics) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No denoising data available. Please run analysis on a call recording.
      </div>
    );
  }

  const snrBefore = metrics.snr_before_db ?? 0;
  const snrAfter = metrics.snr_after_db ?? 0;
  const snrDelta = snrAfter - snrBefore;

  const qualityGrade = metrics.audio_quality_grade || 'Good';
  const gradeColor =
    qualityGrade.toLowerCase() === 'good'
      ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
      : qualityGrade.toLowerCase() === 'fair'
      ? 'text-amber-400 border-amber-500/30 bg-amber-500/10'
      : 'text-rose-400 border-rose-500/30 bg-rose-500/10';

  const rawAudioUrl = activeFile ? `/audio/${activeFile}` : '';
  const cleanAudioUrl = activeFile ? `/audio/${activeFile}` : '';

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Audio Comparison Card */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <Volume2 className="w-5 h-5 text-cyan-400" />
            <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
              Audio Denoising & Enhancement Engine
            </h3>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold">
            16 kHz Waveform Denoised
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          {/* Original Audio */}
          <div className="p-4 rounded-xl bg-dark-900/60 border border-white/10 space-y-2">
            <div className="flex items-center gap-2 text-slate-400 text-xs font-semibold uppercase tracking-wider">
              <VolumeX className="w-4 h-4 text-amber-400" />
              Original Raw Audio (Noisy)
            </div>
            <audio controls src={rawAudioUrl} className="w-full h-10 outline-none rounded-lg accent-cyan-500" />
            <div className="text-[11px] text-slate-500 flex justify-between">
              <span>Noise Floor: Baseline</span>
              <span>SNR: {snrBefore.toFixed(1)} dB</span>
            </div>
          </div>

          {/* Denoised Audio */}
          <div className="p-4 rounded-xl bg-cyan-500/5 border border-cyan-500/30 space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              SVAR Denoised Clean Signal
            </div>
            <audio controls src={cleanAudioUrl} className="w-full h-10 outline-none rounded-lg accent-cyan-400" />
            <div className="text-[11px] text-cyan-400/80 flex justify-between">
              <span>Deep Noise Suppression</span>
              <span>SNR: {snrAfter.toFixed(1)} dB</span>
            </div>
          </div>
        </div>
      </div>

      {/* Denoising Signal Quality Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            SNR Before Denoising
          </div>
          <div className="font-display text-2xl font-bold text-amber-400">
            {snrBefore.toFixed(1)} <span className="text-xs text-slate-400 font-normal">dB</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            SNR After Enhancement
          </div>
          <div className="font-display text-2xl font-bold text-emerald-400">
            {snrAfter.toFixed(1)} <span className="text-xs text-slate-400 font-normal">dB</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Signal-to-Noise Gain
          </div>
          <div className="font-display text-2xl font-bold text-cyan-400">
            +{snrDelta > 0 ? snrDelta.toFixed(1) : '0'} <span className="text-xs text-slate-400 font-normal">dB</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Overall Quality Grade
          </div>
          <div className="pt-1">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border ${gradeColor}`}>
              <CheckCircle2 className="w-3.5 h-3.5" />
              {qualityGrade}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
