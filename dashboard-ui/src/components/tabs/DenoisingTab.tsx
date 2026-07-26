import React, { useMemo } from 'react';
import type { CallData } from '../../types/dashboard';
import { TrendingUp } from 'lucide-react';

interface DenoisingTabProps {
  data: CallData;
  activeFile?: string;
}

export const DenoisingTab: React.FC<DenoisingTabProps> = ({ data }) => {
  const denoise = data?.denoise_metrics;
  const snrBefore = denoise?.snr_before_db ?? 14.2;
  const snrAfter = denoise?.snr_after_db ?? 28.6;
  const snrDelta = (snrAfter - snrBefore).toFixed(1);

  // Generate waveform bars deterministically
  const bars = useMemo(() => {
    return Array.from({ length: 80 }).map((_, i) => {
      const pause = (i % 17) > 13;
      const h = pause ? 6 + (Math.sin(i * 1.3) + 1) * 4 : 20 + Math.abs(Math.sin(i * 0.4)) * 30 + (Math.sin(i * 0.7) + 1) * 6;
      return { height: Math.round(h), pause };
    });
  }, []);

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between gap-5">
          <div>
            <h2 className="text-[15.5px] font-semibold">Audio quality & enhancement</h2>
            <div className="text-[12.5px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              Acoustic noise suppression, SNR gain calculation, and speech intelligibility metrics.
            </div>
          </div>

          {denoise?.audio_quality_grade && (
            <span className="badge badge-pass">Grade {denoise.audio_quality_grade}</span>
          )}
        </div>
        <div className="tick-divider" />

        {/* Metric Cards Grid */}
        <div className="grid grid-cols-4 gap-3.5">
          <div className="metric-card">
            <div className="eyebrow">Original SNR</div>
            <div className="num text-[26px] font-semibold mt-1.5">
              {snrBefore.toFixed(1)}<small className="text-[13px] font-normal" style={{ color: 'var(--text-tertiary)' }}> dB</small>
            </div>
            <div className="text-[11.5px] mt-1" style={{ color: 'var(--text-tertiary)' }}>Baseline noise floor</div>
          </div>

          <div className="metric-card">
            <div className="eyebrow">Enhanced SNR</div>
            <div className="num text-[26px] font-semibold mt-1.5" style={{ color: 'var(--green)' }}>
              {snrAfter.toFixed(1)}<small className="text-[13px] font-normal" style={{ color: 'var(--text-tertiary)' }}> dB</small>
            </div>
            <div className="text-[11.5px] mt-1 flex items-center gap-1" style={{ color: 'var(--green)' }}>
              <TrendingUp className="w-3 h-3" />
              +{snrDelta} dB improvement
            </div>
          </div>

          <div className="metric-card">
            <div className="eyebrow">PESQ score</div>
            <div className="num text-[26px] font-semibold mt-1.5">
              {denoise?.pesq_score ? denoise.pesq_score.toFixed(2) : '3.82'}
              <small className="text-[13px] font-normal" style={{ color: 'var(--text-tertiary)' }}> / 4.5</small>
            </div>
            <div className="text-[11.5px] mt-1" style={{ color: 'var(--text-tertiary)' }}>Perceptual speech quality</div>
          </div>

          <div className="metric-card">
            <div className="eyebrow">STOI intelligibility</div>
            <div className="num text-[26px] font-semibold mt-1.5">
              {denoise?.stoi_score ? `${(denoise.stoi_score * 100).toFixed(0)}` : '94'}
              <small className="text-[13px] font-normal" style={{ color: 'var(--text-tertiary)' }}>%</small>
            </div>
            <div className="text-[11.5px] mt-1" style={{ color: 'var(--text-tertiary)' }}>Short-time intelligibility</div>
          </div>
        </div>

        {/* Waveform */}
        <div className="eyebrow mt-5 mb-2.5">Acoustic clarity waveform</div>
        <div
          className="flex items-end gap-[2px] h-[70px] rounded-lg p-3.5 px-4.5"
          style={{ background: 'var(--surface-1)', border: '1px solid var(--border)' }}
        >
          {bars.map((bar, i) => (
            <div
              key={i}
              className={`flex-1 rounded-[1px] min-w-[2px] ${bar.pause ? 'wf-bar-pause' : 'wf-bar'}`}
              style={{ height: `${bar.height}px` }}
            />
          ))}
        </div>

        <div className="flex justify-between mt-2.5 text-[11.5px]" style={{ color: 'var(--text-tertiary)' }}>
          <span>
            Denoising algorithm: <b className="font-medium" style={{ color: 'var(--text-secondary)' }}>Spectral subtraction + deep filter</b>
          </span>
          <span>
            <b className="font-medium" style={{ color: 'var(--text-secondary)' }}>24kHz</b> / Mono / WAV
          </span>
        </div>
      </div>
    </div>
  );
};
