import React from 'react';
import type { CallData } from '../../types/dashboard';


interface EmotionTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
  currentTime?: number;
}

export const EmotionTab: React.FC<EmotionTabProps> = ({ data, onSeekAudio, currentTime = 0 }) => {
  const fusion = data?.fusion || [];
  const segments = data?.segments || [];

  if (segments.length === 0) {
    return (
      <div className="card text-center text-[13px]" style={{ color: 'var(--text-secondary)', padding: '40px' }}>
        No emotion or sentiment analysis data available.
      </div>
    );
  }

  const customerSentiments = fusion.map((f, idx) => {
    const isCustomer = (segments[idx]?.speaker || '').toLowerCase().includes('customer');
    return isCustomer ? f.sentiment : null;
  }).filter(Boolean);

  const negCount = customerSentiments.filter((s) => s === 'negative').length;
  const posCount = customerSentiments.filter((s) => s === 'positive').length;
  const neuCount = customerSentiments.filter((s) => s === 'neutral').length;

  const dominant = negCount > posCount && negCount > neuCount ? 'Negative' : posCount > negCount ? 'Positive' : 'Neutral';

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between gap-5">
          <div>
            <h2 className="text-[15.5px] font-semibold">
              Emotion & sentiment analysis
              <span className={`tag ml-2 ${dominant === 'Negative' ? 'tag-negative' : dominant === 'Positive' ? 'tag-positive' : 'tag-neutral'}`}>
                Dominant tone: {dominant}
              </span>
            </h2>
            <div className="text-[12.5px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              Turn-by-turn text emotion and acoustic speech prosody evaluation.
            </div>
          </div>

          <div className="flex gap-5 flex-shrink-0">
            <div className="text-right">
              <span className="num text-[18px] font-semibold block" style={{ color: 'var(--red-strong)' }}>{negCount}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>NEGATIVE</span>
            </div>
            <div className="text-right">
              <span className="num text-[18px] font-semibold block">{neuCount}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>NEUTRAL</span>
            </div>
            <div className="text-right">
              <span className="num text-[18px] font-semibold block" style={{ color: 'var(--green)' }}>{posCount}</span>
              <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>POSITIVE</span>
            </div>
          </div>
        </div>
        <div className="tick-divider" />

        {/* Sentiment Turn Rows */}
        <div>
          {segments.map((seg, idx) => {
            const isAgent = (seg.speaker || '').toLowerCase().includes('agent');
            const fItem = fusion[idx];
            const sentiment = fItem?.sentiment || seg.sentiment || 'neutral';
            const emotion = fItem?.emotion || seg.emotion || 'neutral';
            const isNegative = sentiment === 'negative' || emotion === 'anger' || emotion === 'sadness' || emotion === 'disgust';

            const emotionTagClass = isNegative ? 'tag-negative' : emotion === 'joy' ? 'tag-positive' : 'tag-neutral';
            const toneColor = sentiment === 'negative' ? 'var(--red-strong)' : sentiment === 'positive' ? 'var(--green)' : 'var(--text-secondary)';

            const isActive = currentTime >= seg.start_time_s && currentTime < (seg.end_time_s || seg.start_time_s + 4);

            return (
              <div
                key={idx}
                onClick={() => onSeekAudio?.(seg.start_time_s)}
                className={`trow cursor-pointer transition-all duration-150 hover:bg-[#211C15]/80 ${isNegative && !isActive ? 'trow-hi' : ''}`}
                style={isActive ? { background: 'var(--amber-dim)', margin: '0 -14px', padding: '13px 14px', borderRadius: '8px', borderBottomColor: 'transparent', outline: '1px solid var(--amber)' } : undefined}
                title="Click to play audio from this turn"
              >
                <span className={`tag ${isAgent ? 'tag-agent' : 'tag-customer'}`}>
                  {isAgent ? 'Agent' : 'Customer'}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="num text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                      {seg.start_time_s.toFixed(1)}s
                    </span>
                    <span className={`tag ${emotionTagClass}`} style={{ textTransform: 'capitalize' }}>
                      {emotion}
                    </span>
                  </div>
                  <div className="text-[13.5px] leading-relaxed">{seg.text || '(Speech turn)'}</div>
                </div>

                <div className="text-[11px] flex-shrink-0 text-right mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                  Tone<br />
                  <b className="font-medium capitalize" style={{ color: toneColor }}>{sentiment}</b>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
