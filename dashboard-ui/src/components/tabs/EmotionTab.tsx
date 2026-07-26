import React from 'react';
import type { CallData } from '../../types/dashboard';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, Smile, Frown, Meh } from 'lucide-react';


interface EmotionTabProps {
  data: CallData;
  onSeekAudio?: (timeSeconds: number) => void;
}

export const EmotionTab: React.FC<EmotionTabProps> = ({ data, onSeekAudio }) => {
  const fusion = data?.fusion || [];
  const segments = data?.segments || [];

  if (segments.length === 0) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No emotion or sentiment analysis data available.
      </div>
    );
  }

  // Count customer sentiments
  const customerSentiments = fusion.map((f, idx) => {
    const isCustomer = (segments[idx]?.speaker || '').toLowerCase().includes('customer');
    return isCustomer ? f.sentiment : null;
  }).filter(Boolean);

  const negCount = customerSentiments.filter((s) => s === 'negative').length;
  const posCount = customerSentiments.filter((s) => s === 'positive').length;
  const neuCount = customerSentiments.filter((s) => s === 'neutral').length;

  const dominant = negCount > posCount && negCount > neuCount ? 'Negative' : posCount > negCount ? 'Positive' : 'Neutral';

  // Build sentiment trend chart
  const chartData = segments.map((seg, idx) => {
    const fusionItem = fusion[idx];
    const sentiment = fusionItem?.sentiment || seg.sentiment || 'neutral';
    const confidence = fusionItem?.confidence || seg.confidence || 0.7;

    const sentimentVal = sentiment === 'positive' ? 1.0 : sentiment === 'negative' ? -1.0 : 0.0;

    return {
      turn: idx + 1,
      speaker: seg.speaker,
      time: `${seg.start_time_s.toFixed(0)}s`,
      sentiment: sentimentVal,
      confidence: Math.round(confidence * 100),
      text: seg.text || '',
    };
  });

  return (
    <div className="space-y-6">
      {/* Overview Grid */}
      <div className="panel p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-white">Emotion & sentiment analysis</h2>
            <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${dominant === 'Negative' ? 'badge-danger' : dominant === 'Positive' ? 'badge-success' : 'badge-neutral'}`}>
              Dominant tone: {dominant}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Multimodal fusion combining DistilRoBERTa Hindi text emotion with acoustic speech prosody.
          </p>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-400">
          <div className="flex items-center gap-1.5">
            <Frown className="w-4 h-4 text-rose-400" />
            <span>Negative: <strong className="text-slate-100">{negCount}</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Meh className="w-4 h-4 text-slate-400" />
            <span>Neutral: <strong className="text-slate-100">{neuCount}</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Smile className="w-4 h-4 text-emerald-400" />
            <span>Positive: <strong className="text-slate-100">{posCount}</strong></span>
          </div>
        </div>
      </div>

      {/* Sentiment Progression Timeline Chart */}
      <div className="panel p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Sentiment progression timeline</h3>

        <div className="h-52 pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="time" stroke="#475569" tick={{ fill: '#91a0b5', fontSize: 11 }} />
              <YAxis domain={[-1.2, 1.2]} ticks={[-1, 0, 1]} tickFormatter={(val) => (val === 1 ? 'Positive' : val === -1 ? 'Negative' : 'Neutral')} stroke="#475569" tick={{ fill: '#91a0b5', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#121a26', borderColor: '#263245', borderRadius: '6px', fontSize: '12px' }}
                formatter={(val: any) => [val === 1 ? 'Positive' : val === -1 ? 'Negative' : 'Neutral', 'Sentiment']}
              />
              <Area type="monotone" dataKey="sentiment" stroke="#0284c7" fill="#0284c7" fillOpacity={0.15} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Turn-by-Turn Sentiment List */}
      <div className="panel p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Turn-by-turn sentiment details</h3>

        <div className="space-y-2 max-h-[calc(100vh-320px)] overflow-y-auto pr-1">
          {segments.map((seg, idx) => {
            const isAgent = (seg.speaker || '').toLowerCase().includes('agent');
            const fItem = fusion[idx];
            const sentiment = fItem?.sentiment || seg.sentiment || 'neutral';
            const emotion = fItem?.emotion || seg.emotion || 'neutral';
            const confidence = fItem?.confidence || seg.confidence || 0.7;

            const isNegative = sentiment === 'negative' || emotion === 'anger' || emotion === 'sadness';

            return (
              <div
                key={idx}
                className={`p-3.5 rounded-md border transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs ${
                  isNegative
                    ? 'bg-rose-500/5 border-rose-500/20'
                    : 'bg-slate-900/40 border-slate-800'
                }`}
              >
                <div className="space-y-1 truncate pr-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${isAgent ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30' : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'}`}>
                      {isAgent ? 'Agent' : 'Customer'}
                    </span>
                    <span className="text-[11px] font-mono text-slate-500">
                      @ {seg.start_time_s.toFixed(1)}s
                    </span>

                    {/* Emotion Tag */}
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold capitalize ${isNegative ? 'badge-danger' : 'badge-neutral'}`}>
                      {emotion}
                    </span>
                  </div>

                  <p className="text-xs text-slate-200 truncate">{seg.text || '(Speech turn)'}</p>
                </div>

                <div className="flex items-center gap-3 flex-shrink-0">
                  <div className="text-right text-[11px] text-slate-400">
                    <div>Tone: <strong className="capitalize text-slate-200">{sentiment}</strong></div>
                    <div>Confidence: <strong className="text-emerald-400">{(confidence * 100).toFixed(0)}%</strong></div>
                  </div>

                  <button
                    onClick={() => onSeekAudio?.(seg.start_time_s)}
                    className="btn-secondary px-2.5 py-1.5 text-xs flex items-center gap-1.5"
                    title="Play audio from this turn"
                  >
                    <Play className="w-3 h-3 fill-current text-sky-400" />
                    <span>Listen</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
