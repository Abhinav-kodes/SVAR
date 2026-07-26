import React from 'react';
import type { CallData } from '../../types/dashboard';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface EmotionTabProps {
  data: CallData;
}

export const EmotionTab: React.FC<EmotionTabProps> = ({ data }) => {
  const fusion = data?.fusion || [];
  const segments = data?.segments || [];

  if (segments.length === 0) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No turn-by-turn sentiment data available.
      </div>
    );
  }

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
    };
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <h2 className="text-base font-semibold text-white">Sentiment by turn</h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Turn-by-turn customer and agent tone trajectory throughout the recording.
        </p>
      </div>

      {/* Sentiment Chart */}
      <div className="panel p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Sentiment progression</h3>

        <div className="h-56 pt-2">
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
    </div>
  );
};
