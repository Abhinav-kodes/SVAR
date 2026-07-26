import React from 'react';
import type { CallData } from '../../types/dashboard';
import { MessageSquare, Sparkles } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';

interface EmotionTabProps {
  data: CallData;
}

export const EmotionTab: React.FC<EmotionTabProps> = ({ data }) => {
  const segments = data?.segments || [];
  const fusion = data?.fusion || [];

  if (fusion.length === 0) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No multimodal emotion data available.
      </div>
    );
  }

  const emotionChipStyles: Record<string, string> = {
    neutral: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    anger: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    sad: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    fear: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    disgusted: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    surprised: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
    joy: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    annoyed: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
    confident: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    grateful: 'bg-teal-500/10 text-teal-400 border-teal-500/30',
    uncertain: 'bg-slate-500/10 text-slate-400 border-dashed border-slate-500/30',
  };

  // Recharts timeline series data
  const chartData = segments.map((seg, idx) => {
    const f = fusion[idx] || {};
    const isPositive = ['joy', 'confident', 'grateful', 'hopeful', 'impressed'].includes(f.emotion?.toLowerCase() || '');
    const isNegative = ['anger', 'sad', 'fear', 'annoyed', 'disgusted'].includes(f.emotion?.toLowerCase() || '');

    return {
      time: `${seg.start_time_s.toFixed(0)}s`,
      confidence: Math.round((f.confidence || 0) * 100),
      sentimentVal: isPositive ? 1 : isNegative ? -1 : 0,
      speaker: seg.speaker === 'agent' || seg.speaker === 'spk_0' ? 'Agent' : 'Customer',
    };
  });

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Emotion Sentiment Progression Chart */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
              Multimodal Fusion Confidence Timeline
            </h3>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/30 font-medium">
            Text NLP + Acoustic Prosody Fused
          </span>
        </div>

        <div className="h-44 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                itemStyle={{ color: '#c084fc' }}
              />
              <Area
                type="monotone"
                dataKey="confidence"
                stroke="#a855f7"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorConfidence)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment Multimodal Emotion Table */}
      <div className="glass-card p-6 rounded-2xl space-y-4">
        <div className="flex items-center gap-2 border-b border-white/10 pb-3">
          <MessageSquare className="w-5 h-5 text-cyan-400" />
          <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
            Turn-by-Turn Fused Emotion Results
          </h3>
        </div>

        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-left text-xs">
            <thead className="bg-white/5 text-slate-400 uppercase font-semibold text-[11px]">
              <tr>
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">Speaker</th>
                <th className="px-4 py-3">Emotion Category</th>
                <th className="px-4 py-3">Sentiment</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Fusion Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {segments.map((seg, idx) => {
                const f = fusion[idx] || {};
                const emotionKey = f.emotion?.toLowerCase() || 'neutral';
                const styleClass = emotionChipStyles[emotionKey] || emotionChipStyles.neutral;
                const isAgent = seg.speaker === 'agent' || seg.speaker === 'spk_0';

                return (
                  <tr key={idx} className="hover:bg-white/5 text-slate-300 transition-colors">
                    <td className="px-4 py-3 text-slate-500 font-mono">{idx + 1}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                          isAgent
                            ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                            : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                        }`}
                      >
                        {isAgent ? 'Agent' : 'Customer'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-semibold uppercase border ${styleClass}`}>
                        {f.emotion || 'Neutral'}
                      </span>
                    </td>
                    <td className="px-4 py-3 capitalize font-medium text-slate-200">
                      {f.sentiment || 'neutral'}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {((f.confidence || 0) * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-3 text-slate-400 text-[11px] font-mono">
                      {f.source || 'multimodal'} {f.text_weight != null ? `(tw:${f.text_weight})` : ''}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
