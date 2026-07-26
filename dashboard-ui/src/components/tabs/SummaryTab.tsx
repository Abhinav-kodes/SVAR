import React from 'react';
import type { CallData } from '../../types/dashboard';
import { BarChart3, Clock, Zap, ShieldCheck, ShieldAlert, Award, Layers, Users } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

interface SummaryTabProps {
  data: CallData;
}

export const SummaryTab: React.FC<SummaryTabProps> = ({ data }) => {
  if (!data || !data.segments || data.segments.length === 0) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center flex flex-col items-center justify-center min-h-[350px]">
        <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-4">
          <BarChart3 className="w-8 h-8" />
        </div>
        <h3 className="font-display text-lg font-bold text-slate-200">No Call Analysis Data</h3>
        <p className="text-slate-400 text-xs max-w-sm mt-1">
          Select a sample recording from the sidebar and click <strong className="text-cyan-400">Run AI Analysis</strong> to view real-time intelligence metrics.
        </p>
      </div>
    );
  }

  const { duration_s, processing_time_s, segments, talk_ratio, denoise_metrics, qa, compliance } = data;

  const snrBefore = denoise_metrics?.snr_before_db ?? 0;
  const snrAfter = denoise_metrics?.snr_after_db ?? 0;
  const snrGain = snrAfter - snrBefore;

  const qaComponents = qa?.components || {};
  const chartData = [
    { name: 'Sentiment', score: qaComponents.customer_sentiment || 0, color: '#06b6d4' },
    { name: 'Compliance', score: qaComponents.compliance || 0, color: '#10b981' },
    { name: 'Stability', score: qaComponents.agent_stability || 0, color: '#f59e0b' },
    { name: 'Intent', score: qaComponents.intent_resolution || 0, color: '#a855f7' },
    { name: 'Talk Ratio', score: qaComponents.talk_ratio || 0, color: '#ec4899' },
  ];

  const gradeColors: Record<string, string> = {
    A: '#10b981',
    B: '#06b6d4',
    C: '#f59e0b',
    D: '#ef4444',
    F: '#ef4444',
  };
  const gradeColor = gradeColors[qa?.grade || ''] || '#06b6d4';

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Stat Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Call Duration</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="font-display text-2xl font-bold text-white">
            {duration_s ?? '--'} <span className="text-xs text-slate-400 font-normal">sec</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Pipeline Speed</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="font-display text-2xl font-bold text-purple-400">
            {processing_time_s ?? '--'} <span className="text-xs text-slate-400 font-normal">sec</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Diarized Turns</span>
            <Layers className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="font-display text-2xl font-bold text-emerald-400">
            {segments.length} <span className="text-xs text-slate-400 font-normal">segments</span>
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>SNR Gain</span>
            <Award className="w-4 h-4 text-amber-400" />
          </div>
          <div className="font-display text-2xl font-bold text-amber-400">
            +{snrGain > 0 ? snrGain.toFixed(1) : '0'} <span className="text-xs text-slate-400 font-normal">dB</span>
          </div>
        </div>
      </div>

      {/* Main QA Score & Chart Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* QA Grade Ring Card */}
        <div className="glass-card p-6 rounded-2xl flex flex-col items-center justify-center text-center">
          <h3 className="font-display text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Executive Quality Score
          </h3>
          <div className="relative w-36 h-36 flex items-center justify-center mb-3">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="rgba(255,255,255,0.06)"
                strokeWidth="8"
              />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={gradeColor}
                strokeWidth="8"
                strokeDasharray="326.7"
                strokeDashoffset={326.7 * (1 - (qa?.qa_score || 0) / 100)}
                strokeLinecap="round"
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-4xl font-extrabold" style={{ color: gradeColor }}>
                {qa?.grade || '--'}
              </span>
              <span className="text-xs font-semibold text-slate-400 mt-0.5">
                {qa?.qa_score ?? 0} / 100 pts
              </span>
            </div>
          </div>
          <div className="text-xs text-slate-300 font-medium">
            Overall Quality Grade
          </div>
        </div>

        {/* Component Breakdown Chart */}
        <div className="glass-card p-6 rounded-2xl lg:col-span-2 space-y-4">
          <h3 className="font-display text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Evaluation Category Scores
          </h3>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#38bdf8' }}
                />
                <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Speaker Talk Split & Compliance Pill */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-5 rounded-xl space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <span className="flex items-center gap-1.5">
              <Users className="w-4 h-4 text-cyan-400" />
              Talk Time Distribution
            </span>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-cyan-400 font-medium">Agent ({talk_ratio?.agent_duration_s ?? 0}s)</span>
              <span className="text-amber-400 font-medium">Customer ({talk_ratio?.customer_duration_s ?? 0}s)</span>
            </div>
            <div className="w-full h-3 bg-dark-900 rounded-full overflow-hidden flex">
              <div
                className="bg-cyan-500 h-full transition-all duration-500"
                style={{ width: `${((talk_ratio?.agent_ratio ?? 0) * 100).toFixed(1)}%` }}
              />
              <div
                className="bg-amber-500 h-full transition-all duration-500"
                style={{ width: `${((talk_ratio?.customer_ratio ?? 0) * 100).toFixed(1)}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-slate-400">
              <span>{((talk_ratio?.agent_ratio ?? 0) * 100).toFixed(1)}% Agent</span>
              <span>{((talk_ratio?.customer_ratio ?? 0) * 100).toFixed(1)}% Customer</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-5 rounded-xl space-y-3 flex flex-col justify-between">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Regulatory Compliance Status
          </div>

          <div className="flex items-center gap-3">
            {compliance?.compliant ? (
              <div className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 w-full">
                <ShieldCheck className="w-6 h-6 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-sm">Full Regulatory Compliance</div>
                  <div className="text-xs text-emerald-500/80">No RBI/IRDAI or abusive policy flags found</div>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 w-full">
                <ShieldAlert className="w-6 h-6 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-sm">Non-Compliant Call Flagged</div>
                  <div className="text-xs text-rose-400/80">
                    {compliance?.total_violations ?? 0} violations detected in speech transcript
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
