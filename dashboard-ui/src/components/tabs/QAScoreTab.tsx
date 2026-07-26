import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Trophy } from 'lucide-react';

interface QAScoreTabProps {
  data: CallData;
}

export const QAScoreTab: React.FC<QAScoreTabProps> = ({ data }) => {
  const qa = data?.qa;

  if (!qa) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No QA evaluation score data available.
      </div>
    );
  }

  const { qa_score, grade, components = {}, weights_used = {} } = qa;

  const categoryLabels: Record<string, { name: string; color: string }> = {
    customer_sentiment: { name: 'Customer Sentiment & Satisfaction', color: '#06b6d4' },
    compliance: { name: 'Regulatory & Scripts Compliance', color: '#10b981' },
    agent_stability: { name: 'Agent Emotional Stability', color: '#f59e0b' },
    intent_resolution: { name: 'Customer Intent Resolution', color: '#a855f7' },
    talk_ratio: { name: 'Talk-to-Listen Ratio', color: '#ec4899' },
  };

  const gradeColors: Record<string, string> = {
    A: '#10b981',
    B: '#06b6d4',
    C: '#f59e0b',
    D: '#ef4444',
    F: '#ef4444',
  };
  const gradeColor = gradeColors[grade] || '#06b6d4';

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* QA Score Header Card */}
      <div className="glass-card p-6 rounded-2xl space-y-6">
        <div className="flex items-center gap-2 border-b border-white/10 pb-3">
          <Trophy className="w-5 h-5 text-amber-400" />
          <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
            Automated Call Quality Audit Scorecard
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Grade Ring */}
          <div className="flex flex-col items-center justify-center p-6 rounded-xl bg-dark-900/60 border border-white/10">
            <div className="relative w-40 h-40 flex items-center justify-center">
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
                  strokeDashoffset={326.7 * (1 - (qa_score || 0) / 100)}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="font-display text-5xl font-extrabold" style={{ color: gradeColor }}>
                  {grade}
                </span>
                <span className="text-xs font-semibold text-slate-400 mt-1">
                  {qa_score} / 100 pts
                </span>
              </div>
            </div>
            <div className="text-xs text-slate-300 font-semibold mt-3">
              Audited Call Quality Grade
            </div>
          </div>

          {/* Component Bar List */}
          <div className="md:col-span-2 space-y-4">
            {Object.entries(components).map(([key, val]) => {
              const meta = categoryLabels[key] || { name: key, color: '#06b6d4' };
              const weight = weights_used[key] != null ? `${(weights_used[key] * 100).toFixed(0)}%` : '';

              return (
                <div key={key} className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300 font-medium flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
                      {meta.name}
                      {weight && <span className="text-slate-500 font-mono">({weight} weight)</span>}
                    </span>
                    <span className="font-mono font-bold" style={{ color: meta.color }}>
                      {val} pts
                    </span>
                  </div>

                  <div className="w-full h-2.5 bg-dark-900 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${val}%`, backgroundColor: meta.color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
