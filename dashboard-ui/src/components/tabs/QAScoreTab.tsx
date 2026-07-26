import React from 'react';
import type { CallData } from '../../types/dashboard';

interface QAScoreTabProps {
  data: CallData;
}

export const QAScoreTab: React.FC<QAScoreTabProps> = ({ data }) => {
  const qa = data?.qa;
  const compliance = data?.compliance;

  if (!qa) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No quality review data available.
      </div>
    );
  }

  const { qa_score = 0, grade = 'N/A', components = {}, weights_used = {} } = qa;
  const totalViolations = compliance?.total_violations ?? 0;

  const categoryLabels: Record<string, string> = {
    compliance: 'Compliance adherence',
    customer_sentiment: 'Customer sentiment',
    agent_stability: 'Agent tone & professionalism',
    intent_resolution: 'Query resolution',
    talk_ratio: 'Talk ratio balance',
  };

  return (
    <div className="space-y-6">
      {/* Overview Header */}
      <div className="panel p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-white">Call quality review</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluation of conduct, sentiment, resolution, and compliance adherence.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-2xl font-bold font-display text-white">{qa_score.toFixed(0)} <span className="text-xs text-slate-400 font-normal">/ 100</span></div>
            <div className="text-xs text-slate-400">Grade <strong className="text-slate-100">{grade}</strong></div>
          </div>
        </div>
      </div>

      {/* Explanatory Assessment Card */}
      <div className="panel p-5 space-y-2">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Evaluation summary</h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          {totalViolations > 0
            ? `Score penalized due to ${totalViolations} detected compliance/policy violation(s). Immediate supervisor review recommended.`
            : qa_score >= 80
            ? 'Call met or exceeded expected quality standards with no compliance flags.'
            : 'Call completed with satisfactory compliance, but agent tone or resolution scores need attention.'}
        </p>
      </div>

      {/* Category Breakdown Table */}
      <div className="panel p-5 space-y-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Category score breakdown</h3>

        <div className="space-y-3">
          {Object.entries(components).map(([key, value]) => {
            const scoreVal = typeof value === 'number' ? value : 0;
            const weight = weights_used[key] ? `${(weights_used[key] * 100).toFixed(0)}%` : '20%';
            const label = categoryLabels[key] || key.replace(/_/g, ' ');

            return (
              <div key={key} className="space-y-1.5 border-b border-[#263245] pb-3 last:border-0 last:pb-0">
                <div className="flex justify-between text-xs">
                  <span className="font-medium text-slate-200 capitalize">{label}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400 text-[11px]">Weight: {weight}</span>
                    <span className={`font-semibold ${scoreVal < 60 ? 'text-rose-400' : scoreVal < 80 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {scoreVal.toFixed(0)} / 100
                    </span>
                  </div>
                </div>

                <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${scoreVal < 60 ? 'bg-rose-500' : scoreVal < 80 ? 'bg-amber-500' : 'bg-sky-500'}`}
                    style={{ width: `${Math.min(100, Math.max(0, scoreVal))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
