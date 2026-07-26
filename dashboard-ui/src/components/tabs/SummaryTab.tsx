import React from 'react';
import type { CallData } from '../../types/dashboard';
import { ShieldAlert, ShieldCheck, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface SummaryTabProps {
  data: CallData;
}

export const SummaryTab: React.FC<SummaryTabProps> = ({ data }) => {
  const qa = data?.qa;
  const compliance = data?.compliance;
  const crm = data?.crm_note;
  const talkRatio = data?.talk_ratio;

  const totalViolations = compliance?.total_violations ?? 0;
  const isCompliant = compliance?.compliant ?? true;
  const qaScore = qa?.qa_score ?? 0;
  const grade = qa?.grade ?? 'N/A';

  // Quality component chart data
  const components = qa?.components || {};
  const chartData = [
    { name: 'Compliance', score: components.compliance ?? 100 },
    { name: 'Sentiment', score: components.customer_sentiment ?? 70 },
    { name: 'Agent tone', score: components.agent_stability ?? 80 },
    { name: 'Resolution', score: components.intent_resolution ?? 75 },
    { name: 'Talk ratio', score: components.talk_ratio ?? 85 },
  ];

  return (
    <div className="space-y-6">
      {/* Risk Banner & Executive Status */}
      <div className="panel p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          {totalViolations > 0 ? (
            <div className="w-10 h-10 rounded-md bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center flex-shrink-0">
              <ShieldAlert className="w-5 h-5" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
          )}

          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base font-semibold text-white">
                {totalViolations > 0 ? 'High risk call detected' : 'Standard compliant call'}
              </h2>
              <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${totalViolations > 0 ? 'badge-danger' : 'badge-success'}`}>
                {isCompliant ? 'Compliant' : 'Non-compliant'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              {totalViolations > 0
                ? `${totalViolations} policy violation(s) flagged during recording.`
                : 'No regulatory or policy violations detected.'}
            </p>
          </div>
        </div>

        {/* Quick Action Recommendation */}
        {crm?.recommended_action && (
          <div className="p-3 rounded-md bg-slate-900/60 border border-slate-800 text-xs text-slate-300 max-w-sm flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-200">Recommended action: </span>
              {crm.recommended_action}
            </div>
          </div>
        )}
      </div>

      {/* Metrics & Reasoning Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Quality Scorecard Summary */}
        <div className="panel p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#263245] pb-3">
            <h3 className="text-sm font-semibold text-white">Call quality review</h3>
            <span className="text-xs text-slate-400">Grade <strong className="text-slate-100 font-bold">{grade}</strong></span>
          </div>

          <div className="flex items-baseline gap-3">
            <div className="text-4xl font-bold font-display text-white">{qaScore.toFixed(0)}</div>
            <div className="text-xs text-slate-400">/ 100 overall score</div>
          </div>

          {/* Component Scores Bar Chart */}
          <div className="h-44 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                <XAxis type="number" domain={[0, 100]} hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#91a0b5', fontSize: 11 }} width={75} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#121a26', borderColor: '#263245', borderRadius: '6px', fontSize: '12px' }}
                  formatter={(val: any) => [`${val}/100`, 'Score']}
                />
                <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={12}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.score < 60 ? '#ef4444' : entry.score < 80 ? '#f59e0b' : '#0284c7'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Middle Column: Executive Narrative Summary */}
        <div className="panel p-5 space-y-3 lg:col-span-2 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white border-b border-[#263245] pb-3 mb-3">
              Executive call narrative
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              {crm?.summary || 'Call analysis processing complete. Detailed transcript and turn details are available.'}
            </p>

            {crm?.key_points && crm.key_points.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Key discussion points</h4>
                <ul className="space-y-1.5 text-xs text-slate-300">
                  {crm.key_points.map((pt, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-sky-400 mt-1.5 flex-shrink-0" />
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Talk Ratio Statistics */}
          <div className="pt-4 border-t border-[#263245] grid grid-cols-2 gap-4 text-xs text-slate-400">
            <div>
              <span>Agent talk ratio: </span>
              <strong className="text-slate-200">{((talkRatio?.agent_ratio ?? 0.5) * 100).toFixed(0)}%</strong>
            </div>
            <div>
              <span>Customer talk ratio: </span>
              <strong className="text-slate-200">{((talkRatio?.customer_ratio ?? 0.5) * 100).toFixed(0)}%</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
