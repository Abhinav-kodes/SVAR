import React from 'react';
import type { CallData } from '../../types/dashboard';



interface QAScoreTabProps {
  data: CallData;
}

const GaugeSVG: React.FC<{ score: number }> = ({ score }) => {
  const angle = (score / 100) * 180 - 180;
  return (
    <svg width="200" height="124" viewBox="0 0 180 112">
      <path d="M 20 90 A 70 70 0 0 1 68.4 23.4" fill="none" stroke="var(--red)" strokeWidth="9" strokeLinecap="round" opacity={0.55} />
      <path d="M 68.4 23.4 A 70 70 0 0 1 131.2 33.4" fill="none" stroke="var(--amber)" strokeWidth="9" opacity={0.3} />
      <path d="M 131.2 33.4 A 70 70 0 0 1 160 90" fill="none" stroke="var(--green)" strokeWidth="9" opacity={0.3} />
      <g style={{ transformOrigin: '90px 90px', transform: `rotate(${angle}deg)`, transition: 'transform 1s cubic-bezier(.2,.8,.2,1) .3s' }}>
        <line x1="90" y1="90" x2="30" y2="79" stroke="var(--amber-strong)" strokeWidth="2.5" strokeLinecap="round" />
      </g>
      <circle cx="90" cy="90" r="5.5" fill="var(--amber-strong)" />
    </svg>
  );
};

export const QAScoreTab: React.FC<QAScoreTabProps> = ({ data }) => {
  const qa = data?.qa;
  const compliance = data?.compliance;

  if (!qa) {
    return (
      <div className="card text-center text-[13px]" style={{ color: 'var(--text-secondary)', padding: '40px' }}>
        No quality review data available.
      </div>
    );
  }

  const { qa_score = 0, grade = 'N/A', components = {} } = qa;
  const totalViolations = compliance?.total_violations ?? 0;

  const categoryLabels: Record<string, string> = {
    customer_sentiment: 'Customer sentiment',
    compliance: 'Compliance adherence',
    agent_stability: 'Agent tone & professionalism',
    intent_resolution: 'Query resolution',
    talk_ratio: 'Talk ratio balance',
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between mb-0.5 gap-5">
          <h2 className="text-[15.5px] font-semibold">Call quality review</h2>
          <span className="text-[12.5px]" style={{ color: 'var(--text-secondary)' }}>
            Weighted scorecard evaluating conduct, sentiment, resolution, and compliance rules.
          </span>
        </div>
        <div className="tick-divider" />

        {/* Gauge */}
        <div className="flex flex-col items-center pt-1.5 pb-0.5">
          <GaugeSVG score={qa_score} />
          <div className="flex items-baseline gap-1.5 -mt-3.5">
            <span className="num text-[40px] font-semibold" style={{ color: qa_score < 60 ? 'var(--red-strong)' : qa_score < 80 ? 'var(--amber)' : 'var(--green)', letterSpacing: '-0.02em' }}>
              {qa_score.toFixed(0)}
            </span>
            <span className="num text-sm" style={{ color: 'var(--text-tertiary)' }}>/100</span>
            <span
              className="num text-xs font-semibold ml-2 px-2 py-0.5 rounded-[5px]"
              style={{
                background: qa_score < 60 ? 'rgba(193,71,58,0.2)' : qa_score < 80 ? 'var(--amber-dim)' : 'var(--green-dim)',
                color: qa_score < 60 ? 'var(--red-strong)' : qa_score < 80 ? 'var(--amber-strong)' : 'var(--green)',
              }}
            >
              Grade {grade}
            </span>
          </div>
        </div>

        {/* Evaluation Summary */}
        <div
          className="rounded-r-lg p-3.5 px-4 text-[13px] leading-relaxed my-4"
          style={{
            borderLeft: `3px solid ${totalViolations > 0 ? 'var(--red-strong)' : 'var(--amber)'}`,
            background: totalViolations > 0 ? 'var(--red-dim)' : 'var(--amber-dim)',
            color: 'var(--text-secondary)',
          }}
        >
          <b style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Evaluation summary</b> —{' '}
          {totalViolations > 0
            ? `Call penalized for ${totalViolations} detected compliance and policy violations. Immediate supervisor review recommended.`
            : qa_score >= 80
            ? 'Call met or exceeded expected quality standards.'
            : 'Call completed with satisfactory compliance but agent tone or resolution require review.'}
        </div>

        {/* Category Breakdown */}
        <div className="eyebrow mb-3">Category score breakdown</div>
        <div className="flex flex-col gap-3.5">
          {Object.entries(components).map(([key, value]) => {
            const scoreVal = typeof value === 'number' ? value : 0;
            const label = categoryLabels[key] || key.replace(/_/g, ' ');
            const fillClass = scoreVal < 60 ? 'cat-fill-red' : scoreVal < 80 ? 'cat-fill-amber' : 'cat-fill-green';

            return (
              <div key={key}>
                <div className="flex justify-between items-baseline mb-1.5 gap-2">
                  <span className="text-[12.5px] font-medium capitalize">{label}</span>
                  <span
                    className="num text-[12.5px] font-semibold"
                    style={{ color: scoreVal < 60 ? 'var(--red-strong)' : scoreVal < 80 ? 'var(--amber)' : 'var(--green)' }}
                  >
                    {scoreVal.toFixed(0)}/100
                  </span>
                </div>
                <div className="cat-track">
                  <div className={`cat-fill ${fillClass}`} style={{ width: `${Math.max(1, scoreVal)}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
