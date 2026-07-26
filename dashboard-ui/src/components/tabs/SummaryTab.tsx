import React from 'react';
import type { CallData } from '../../types/dashboard';
import { ShieldAlert } from 'lucide-react';

interface SummaryTabProps {
  data: CallData;
}

const GaugeSVG: React.FC<{ score: number }> = ({ score }) => {
  // 0 score = 0deg (pointing left at (30,90)), 100 score = 180deg (pointing right at (150,90))
  const angle = (score / 100) * 180;
  return (
    <svg width="180" height="112" viewBox="0 0 180 112">
      <path d="M 20 90 A 70 70 0 0 1 68.4 23.4" fill="none" stroke="var(--red)" strokeWidth="9" strokeLinecap="round" opacity={0.55} />
      <path d="M 68.4 23.4 A 70 70 0 0 1 131.2 33.4" fill="none" stroke="var(--amber)" strokeWidth="9" opacity={0.3} />
      <path d="M 131.2 33.4 A 70 70 0 0 1 160 90" fill="none" stroke="var(--green)" strokeWidth="9" opacity={0.3} />
      <g style={{ transformOrigin: '90px 90px', transform: `rotate(${angle}deg)`, transition: 'transform 1s cubic-bezier(.2,.8,.2,1) .3s' }}>
        <line x1="90" y1="90" x2="30" y2="90" stroke="var(--amber-strong)" strokeWidth="2.5" strokeLinecap="round" />
      </g>
      <circle cx="90" cy="90" r="5.5" fill="var(--amber-strong)" />
    </svg>
  );
};

export const SummaryTab: React.FC<SummaryTabProps> = ({ data }) => {
  const qa = data?.qa;
  const compliance = data?.compliance;
  const crm = data?.crm_note;
  const talkRatio = data?.talk_ratio;

  const totalViolations = compliance?.total_violations ?? 0;

  const components = qa?.components || {};

  const qaScore = qa?.qa_score ?? 0;
  const grade = qa?.grade ?? 'N/A';

  const categoryLabels: Record<string, string> = {
    customer_sentiment: 'Customer sentiment',
    compliance: 'Compliance adherence',
    agent_stability: 'Agent tone',
    intent_resolution: 'Query resolution',
    talk_ratio: 'Talk ratio balance',
  };

  const agentRatio = ((talkRatio?.agent_ratio ?? 0.5) * 100).toFixed(0);
  const customerRatio = ((talkRatio?.customer_ratio ?? 0.5) * 100).toFixed(0);

  return (
    <div className="flex flex-col gap-5">
      {/* Risk Alert Banner */}
      {totalViolations > 0 && (
        <div className="alert-banner">
          <div className="flex gap-3.5 items-start">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(193,71,58,0.2)', color: 'var(--red-strong)' }}
            >
              <ShieldAlert className="w-[18px] h-[18px]" strokeWidth={1.6} />
            </div>
            <div>
              <div className="flex items-center gap-2.5 mb-1">
                <h1 className="text-[15.5px] font-semibold">High risk call detected</h1>
                <span className="badge badge-danger">Non-compliant</span>
              </div>
              <div className="text-[13px]" style={{ color: 'var(--text-secondary)' }}>
                {totalViolations} regulatory or conduct policy violation(s) flagged during this recording.
              </div>
            </div>
          </div>

          {crm?.recommended_action && (
            <div
              className="rounded-lg p-3 px-4 text-[12.5px] leading-relaxed max-w-[400px] flex-shrink-0"
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
              }}
            >
              <b style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Recommended action</b> — {crm.recommended_action}
            </div>
          )}
        </div>
      )}

      {/* Two-column: Gauge + Narrative */}
      <div className="grid gap-5" style={{ gridTemplateColumns: '400px 1fr' }}>
        {/* Left: Quality Score Gauge */}
        <div className="card">
          <div className="flex items-start justify-between mb-0.5">
            <h2 className="text-[15.5px] font-semibold">Call quality score</h2>
            <span className="eyebrow">Weighted</span>
          </div>
          <div className="tick-divider" />

          <div className="flex flex-col items-center pt-1.5 pb-0.5">
            <GaugeSVG score={qaScore} />
            <div className="flex items-baseline gap-1.5 -mt-3.5">
              <span className="num text-[40px] font-semibold" style={{ color: qaScore < 60 ? 'var(--red-strong)' : qaScore < 80 ? 'var(--amber)' : 'var(--green)', letterSpacing: '-0.02em' }}>
                {qaScore.toFixed(0)}
              </span>
              <span className="num text-sm" style={{ color: 'var(--text-tertiary)' }}>/100</span>
              <span
                className="num text-xs font-semibold ml-2 px-2 py-0.5 rounded-[5px]"
                style={{
                  background: qaScore < 60 ? 'rgba(193,71,58,0.2)' : qaScore < 80 ? 'var(--amber-dim)' : 'var(--green-dim)',
                  color: qaScore < 60 ? 'var(--red-strong)' : qaScore < 80 ? 'var(--amber-strong)' : 'var(--green)',
                }}
              >
                Grade {grade}
              </span>
            </div>
            <div className="text-[13px] font-medium mt-1.5">
              {qaScore < 60 ? 'Unacceptable quality' : qaScore < 80 ? 'Needs improvement' : 'Meets standard'}
            </div>
            <div className="text-xs text-center mt-1 max-w-[300px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {qaScore < 60
                ? 'Critical penalties applied for policy breaches and negative sentiment.'
                : 'Overall interaction met baseline call quality thresholds.'}
            </div>
          </div>

          {/* Category Breakdown */}
          <div className="flex flex-col gap-3.5 mt-[18px]">
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
                    <div
                      className={`cat-fill ${fillClass}`}
                      style={{ width: `${Math.max(1, scoreVal)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Executive Narrative */}
        <div className="card flex flex-col">
          <div className="flex items-start justify-between mb-0.5">
            <h2 className="text-[15.5px] font-semibold">Executive call narrative</h2>
          </div>
          <div className="tick-divider" />

          <p className="text-[13.5px] leading-[1.7] mb-4" style={{ color: 'var(--text-secondary)' }}>
            {crm?.summary || 'Call processing complete. Transcript and turn details are available.'}
          </p>

          {crm?.key_points && crm.key_points.length > 0 && (
            <>
              <div className="eyebrow mb-2.5">Key discussion points</div>
              <ul className="flex flex-col gap-2 mb-1 list-none">
                {crm.key_points.map((pt, idx) => (
                  <li key={idx} className="flex gap-2.5 text-[13px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                    <span
                      className="w-[5px] h-[5px] rounded-[1px] mt-[7px] flex-shrink-0"
                      style={{ background: 'var(--amber)' }}
                    />
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          <div className="tick-divider mt-auto" />

          {/* Talk Ratio */}
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <div className="flex justify-between text-xs mb-1.5">
                <span style={{ color: 'var(--text-tertiary)' }}>Talk ratio</span>
                <span className="num font-semibold">{agentRatio} / {customerRatio}</span>
              </div>
              <div className="h-1.5 rounded-[3px] overflow-hidden flex" style={{ background: 'var(--surface-3)' }}>
                <div style={{ width: `${agentRatio}%`, background: 'var(--agent)', height: '100%' }} />
                <div style={{ width: `${customerRatio}%`, background: 'var(--amber)', height: '100%' }} />
              </div>
            </div>
          </div>
          <div className="flex gap-4 mt-2.5">
            <div className="flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              <span className="w-[7px] h-[7px] rounded-[2px]" style={{ background: 'var(--agent)' }} />
              Agent
            </div>
            <div className="flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              <span className="w-[7px] h-[7px] rounded-[2px]" style={{ background: 'var(--amber)' }} />
              Customer
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
