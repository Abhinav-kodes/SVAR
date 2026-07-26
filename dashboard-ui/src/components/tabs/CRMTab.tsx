import React from 'react';
import type { CallData } from '../../types/dashboard';
import { Copy } from 'lucide-react';

interface CRMTabProps {
  data: CallData;
}

export const CRMTab: React.FC<CRMTabProps> = ({ data }) => {
  const crm = data?.crm_note;

  if (!crm) {
    return (
      <div className="card text-center text-[13px]" style={{ color: 'var(--text-secondary)', padding: '40px' }}>
        No CRM note generated for this call.
      </div>
    );
  }

  const handleCopy = () => {
    const lines = [
      `Executive Summary: ${crm.summary || ''}`,
      '',
      'Key Discussion Points:',
      ...(crm.key_points || []).map((p, i) => `${i + 1}. ${p}`),
      '',
      `Compliance Overview: ${crm.compliance_summary || ''}`,
      '',
      `Recommended Action: ${crm.recommended_action || ''}`,
    ];
    navigator.clipboard.writeText(lines.join('\n')).catch(console.error);
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="card">
        <div className="flex items-start justify-between gap-5">
          <div>
            <h2 className="text-[15.5px] font-semibold">CRM note</h2>
            <div className="text-[12.5px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              Structured call audit documentation, ready for CRM attachment.
            </div>
          </div>
          <button className="btn-ghost" onClick={handleCopy}>
            <Copy className="w-[13px] h-[13px]" strokeWidth={1.6} />
            Copy to CRM
          </button>
        </div>
        <div className="tick-divider" />

        {/* Executive Summary */}
        <div className="eyebrow mb-2">Executive summary</div>
        <p className="text-[13.5px] leading-[1.7] mb-5" style={{ color: 'var(--text-secondary)' }}>
          {crm.summary || 'No summary available.'}
        </p>

        {/* Key Discussion Points */}
        {crm.key_points && crm.key_points.length > 0 && (
          <>
            <div className="eyebrow mb-2.5">Key discussion points</div>
            <ul className="flex flex-col gap-2 list-none mb-5">
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

        {/* Compliance Overview */}
        {crm.compliance_summary && (
          <>
            <div className="eyebrow mb-2">Compliance overview</div>
            <p className="text-[13.5px] leading-[1.7] mb-5" style={{ color: 'var(--text-secondary)' }}>
              {crm.compliance_summary}
            </p>
          </>
        )}

        {/* Recommended Action */}
        {crm.recommended_action && (
          <>
            <div className="eyebrow mb-2">Recommended action</div>
            <div className="rec-block">
              {crm.recommended_action}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
