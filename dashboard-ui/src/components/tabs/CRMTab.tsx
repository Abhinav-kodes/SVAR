import React, { useState } from 'react';
import type { CallData } from '../../types/dashboard';
import { Copy, Check } from 'lucide-react';

interface CRMTabProps {
  data: CallData;
}

export const CRMTab: React.FC<CRMTabProps> = ({ data }) => {
  const crm = data?.crm_note;
  const [copied, setCopied] = useState(false);

  if (!crm) {
    return (
      <div className="panel p-8 text-center text-slate-400 text-xs">
        No CRM note available.
      </div>
    );
  }

  const { summary, key_points = [], compliance_summary, recommended_action } = crm;

  const fullText = `
CRM AUDIT NOTE
--------------
SUMMARY: ${summary || 'N/A'}

KEY POINTS:
${key_points.map((p) => `- ${p}`).join('\n')}

COMPLIANCE: ${compliance_summary || 'N/A'}
RECOMMENDED ACTION: ${recommended_action || 'N/A'}
  `.trim();

  const handleCopy = () => {
    navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-white">CRM note</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Structured call audit documentation ready for CRM attachment.
          </p>
        </div>

        <button
          onClick={handleCopy}
          className="btn-secondary px-3 py-1.5 text-xs flex items-center gap-1.5"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
          <span>{copied ? 'Copied' : 'Copy to CRM'}</span>
        </button>
      </div>

      {/* Structured Content Panel */}
      <div className="panel p-6 space-y-5">
        {/* Summary */}
        <div className="space-y-1.5">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Executive summary</h3>
          <p className="text-xs text-slate-200 leading-relaxed">{summary || 'No summary available.'}</p>
        </div>

        {/* Key Points */}
        {key_points.length > 0 && (
          <div className="space-y-2 border-t border-[#263245] pt-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Key discussion points</h3>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {key_points.map((pt, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-sky-400 mt-1.5 flex-shrink-0" />
                  <span>{pt}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Compliance Assessment */}
        <div className="space-y-1.5 border-t border-[#263245] pt-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Compliance overview</h3>
          <p className="text-xs text-slate-300">{compliance_summary || 'No compliance issues noted.'}</p>
        </div>

        {/* Recommended Action */}
        <div className="space-y-1.5 border-t border-[#263245] pt-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Recommended action</h3>
          <p className="text-xs font-medium text-sky-300">{recommended_action || 'No action required.'}</p>
        </div>
      </div>
    </div>
  );
};
