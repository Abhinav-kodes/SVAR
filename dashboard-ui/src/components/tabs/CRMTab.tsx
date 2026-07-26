import React, { useState } from 'react';
import type { CallData } from '../../types/dashboard';
import { ClipboardList, Check, Copy, Sparkles, AlertCircle, ArrowRight } from 'lucide-react';

interface CRMTabProps {
  data: CallData;
}

export const CRMTab: React.FC<CRMTabProps> = ({ data }) => {
  const crm = data?.crm_note;
  const [copied, setCopied] = useState(false);

  if (!crm) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center text-slate-400">
        No auto-generated CRM note available.
      </div>
    );
  }

  const { summary, key_points = [], compliance_summary, recommended_action } = crm;

  const handleCopyCRM = () => {
    const formattedText = `--- SVAR CRM AUDIT NOTE ---
SUMMARY: ${summary || 'N/A'}

KEY POINTS:
${key_points.map((p) => `- ${p}`).join('\n')}

COMPLIANCE OVERVIEW: ${compliance_summary || 'N/A'}

RECOMMENDED ACTION: ${recommended_action || 'N/A'}
---------------------------`;

    navigator.clipboard.writeText(formattedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* CRM Card */}
      <div className="glass-card p-6 rounded-2xl space-y-6">
        <div className="flex items-center justify-between border-b border-white/10 pb-4">
          <div className="flex items-center gap-2">
            <ClipboardList className="w-5 h-5 text-cyan-400" />
            <h3 className="font-display text-sm font-bold text-white uppercase tracking-wider">
              Auto-Generated CRM Audit Note
            </h3>
          </div>

          <button
            onClick={handleCopyCRM}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs font-semibold transition-all"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Copied to Clipboard!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy CRM Note</span>
              </>
            )}
          </button>
        </div>

        {/* Executive Summary */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-2">
          <div className="text-xs font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            Call Executive Summary
          </div>
          <p className="text-sm text-slate-200 leading-relaxed font-sans">
            {summary || 'No summary available.'}
          </p>
        </div>

        {/* Key Points Bullet List */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
          <div className="text-xs font-semibold text-amber-400 uppercase tracking-wider">
            Key Discussion Points
          </div>
          <ul className="space-y-2 text-xs text-slate-300">
            {key_points.map((point, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 flex-shrink-0" />
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Compliance Overview & Recommended Action Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-2">
            <div className="text-xs font-semibold text-purple-400 uppercase tracking-wider flex items-center gap-1.5">
              <AlertCircle className="w-4 h-4 text-purple-400" />
              Compliance Overview
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {compliance_summary || 'N/A'}
            </p>
          </div>

          <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-2">
            <div className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <ArrowRight className="w-4 h-4 text-emerald-400" />
              Recommended Follow-Up Action
            </div>
            <p className="text-xs text-emerald-300 leading-relaxed">
              {recommended_action || 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
