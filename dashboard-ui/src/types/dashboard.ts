export type TabId = 
  | 'summary' 
  | 'denoising' 
  | 'diarization' 
  | 'transcript' 
  | 'emotion' 
  | 'compliance' 
  | 'qascore' 
  | 'crm';

export interface StageStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  time_s: number;
}

export interface ProgressState {
  status: 'idle' | 'running' | 'completed' | 'error';
  current_stage: string;
  percent: number;
  stages: Record<string, StageStatus>;
  error?: string | null;
  time_s?: number;
  start_time?: number;
}

export interface Segment {
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  speaker: string;
  confidence?: number;
  uncertain?: boolean;
  text?: string;
  emotion?: string;
  sentiment?: string;
  fusion_source?: string;
  acoustic_emotion?: {
    emotion: string;
    confidence: number;
    indeterminate?: boolean;
    all_scores?: Record<string, number>;
    prosodic_features?: Record<string, number>;
  };
}

export interface DenoiseMetrics {
  snr_before_db?: number;
  snr_after_db?: number;
  audio_quality_grade?: string;
  pesq_score?: number;
  stoi_score?: number;
  rt60_change?: number;
}

export interface TalkRatio {
  agent_duration_s?: number;
  agent_ratio?: number;
  customer_duration_s?: number;
  customer_ratio?: number;
  total_speech_s?: number;
}

export interface RoleResolution {
  method?: string;
  confidence?: number;
  status?: string;
  mapping?: Record<string, string>;
  applied?: boolean;
}

export interface FusionResult {
  emotion: string;
  sentiment: string;
  confidence: number;
  source?: string;
  text_weight?: number;
}

export interface ComplianceResult {
  compliant: boolean;
  total_violations: number;
  agent_violations: number;
  customer_violations: number;
  segment_results?: Array<{
    speaker: string;
    start: number;
    violation_count: number;
    flags: string[];
  }>;
}

export interface QAScoreResult {
  qa_score: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F' | string;
  components?: {
    customer_sentiment?: number;
    compliance?: number;
    agent_stability?: number;
    intent_resolution?: number;
    talk_ratio?: number;
  };
  weights_used?: Record<string, number>;
}

export interface CRMNoteResult {
  summary?: string;
  key_points?: string[];
  compliance_summary?: string;
  recommended_action?: string;
}

export interface CallData {
  duration_s?: number;
  processing_time_s?: number;
  segments?: Segment[];
  talk_ratio?: TalkRatio;
  denoise_metrics?: DenoiseMetrics;
  role_resolution?: RoleResolution;
  fusion?: FusionResult[];
  compliance?: ComplianceResult;
  qa?: QAScoreResult;
  crm_note?: CRMNoteResult;
}
