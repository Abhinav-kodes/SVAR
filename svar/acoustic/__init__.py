"""Acoustic analysis subpackage for speaker-normalized emotion detection."""
from .speaker_baseline import SpeakerBaselineBuilder, SpeakerBaselineProfile
from .baseline_features import extract_baseline_features
from .relative_features import build_relative_acoustic_vector
from .trajectory_model import SpeakerShiftTemporalModel
