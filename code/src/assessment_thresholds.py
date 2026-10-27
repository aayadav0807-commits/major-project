"""Configurable demo review/quality gates for NeuroScreen AI.

These are NOT clinically validated cut-offs and must not be used to diagnose
or rule out learning disabilities. Adjust only after appropriate validation.
"""
from __future__ import annotations
import math

MODEL_CONFIDENCE_MIN_PERCENT = 70.0
TASK_ACCURACY_TARGET_PERCENT = 70.0
EYE_FACE_DETECTION_MIN_PERCENT = 70.0
EYE_VALID_SAMPLES_MIN = 30
EYE_MEAN_TARGET_ERROR_MAX_PERCENTAGE_POINTS = 20.0
READING_WORD_MATCH_TARGET_PERCENT = 70.0


def meets_minimum(value: float, threshold: float) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number >= float(threshold)
    except (TypeError, ValueError):
        return False


def eye_tracking_quality(face_detection_percent, valid_samples, mean_error):
    issues = []
    if not meets_minimum(face_detection_percent, EYE_FACE_DETECTION_MIN_PERCENT):
        issues.append(f"Face detection should be at least {EYE_FACE_DETECTION_MIN_PERCENT:.0f}%.")
    try:
        sample_count = int(valid_samples)
    except (TypeError, ValueError):
        sample_count = 0
    if sample_count < EYE_VALID_SAMPLES_MIN:
        issues.append(f"Collect at least {EYE_VALID_SAMPLES_MIN} valid gaze samples.")
    try:
        error = float(mean_error)
    except (TypeError, ValueError):
        error = float("nan")
    if not math.isfinite(error):
        issues.append("Mean target error is unavailable; repeat the tracking task.")
    elif error > EYE_MEAN_TARGET_ERROR_MAX_PERCENTAGE_POINTS:
        issues.append("Mean target error should be no more than "
                      f"{EYE_MEAN_TARGET_ERROR_MAX_PERCENTAGE_POINTS:.0f} percentage points.")
    return not issues, issues
