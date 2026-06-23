"""Distillation helpers — capture teacher signals for training students offline."""

from .logprobs import (
    MEDIA_TYPE,
    SCHEMA,
    TeacherLogprobs,
    load_from_blob,
    load_teacher_logprobs,
)

__all__ = [
    "TeacherLogprobs",
    "load_teacher_logprobs",
    "load_from_blob",
    "SCHEMA",
    "MEDIA_TYPE",
]
