"""Object-centric visual cognition building blocks.

These modules provide a minimal cognitive-loop skeleton on top of the
existing encoder/objectness/tracking/prototype-memory pipeline. They do not
claim to implement a full brain model; this stage only creates explicit
object-file, episodic-memory, and predictive-recognition interfaces.
"""

from .object_file import ObjectFile, ObjectFileBuilder, SupportMaskSummary

__all__ = [
    "CognitiveFrameResult",
    "ObjectFile",
    "ObjectFileBuilder",
    "PredictiveRecognizer",
    "RecognitionDecision",
    "SupportMaskSummary",
    "VisualCognitiveLoop",
]


def __getattr__(name: str):
    if name in {"PredictiveRecognizer", "RecognitionDecision"}:
        from .predictive_recognition import PredictiveRecognizer, RecognitionDecision

        return {"PredictiveRecognizer": PredictiveRecognizer, "RecognitionDecision": RecognitionDecision}[name]
    if name in {"CognitiveFrameResult", "VisualCognitiveLoop"}:
        from .visual_cognitive_loop import CognitiveFrameResult, VisualCognitiveLoop

        return {"CognitiveFrameResult": CognitiveFrameResult, "VisualCognitiveLoop": VisualCognitiveLoop}[name]
    raise AttributeError(name)
