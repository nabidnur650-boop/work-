"""EviAudio-MT research prototype."""

from .model import EviAudioMT
from .qcr import CLAPDiagonalMetricRanker, CLAPPriorResidualRanker, QCRRankerOutput
from .qcr_data import QCRManifestDataset, collate_qcr
from .event_needle import MaterializedNeedle, materialize_recipe, temporal_overlap_fraction
from .event_data import EventNeedleArchiveDataset, collate_event_needle

__all__ = [
    "CLAPPriorResidualRanker",
    "CLAPDiagonalMetricRanker",
    "EviAudioMT",
    "QCRManifestDataset",
    "QCRRankerOutput",
    "collate_qcr",
    "MaterializedNeedle",
    "materialize_recipe",
    "temporal_overlap_fraction",
    "EventNeedleArchiveDataset",
    "collate_event_needle",
]
__version__ = "0.1.0"
