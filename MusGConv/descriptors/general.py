import numpy as np
import numpy.lib.recfunctions as rfn
import partitura
import partitura.score

from .utils import *
from typing import Union, Tuple, List
# If imported should contain the name of descriptors
from musgconv.utils.globals import *
import types
import sys


def notewise(part: Union[List, Union[partitura.score.Part, partitura.score.PartGroup]], dnames:Union[str, List] = "all", include_rests=False) -> Tuple[np.ndarray, List]:
    """

    Parameters
    ----------
    part
        A partitura part or PartGroup or List of parts.
    dnames
        The names of functions of discriptors
    include_rests : bool
        include_rests to feature computation.

    Returns
    -------
    out_feats : np.ndarray
        An array of feature values
    out_dnames : list
        A list of feature names
    """
    if dnames == "all":
        descriptors = list(map(lambda x: getattr(sys.modules[__name__], x), DESCRIPTORS.values()))
    elif isinstance(dnames, str):
        descriptors = [getattr(sys.modules[__name__], dnames)]
    elif isinstance(dnames, types.FunctionType):
        descriptors = [dnames]
    else:
        descriptors = list(map(lambda x: getattr(sys.modules[__name__], dnames)))

    if isinstance(part, list) or isinstance(part, partitura.score.PartGroup):
        part = partitura.score.merge_parts(part)

    note_array = part.note_array(
        include_time_signature=True,
        include_staff=True,
        include_grace_notes=True,
        include_metrical_position=True
        )

    if include_rests:
        rest_array = part.rest_array(
            include_time_signature=True,
            include_staff=True,
            include_grace_notes=True,
            include_metrical_position=True
        )
        note_array = rfn.merge_arrays((note_array, rest_array))

    out_feats = list()
    out_names = list()
    for fdes in descriptors:
        x, fname = fdes(part, note_array)

        out_feats.append(x)

        if isinstance(fname, str):
            out_names.append(fname)
        elif isinstance(fname, list):
            out_names += fname

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        assert len(x.shape) == 2

    out_feats = np.hstack(out_feats)

    return out_feats, out_names


def onsetwise(part: Union[List, Union[partitura.score.Part, partitura.score.PartGroup]], dnames:Union[str, List]="all") -> Tuple[np.ndarray, List]:
    pass


def measurewise(part: Union[partitura.score.Part, partitura.score.PartGroup], dnames:Union[str, List]="all") -> Tuple[np.ndarray, List]:
    pass


def pitch_spelling_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    features, fnames = get_input_irrelevant_features(part)
    return features, fnames

def cadence_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    chord_features, chord_names = get_chord_analysis_features(part)
    cad_features, cad_names = get_cad_features(part)
    features = np.hstack((chord_features, cad_features))
    fnames = chord_names + cad_names
    return features, fnames

def voice_separation_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    features, fnames = get_voice_separation_features(part)
    return features, fnames


def chord_analysis_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    features, fnames = get_chord_analysis_features(part)
    return features, fnames


def panalysis_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    features, fnames = get_panalysis_features(part)
    return features, fnames


def select_features(part, features):
    if features == "voice":
        note_features, _ = voice_separation_features(part)
    elif features == "chord":
        note_features, _ = chord_analysis_features(part)
    elif features == "panalysis":
        note_features, _ = panalysis_features(part)
    elif features == "pitch_spelling":
        note_features, _ = pitch_spelling_features(part)
    elif features == "cadence":
        note_features, _ = cadence_features(part)
    else:
        note_features, _ = get_input_irrelevant_features(part)
    return note_features
