from itertools import combinations
import re
import partitura
import numpy as np
import pandas as pd
from musgconv.utils.globals import *
from musgconv.utils.general import exit_after
from fractions import Fraction
import numpy.lib.recfunctions as rfn
from music21.key import Key
from music21.pitch import Pitch
from music21.interval import Interval, intervalFromGenericAndChromatic
from .chord_representations_latest import create_data_latest, _getTranspositions_latest


_transposeKey = {}
_transposePitch = {}
_transposePcSet = {}
_pitchObj = {}
_keyObj = {}
_intervalObj = {}
_weberEuclidean = {}
_getTonicizationScaleDegree = {}


def chord_to_intervalVector(midi_pitches, return_pc_class=False):
    '''Given a chord it calculates the Interval Vector.


    Parameters
    ----------
    midi_pitches : list(int)
        The midi_pitches, is a list of integers < 128.

    Returns
    -------
    intervalVector : list(int)
        The interval Vector is a list of six integer values.
    '''
    intervalVector = [0, 0, 0, 0, 0, 0]
    PC = set([mp%12 for mp in midi_pitches])
    for p1, p2 in combinations(PC, 2):
        interval = int(abs(p1 - p2))
        if interval <= 6:
            index = interval
        else:
            index = 12 - interval
        if index != 0:
            index = index-1
            intervalVector[index] += 1
    if return_pc_class:
        return intervalVector, list(PC)
    else:
        return intervalVector


def encode_one_hot(df, object, transposition=None):
    '''Given a dataframe it encodes the chord column as one hot.'''
    return object(df).run(transposition=transposition)


def fixkey(key_string):
    if key_string == "A#":
        return "a#"
    else:
        return key_string


def create_data(filtered_df, time_signature, interval="P1"):
    note_array = list()
    onset = np.expand_dims(filtered_df["j_offset"].to_numpy(), axis=1)
    localkey = encode_one_hot(filtered_df, LocalKey35, transposition=interval)
    tonkey = encode_one_hot(filtered_df, TonicizedKey35, transposition=interval)
    quality = encode_one_hot(filtered_df, ChordQuality15, transposition=interval)
    root = encode_one_hot(filtered_df, ChordRoot35, transposition=interval)
    inversion = encode_one_hot(filtered_df, Inversion4, transposition=interval)
    degree1 = encode_one_hot(filtered_df, PrimaryDegree22, transposition=interval)
    degree2 = encode_one_hot(filtered_df, SecondaryDegree22, transposition=interval)
    bass = encode_one_hot(filtered_df, Bass35, transposition=interval)
    hrythm = encode_one_hot(filtered_df, HarmonicRhythm2, transposition=interval)
    pcset = encode_one_hot(filtered_df, PitchClassSet94, transposition=interval)
    romanNumeral = encode_one_hot(filtered_df, RomanNumeral76, transposition=interval)
    y = np.stack(
        (localkey, tonkey, degree1, degree2, quality, inversion, root, romanNumeral, hrythm, pcset, bass, onset),
        axis=1)
    for i, row in filtered_df.iterrows():
        pitches = eval(row["s_notes"])
        # is_onset = eval(row["s_isOnset"])
        for j, pitch in enumerate(pitches):
            pitch = TransposePitch(pitch, interval)
            p = re.findall(r'[A-Za-z]+|\W+|\d+', pitch)
            step, alter, octave = (p[0], p[1], eval(p[2])) if len(p) == 3 else (p[0], "", eval(p[1]))
            alter = ALTER[alter]
            mp = partitura.utils.pitch_spelling_to_midi_pitch(step, alter, octave)
            note_array.append((row["j_offset"], row["s_duration"], mp, int(time_signature), 4, step, alter, octave))
    dtype = np.dtype(
        [('onset_beat', float), ('duration_beat', float), ('pitch', int), ('ts_beats', int), ('ts_beat_type', int),
         ("step", '<U10'), ("alter", int), ("octave", int)])
    X = np.array(note_array, dtype)
    return X, y




def time_divided_tsv_to_note_array(time_divided_tsv_path, transpose=False, version="v1.0.0", include_measures=False):
    '''Given a time divided tsv file from AugmentedNet Dataset it returns a numpy array of notes.

    Parameters
    ----------
    time_divided_tsv_path : str
        The path to the time divided csv file.

    Returns
    -------
    note_array : np.array
        The note array is a numpy array of notes.
    '''

    df = pd.read_csv(time_divided_tsv_path, sep='\t', header=0)
    if "j_offset" not in df.columns:
        df["j_offset"] = df["Unnamed: 0"]
    time_signature = len(df[df["s_measure"] == 2]) / 8
    measures = df["s_measure"].to_numpy()
    diffs = np.r_[True, np.diff(measures) == 1]
    measure_start_beats = df["j_offset"][diffs].to_numpy()
    measure_end_beats = np.r_[measure_start_beats[1:], 2*measure_start_beats[-1] - measure_start_beats[-2]]
    # Assume 4/4 time signature when 0
    time_signature = 4 if time_signature == 0 else time_signature
    has_onsets = df["s_isOnset"].apply(lambda x: any(eval(x))).to_numpy()
    num_notes = df["s_isOnset"].apply(lambda x: len(eval(x))).to_numpy()
    # Filter when rests are present
    durations = np.absolute(df["s_duration"].to_numpy()[:] - np.roll(df["s_duration"].to_numpy()[:], 1)) > 0
    # Filter when a note stop sounding in a set others are still present.
    has_nnotes = np.absolute(num_notes[:] - np.roll(num_notes[:], 1)) > 0
    idx = np.nonzero(np.logical_or.reduce((has_onsets, has_nnotes, durations)))[0]
    # NOTE: reset index.
    filtered_df = df.iloc[idx, :].sort_values(by=["j_offset"])
    filtered_df["a_degree1"] = filtered_df["a_degree1"].astype(str)
    filtered_df["a_pcset"] = filtered_df["a_pcset"].apply(eval)
    filtered_df["a_localKey"] = filtered_df["a_localKey"].apply(fixkey)
    createfunction = create_data if version == "v1.0.0" else create_data_latest
    if transpose:
        transpositions = _getTranspositions(filtered_df) if version=="v1.0.0" else _getTranspositions_latest(filtered_df)
        X, y = createfunction(filtered_df, time_signature)
        data = [(X, y, np.vstack((measure_start_beats, measure_end_beats)).T)]
        for interval in transpositions:
            X, y = createfunction(filtered_df, time_signature, interval=interval)
            data.append((X, y, np.vstack((measure_start_beats, measure_end_beats)).T))
        return data
    else:
        X, y = createfunction(filtered_df, time_signature)
        return X, y, np.vstack((measure_start_beats, measure_end_beats)).T



def create_divs_from_beats(note_array):
    duration_fractions = [Fraction(float(ix)).limit_denominator(256) for ix in note_array["duration_beat"]]
    onset_fractions = [Fraction(float(ix)).limit_denominator(256) for ix in note_array["onset_beat"]]
    divs = np.lcm.reduce(
        [Fraction(float(ix)).limit_denominator(256).denominator for ix in np.unique(note_array["duration_beat"])])
    onset_divs = list(map(lambda r: int(divs * r.numerator / r.denominator), onset_fractions))
    min_onset_div = min(onset_divs)
    if min_onset_div < 0:
        onset_divs = list(map(lambda x: x - min_onset_div, onset_divs))
    duration_divs = list(map(lambda r: int(divs * r.numerator / r.denominator), duration_fractions))
    na_divs = np.array(list(zip(onset_divs, duration_divs)), dtype=[("onset_div", int), ("duration_div", int)])
    return rfn.merge_arrays((note_array, na_divs), flatten=True, usemask=False), divs


def tie_consecutive_notes(note_array, labels):
    """Ties consecutive notes with the same pitch and onset time.

    Parameters
    ----------
    note_array : np.ndarray
        The note array.
    labels : np.ndarray
        The labels array.

    Returns
    -------
    note_array : np.ndarray
        The note array with tied notes.
    """
    note_array = note_array.copy()
    note_array = np.sort(note_array, order=["onset_beat", "pitch"])
    skip_indices = []
    for i in range(0, len(note_array)-1):
        note = note_array[i]
        cond = True
        if i in skip_indices:
            continue
        while cond:
            idx = np.where((note_array[i+1:]["onset_beat"] == note["onset_beat"]+note["duration_beat"]) & (note_array[i+1:]["pitch"] == note["pitch"]))[0]
            if len(idx) == 0:
                cond = False
            else:
                idx = int(idx[0] + i + 1)
                note["duration_beat"] += note_array[idx]["duration_beat"]
                skip_indices.append(idx)
    skip_indices = np.unique(skip_indices).astype(int)
    note_array = np.delete(note_array, skip_indices)
    unique_onsets = np.unique(note_array["onset_beat"])
    labels_to_skip = np.where(~np.isin(labels[:, -1], unique_onsets))[0]
    labels = np.delete(labels, labels_to_skip, axis=0)
    assert(np.all(np.diff(note_array["onset_beat"]) >= 0), "The Note array is not sorted.")
    assert(np.all(np.diff(labels[:, -1]) >= 0), "The Label onsets array are not sorted.")
    return note_array, labels


# @exit_after(45)
def time_divided_tsv_to_part(time_divided_tsv_path, transpose=False, version="v1.0.0"):
    '''Given a time divided tsv file from AugmentedNet Dataset it returns a partitura Part object.

    Parameters
    ----------
    time_divided_tsv_path : str
        The path to the time divided csv file.

    Returns
    -------
    part : partitura.partitura.music.Part
        The part object.
    '''
    if transpose:
        new_X = []
        X = time_divided_tsv_to_note_array(time_divided_tsv_path, transpose=True, version=version)
        for (note_array, labels, measures) in X:
            note_array, labels = tie_consecutive_notes(note_array, labels)
            note_array, divs = create_divs_from_beats(note_array)
            measures = measures*divs
            new_X.append((note_array, labels, measures))
        return new_X
    else:
        note_array, labels, measures = time_divided_tsv_to_note_array(time_divided_tsv_path, transpose=transpose, version=version)
        note_array, labels = tie_consecutive_notes(note_array, labels)
        note_array, divs = create_divs_from_beats(note_array)
        return note_array, labels, measures*divs


"""The output tonal representations learned through multitask learning.
Classes and data structures related to tonal features."""



def TransposeKey(key, interval):
    """Transposes a key based on an interval string (e.g., 'm3')."""
    duple = (key, interval)
    if duple in _transposeKey:
        return _transposeKey[duple]
    keyObj = m21Key(key)
    transposed = keyObj.transpose(interval).tonicPitchNameWithCase
    _transposeKey[duple] = transposed
    return transposed


def TransposePitch(pitch, interval):
    """Transposes a pitch based on an interval string (e.g., 'm3')."""
    duple = (pitch, interval)
    if duple in _transposePitch:
        return _transposePitch[duple]
    pitchObj = m21Pitch(pitch)
    transposed = pitchObj.transpose(interval).nameWithOctave
    _transposePitch[duple] = transposed
    return transposed


def TransposePcSet(pcset, interval):
    """Transposes a pcset based on an interval string (e.g., 'm3')."""
    duple = (pcset, interval)
    if duple in _transposePcSet:
        return _transposePcSet[duple]
    semitones = m21IntervalStr(interval).semitones
    transposed = [(x + semitones) % 12 for x in pcset]
    transposed = tuple(sorted(transposed))
    _transposePcSet[duple] = transposed
    return transposed


def m21IntervalStr(interval):
    """A cached interval object, based on the string (e.g., 'm3')."""
    if interval in _intervalObj:
        return _intervalObj[interval]
    intervalObj = Interval(interval)
    _intervalObj[interval] = intervalObj
    return intervalObj


def m21Key(key):
    """A cached key object, based on a string (e.g., 'c#')."""
    if key in _keyObj:
        return _keyObj[key]
    keyObj = Key(key)
    _keyObj[key] = keyObj
    return keyObj


def m21Pitch(pitch):
    """A cached pitch object, based on a string (e.g., 'C#')."""
    if pitch in _pitchObj:
        return _pitchObj[pitch]
    pitchObj = Pitch(pitch)
    _pitchObj[pitch] = pitchObj
    return pitchObj


def _getTranspositions(df):
    localKeys = df.a_localKey.to_list()
    localKeys = set(localKeys)
    ret = []
    for interval in INTERVALCLASSES:
        if interval == "P1":
            continue
        transposed = [TransposeKey(k, interval) for k in localKeys]
        # Transpose to this interval if every modulation lies within
        # the set of KEY classes that we can classify
        if set(transposed).issubset(set(KEYS)):
            ret.append(interval)
    return ret


class FeatureRepresentation(object):
    features = 1

    def __init__(self, df):
        self.df = df
        self.frames = len(df.index)
        self.dtype = "i8"
        self.array = self.run()

    @property
    def shape(self):
        return (self.frames, self.features)

    def run(self, tranposition=None):
        array = np.zeros(self.shape, dtype=self.dtype)
        return array

    def dataAugmentation(self, intervals):
        for interval in intervals:
            yield self.run(transposition=interval)
        return

    @classmethod
    def encodeManyHot(cls, array, timestep, index, value=1):
        if 0 <= index < cls.features:
            array[timestep, index] = value
        else:
            raise IndexError

    @classmethod
    def encodeCategorical(cls, array, timestep, classNumber):
        if 0 <= classNumber < cls.features:
            array[timestep] = classNumber
        else:
            raise IndexError


class FeatureRepresentationTI(FeatureRepresentation):
    """TI stands for Transposition Invariant.

    If a representation is TI, dataAugmentation consists of
    returning a copy of the array that was already computed.
    """

    def dataAugmentation(self, intervals):
        for _ in intervals:
            yield np.copy(self.array)
        return


class OutputRepresentation(FeatureRepresentation):
    """Output representations are all one-hot encoded (no many-hots).

    That makes them easier to template.
    """

    classList = []
    dfFeature = ""
    transpositionFn = None

    def run(self, transposition="P1"):
        array = np.zeros(self.shape, dtype=self.dtype)
        for frame, dfFeature in enumerate(self.df[self.dfFeature]):
            transposed = self.transpositionFn(dfFeature, transposition)
            if transposed in self.classList:
                rnIndex = self.classList.index(transposed)
                array[frame] = rnIndex
            else:
                array[frame] = self.classesNumber() - 1
        return array

    @classmethod
    def classesNumber(cls):
        return len(cls.classList)

    @classmethod
    def decode(cls, array):
        return [cls.classList[index] for index in array.reshape(-1)]

    @classmethod
    def decodeOneHot(cls, array):
        if len(array.shape) != 2 or array.shape[1] != len(cls.classList):
            raise IndexError("Strange array shape.")
        return [cls.classList[np.argmax(onehot)] for onehot in array]


class OutputRepresentationTI(FeatureRepresentationTI):
    """Output representations are all one-hot encoded (no many-hots).

    That makes them easier to template.
    """

    classList = []
    dfFeature = ""

    def run(self, transposition="P1"):
        array = np.zeros(self.shape, dtype=self.dtype)
        for frame, dfFeature in enumerate(self.df[self.dfFeature]):
            if dfFeature in self.classList:
                rnIndex = self.classList.index(dfFeature)
                array[frame] = rnIndex
            else:
                array[frame] = self.classesNumber() - 1
        return array

    @classmethod
    def classesNumber(cls):
        return len(cls.classList)

    @classmethod
    def decode(cls, array):
        return [cls.classList[index] for index in array.reshape(-1)]

    @classmethod
    def decodeOneHot(cls, array):
        if len(array.shape) != 2 or array.shape[1] != len(cls.classList):
            raise IndexError("Strange array shape.")
        return [cls.classList[np.argmax(onehot)] for onehot in array]


def identity(a, b):
    return a


class Bass35(OutputRepresentation):
    classList = SPELLINGS
    dfFeature = "a_bass"
    transpositionFn = staticmethod(TransposePitch)


class Inversion4(OutputRepresentationTI):
    classList = list(range(4))
    dfFeature = "a_inversion"

    def run(self, transposition="P1"):
        array = np.zeros(self.shape, dtype=self.dtype)
        for frame, inversion in enumerate(self.df[self.dfFeature]):
            if inversion > 3:
                # Any chord beyond sevenths is encoded as "root" position
                inversion = 0
            array[frame] = int(inversion)
        return array


class HarmonicRhythm2(OutputRepresentationTI):
    classList = [True, False]
    dfFeature = "a_isOnset"


class RomanNumeral76(OutputRepresentationTI):
    classList = COMMON_ROMAN_NUMERALS
    dfFeature = "a_romanNumeral"


class PrimaryDegree22(OutputRepresentationTI):
    classList = DEGREES
    dfFeature = "a_degree1"


class SecondaryDegree22(OutputRepresentationTI):
    classList = DEGREES
    dfFeature = "a_degree2"


class LocalKey35(OutputRepresentation):
    classList = KEYS
    dfFeature = "a_localKey"
    transpositionFn = staticmethod(TransposeKey)


class TonicizedKey35(OutputRepresentation):
    classList = KEYS
    dfFeature = "a_tonicizedKey"
    transpositionFn = staticmethod(TransposeKey)


class ChordRoot35(OutputRepresentation):
    classList = SPELLINGS
    dfFeature = "a_root"
    transpositionFn = staticmethod(TransposePitch)


class ChordQuality15(OutputRepresentationTI):
    classList = CHORD_QUALITIES
    dfFeature = "a_quality"


class PitchClassSet94(OutputRepresentation):
    classList = PCSETS
    dfFeature = "a_pcset"
    transpositionFn = staticmethod(TransposePcSet)


#
# available_representations = {
#     "Bass35": Bass35,
#     "Tenor35": Tenor35,
#     "Alto35": Alto35,
#     "Soprano35": Soprano35,
#     "ChordQuality11": ChordQuality11,
#     "ChordRoot35": ChordRoot35,
#     "HarmonicRhythm7": HarmonicRhythm7,
#     "Inversion4": Inversion4,
#     "LocalKey38": LocalKey38,
#     "PitchClassSet121": PitchClassSet121,
#     "PrimaryDegree22": PrimaryDegree22,
#     "RomanNumeral31": RomanNumeral31,
#     "SecondaryDegree22": SecondaryDegree22,
#     "TonicizedKey38": TonicizedKey38,
# }


available_representations = {
    "localkey": LocalKey35,
    "tonkey": TonicizedKey35,
    "degree1": PrimaryDegree22,
    "degree2": SecondaryDegree22,
    "quality": ChordQuality15,
    "inversion": Inversion4,
    "root": ChordRoot35,
    "romanNumeral": RomanNumeral76,
    "hrhythm": HarmonicRhythm2,
    "pcset": PitchClassSet94,
    "bass": Bass35,
}


inversions = {
    "triad": {
        0: "",
        1: "6",
        2: "64",
    },
    "seventh": {
        0: "7",
        1: "65",
        2: "43",
        3: "2",
    },
}


import numpy as np

# The diagonal of keys in the Weber tonal chart
# constrained to the range of keys between [Fb, G#]
WEBERDIAGONAL = [
    "B--",
    "c-",
    "F-",
    "g-",
    "C-",
    "d-",
    "G-",
    "a-",
    "D-",
    "e-",
    "A-",
    "b-",
    "E-",
    "f",
    "B-",
    "c",
    "F",
    "g",
    "C",
    "d",
    "G",
    "a",
    "D",
    "e",
    "A",
    "b",
    "E",
    "f#",
    "B",
    "c#",
    "F#",
    "g#",
    "C#",
    "d#",
    "G#",
    "a#",
    "D#",
    "e#",
    "A#",
    "b#",
]

# Adding this vector takes you to the next coordinates of a key
# in the Weber tonal chart, ascending from "flatness" to "sharpness"
TRANSPOSITION = np.array((2, 3))


def _we(k1, k2):
    """A measurement of key distance based on the Weber tonal chart."""
    i1, i2 = WEBERDIAGONAL.index(k1), WEBERDIAGONAL.index(k2)
    flatterkey, sharperkey = sorted((i1, i2))
    coord1 = np.array((flatterkey, flatterkey))
    coord2 = np.array((sharperkey, sharperkey))
    smallerdistance = 1337
    for i in range(len(WEBERDIAGONAL) // 2):
        trans = TRANSPOSITION * i
        newcoord1 = trans + coord1
        distance = np.linalg.norm(coord2 - newcoord1)
        if distance < smallerdistance:
            smallerdistance = distance
    return smallerdistance


inversions = {
    "triad": {
        0: "",
        1: "6",
        2: "64",
    },
    "seventh": {
        0: "7",
        1: "65",
        2: "43",
        3: "2",
    },
}


def formatChordLabel(cl):
    """Format the chord label for end-user presentation."""
    # The only change I can think of: Cmaj -> C
    cl = cl.replace("maj", "") if cl.endswith("maj") else cl
    cl = cl.replace("-", "b")
    return cl


def formatRomanNumeral(rn, key):
    """Format the Roman numeral label for end-user presentation."""
    # Something of "I" and "I" of something
    if rn == "I/I":
        rn = "I"
    return rn


def solveChordSegmentation(df):
    return df.dropna()[df.hrhythm == 0]


def resolveRomanNumeralCosine(b, t, a, s, pcs, key, numerator, tonicizedKey):
    import music21

    pcs = eval(pcs) if isinstance(pcs, str) else pcs
    pcsetVector = np.zeros(12)
    chord = music21.chord.Chord(f"{b}2 {t}3 {a}4 {s}5")
    for n in chord.pitches:
        pcsetVector[n.pitchClass] += 1
    for pc in pcs:
        pcsetVector[pc] += 1
    chordNumerator = music21.roman.RomanNumeral(
        numerator.replace("Cad", "Cad64"), tonicizedKey
    ).pitchClasses
    for pc in chordNumerator:
        pcsetVector[pc] += 1
    smallestDistance = -2
    for pcs in frompcset:
        v2 = np.zeros(12)
        for p in pcs:
            v2[p] = 1
        similarity = cosineSimilarity(pcsetVector, v2)
        if similarity > smallestDistance:
            pcset = pcs
            smallestDistance = similarity
    if tonicizedKey not in frompcset[pcset]:
        # print("Forcing a tonicization")
        candidateKeys = list(frompcset[pcset].keys())
        # prioritize modal mixture
        tonicizedKey = forceTonicization(key, candidateKeys)
    rnfigure = frompcset[pcset][tonicizedKey]["rn"]
    chord = frompcset[pcset][tonicizedKey]["chord"]
    quality = frompcset[pcset][tonicizedKey]["quality"]
    chordtype = "seventh" if len(pcset) == 4 else "triad"
    # if you can't find the predicted bass
    # in the pcset, assume root position
    inv = chord.index(b) if b in chord else 0
    invfigure = inversions[chordtype][inv]
    if invfigure in ["65", "43", "2"]:
        rnfigure = rnfigure.replace("7", invfigure)
    elif invfigure in ["6", "64"]:
        rnfigure += invfigure
    rn = rnfigure
    if numerator == "Cad" and inv == 2:
        rn = "Cad64"
    if tonicizedKey != key:
        denominator = getTonicizationScaleDegree(key, tonicizedKey)
        rn = f"{rn}/{denominator}"
    chordLabel = f"{chord[0]}{quality}"
    if inv != 0:
        chordLabel += f"/{chord[inv]}"
    return rn, chordLabel


def generateRomanText(h, ts):
    metadata = h.metadata
    metadata.composer = metadata.composer or "Unknown"
    metadata.title = metadata.title or "Unknown"
    composer = metadata.composer.split("\n")[0]
    title = metadata.title.split("\n")[0]
    rntxt = ""
#     rntxt = f"""\
# Composer: {composer}
# Title: {title}
# Analyst: AugmentedNet v{__version__} - https://github.com/napulen/AugmentedNet
# """
    currentMeasure = -1
    for n in h.flat.notes:
        if not n.lyric:
            continue
        rn = n.lyric.split()[0]
        key = ""
        measure = n.measureNumber
        beat = float(n.beat)
        if beat.is_integer():
            beat = int(beat)
        newts = ts.get((measure, beat), None)
        if newts:
            rntxt += f"\nTime Signature: {newts}\n"
        if ":" in rn:
            key, rn = rn.split(":")
        if measure != currentMeasure:
            rntxt += f"\nm{measure}"
            currentMeasure = measure
        if beat != 1:
            rntxt += f" b{round(beat, 3)}"
        if key:
            rntxt += f" {key.replace('-', 'b')}:"
        rntxt += f" {rn}"
    return


def weberEuclidean(k1, k2):
    """A cached version of keydistance.weberEuclidean."""
    duple = (k1, k2)
    if duple in _weberEuclidean:
        return _weberEuclidean[duple]
    distance = _we(k1, k2)
    _weberEuclidean[duple] = distance
    return distance


def getTonicizationScaleDegree(localKey, tonicizedKey):
    """A cached version of keydistance.weberEuclidean."""
    duple = (localKey, tonicizedKey)
    if duple in _getTonicizationScaleDegree:
        return _getTonicizationScaleDegree[duple]
    # TODO replace the following line to work without music21
    degree = ""
    degree = _gtsd(localKey, tonicizedKey)
    _getTonicizationScaleDegree[duple] = degree
    return degree


def _gtsd(localKey, tonicizedKey):
    import music21
    """Return the Roman numeral degree of a tonicization (denominator)."""
    tonic, _, third, _, fifth, _, _, _ = music21.key.Key(tonicizedKey).pitches
    c1 = music21.chord.Chord([tonic, third, fifth])
    # TODO: Use harmalysis to solve this problem, not romanNumeralFromChord
    degree = music21.roman.romanNumeralFromChord(c1, localKey).figure
    # TODO: This is a hack to workaround music21
    if localKey.islower() and degree == "bVI":
        degree = "VI"
    return degree

def forceTonicization(localKey, candidateKeys):
    """Forces a tonicization of candidateKey that exist in vocabulary."""
    tonicizationDistance = 1337
    tonicization = ""
    for candidateKey in candidateKeys:
        distance = weberEuclidean(localKey, candidateKey)
        # print(f"\t{localKey} -> {candidateKey} = {distance}")
        scaleDegree = getTonicizationScaleDegree(localKey, candidateKey)
        if scaleDegree not in ["i", "III"]:
            # Slight preference for parallel minor and relative major
            distance *= 1.05
        if scaleDegree not in ["i", "I", "III", "iv", "IV", "v", "V"]:
            distance *= 1.05
        if distance < tonicizationDistance:
            tonicization = candidateKey
            tonicizationDistance = distance
    return tonicization


def cosineSimilarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def closestPcSet(pcset):
    """Get the closest matching pcset from the vocabulary.
    Uses cosine similarity to measure the distance between
    the given pcset and all pcsets in the vocabulary.
    """
    v1 = np.zeros(12)
    for pc in pcset:
        v1[pc] = 1
    mostSimilarScore = -2
    closestPcSet = []
    for pcs in frompcset:
        v2 = np.zeros(12)
        for p in pcs:
            v2[p] = 1
        similarity = cosineSimilarity(v1, v2)
        if similarity > mostSimilarScore:
            closestPcSet = pcs
            mostSimilarScore = similarity
    return closestPcSet