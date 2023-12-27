import numpy
import numpy as np
from musgconv.utils import chord_to_intervalVector
import partitura as pt
from typing import List, Tuple


CHORDS = {
    "M/m": [0, 0, 1, 1, 1, 0],
    "sus4": [0, 1, 0, 0, 2, 0],
	"M7": [0, 1, 2, 1, 1, 1],
	"M7wo5": [0, 1, 0, 1, 0, 1],
	"Mmaj7": [1, 0, 1, 2, 2, 0],
	"Mmaj7maj9" : [1, 2, 2, 2, 3, 0],
	"M9": [1, 1, 4, 1, 1, 2],
	"M9wo5": [1, 1, 2, 1, 0, 1],
	"m7": [0, 1, 2, 1, 2, 0],
	"m7wo5": [0, 1, 1, 0, 1, 0],
	"m9": [1, 2, 2, 2, 3, 0],
	"m9wo5": [1, 2, 1, 1, 1, 0],
	"m9wo7": [1, 1, 1, 1, 2, 0],
	"mmaj7": [1, 0, 1, 3, 1, 0],
	"Maug": [0, 0, 0, 3, 0, 0],
	"Maug7": [1, 0, 1, 3, 1, 0],
	"mdim": [0, 0, 2, 0, 0, 1],
	"mdim7": [0, 0, 4, 0, 0, 2]
}

NOTE_FEATURES = ["int_vec1", "int_vec2", "int_vec3", "int_vec4", "int_vec5", "int_vec6"] + \
    ["interval"+str(i) for i in range(13)] + list(CHORDS.keys()) + \
    ["is_maj_triad", "is_pmaj_triad", "is_min_triad", 'ped_note',
     'hv_7', "hv_5", "hv_3", "hv_1", "chord_has_2m", "chord_has_2M"]


def get_general_features(part, note_array):
    """
    Create general features on the note level.

    Parameters
    ----------
    note_array : numpy structured array
        A part note array. Attention part must contain time signature information.

    Returns
    -------
    feat_array : numpy structured array
        A structured array of features. Each line corresponds to a note in the note array.
    feature_fn : list
        A list of the feature names.
    """
    ca = np.zeros((len(note_array), len(NOTE_FEATURES)))
    for i, n in enumerate(note_array):
        n_onset = note_array[note_array["onset_div"] == n["onset_div"]]
        n_dur = note_array[np.where((note_array["onset_div"] < n["onset_div"]) & (note_array["onset_div"] + note_array["duration_div"] > n["onset_div"]))]
        n_cons = note_array[note_array["onset_div"] + note_array["duration_div"] == n["onset_div"]]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist(), return_pc_class=True)
        chords_features = {k: (1 if int_vec == v else 0) for k,v in CHORDS.items()}
        pc_class_recentered = sorted(list(map(lambda x : x - min(pc_class), pc_class)))
        is_maj_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 4, 7], [0, 5, 9], [0, 3, 8]] else 0
        is_min_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 3, 7], [0, 5, 8], [0, 4, 9]] else 0
        is_pmaj_triad = 1 if is_maj_triad and 4 in (chord_pitch - chord_pitch.min())%12 and 7 in (chord_pitch - chord_pitch.min())%12 else 0
        ped_note = 1 if n["duration_div"] > n["ts_beats"] else 0
        hv_7 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 10 else 0
        hv_5 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 7 else 0
        hv_3 = 1 if (chord_pitch.max() - chord_pitch.min())%12 in [3, 4] else 0
        hv_1 = 1 if (chord_pitch.max() - chord_pitch.min())%12 == 0 and chord_pitch.max() != chord_pitch.min() else 0
        chord_has_2m = 1 if n["pitch"] - chord_pitch.min() in [1, -1] else 0
        chord_has_2M = 1 if n["pitch"] - chord_pitch.min() in [2, -2] else 0
        intervals = {"interval" + str(i): (1 if i in (n_cons["pitch"] - n["pitch"]) or -i in (
                n_cons["pitch"] - n["pitch"]) else 0) for i in range(13)} if n_cons.size else {"interval" + str(i): 0 for i in range(13)}
        ca[i] = np.array(int_vec + list(intervals.values()) + list(chords_features.values()) +
                         [is_maj_triad, is_pmaj_triad, is_min_triad, ped_note, hv_7, hv_5, hv_3, hv_1, chord_has_2m,
                          chord_has_2M])
    # fa, fnames = pt.musicanalysis.make_note_feats(part, "all")
    # out = np.hstack((fa, ca))
    out = ca
    feature_fn = NOTE_FEATURES
    return out, feature_fn


def get_pc_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["pc_{:02d}".format(i) for i in range(12)]


def get_full_pitch_one_hot(part, note_array, piano_range = True):
    one_hot = np.zeros((len(note_array), 127))
    idx = (np.arange(len(note_array)),note_array["pitch"])
    one_hot[idx] = 1
    if piano_range:
        one_hot = one_hot[:, 21:109]
    return one_hot, ["pc_{:02d}".format(i) for i in range(one_hot.shape[1])]


def get_octave_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["octave_{:02d}".format(i) for i in range(10)]


def get_spelling_features(note_array):
    step_onehot = np.zeros((len(note_array), 7))
    alter_onehot = np.zeros((len(note_array), 7))
    step_vocabulary = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
    alter_vocabulary = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
    step_idx = (np.arange(len(note_array)), np.vectorize(step_vocabulary.get)(note_array["step"]))
    alter_idx = (np.arange(len(note_array)), np.vectorize(alter_vocabulary.get)(note_array["alter"]))
    step_onehot[step_idx] = 1
    alter_onehot[alter_idx] = 1
    return np.hstack((step_onehot, alter_onehot)), ["step_" + s for s in step_vocabulary.keys()] + [
        "alter_" + str(a) for a in alter_vocabulary.keys()]



def get_input_irrelevant_features(part) -> Tuple[np.ndarray, List]:
    """
    Returns features irrelevant of the input being a score or a performance part.

    Parameters
    ----------
    part: Part, PartGroup or PerformedPart

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_div", "f4"), ("duration_div", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    else:
        note_array = part.note_array()

    ca = np.zeros((len(note_array), len(NOTE_FEATURES)))
    for i, n in enumerate(note_array):
        n_onset = note_array[note_array["onset_div"] == n["onset_div"]]
        n_dur = note_array[np.where((note_array["onset_div"] < n["onset_div"]) & (
                    note_array["onset_div"] + note_array["duration_div"] > n["onset_div"]))]
        n_cons = note_array[note_array["onset_div"] + note_array["duration_div"] == n["onset_div"]]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist(), return_pc_class=True)
        chords_features = {k: (1 if int_vec == v else 0) for k, v in CHORDS.items()}
        pc_class_recentered = sorted(list(map(lambda x: x - min(pc_class), pc_class)))
        is_maj_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 4, 7], [0, 5, 9], [0, 3, 8]] else 0
        is_min_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 3, 7], [0, 5, 8], [0, 4, 9]] else 0
        is_pmaj_triad = 1 if is_maj_triad and 4 in (chord_pitch - chord_pitch.min()) % 12 and 7 in (
                    chord_pitch - chord_pitch.min()) % 12 else 0
        ped_note = 1 if n["duration_div"] > 4 else 0
        hv_7 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 10 else 0
        hv_5 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 7 else 0
        hv_3 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 in [3, 4] else 0
        hv_1 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 0 and chord_pitch.max() != chord_pitch.min() else 0
        chord_has_2m = 1 if n["pitch"] - chord_pitch.min() in [1, -1] else 0
        chord_has_2M = 1 if n["pitch"] - chord_pitch.min() in [2, -2] else 0
        octave = int(n["pitch"]/12)
        intervals = {"interval" + str(i): (1 if i in (n_cons["pitch"] - n["pitch"]) or -i in (
                n_cons["pitch"] - n["pitch"]) else 0) for i in range(13)} if n_cons.size else {"interval" + str(i): 0
                                                                                               for i in range(13)}
        ca[i] = np.array(int_vec + list(intervals.values()) + list(chords_features.values()) +
                         [is_maj_triad, is_pmaj_triad, is_min_triad, ped_note, hv_7, hv_5, hv_3, hv_1, chord_has_2m,
                          chord_has_2M, octave])
    pc, pc_names = get_pc_one_hot(part, note_array)

    dur_feats, dur_names = pt.musicanalysis.note_features.duration_feature(note_array, part)
    on_feats, on_names = pt.musicanalysis.note_features.onset_feature(note_array, part)
    pitch_feats, pitch_names = pt.musicanalysis.note_features.polynomial_pitch_feature(note_array, part)
    out = np.hstack((on_feats, dur_feats, pitch_feats, ca, pc))
    names = on_names + dur_names + pitch_names + NOTE_FEATURES + pc_names
    return out, names


def get_voice_separation_features(part) -> Tuple[np.ndarray, List]:
    """
    Returns features Voice Detection features.

    Parameters
    ----------
    part: Part, PartGroup or PerformedPart

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_div", "f4"), ("duration_div", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    elif isinstance(part, np.ndarray):
        note_array = part
        part = None
    else:
        note_array = part.note_array(include_time_signature=True)

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_div"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    # duration_feature = np.expand_dims(np.remainder(note_array["duration_div"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # # new attempt! To delete in case
    # # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_div"]/note_array["ts_beats"])))-0.5)*2, 1)
    # pitch_norm = np.expand_dims(note_array["pitch"] / 127., 1)
    # on_names = ["barnorm_onset", "piecenorm_onset"]
    # dur_names = ["barnorm_duration"]
    # pitch_names = ["pitchnorm"]
    # names = on_names + dur_names + pitch_names + pc_names + octave_names
    # out = np.hstack((onset_feature, np.expand_dims(on_feats[:, 1], 1), duration_feature, pitch_norm, pc_oh, octave_oh))

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pitch_oh, pitch_names = get_full_pitch_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_div"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    octave_oh, octave_names = get_octave_one_hot(part, note_array)
    pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_div"]/note_array["ts_beats"])))-0.5)*2, 1)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_div"]/note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]
    # on_names = ["barnorm_onset", "piecenorm_onset"]
    names = dur_names + pc_names + octave_names 
    out = np.hstack((duration_feature, pc_oh, octave_oh))
    return out, names


def get_chord_analysis_features(part, one_hot=False) -> Tuple[np.ndarray, List]:
    """
    Returns features for chord analysis.

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_div", "f4"), ("duration_div", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    elif isinstance(part, np.ndarray):
        note_array = part
        part = None
    else:
        note_array = part.note_array(include_time_signature=True)
    spelling_features, snames = get_spelling_features(note_array)
    if one_hot:
        octave_oh, octave_names = get_octave_one_hot(part, note_array)
        pc_oh, pc_names = get_pc_one_hot(part, note_array)
        pitch_features = np.hstack((pc_oh, octave_oh))

        pitch_names = pc_names + octave_names
    else:
        pitch_features = np.expand_dims(note_array["pitch"], 1)
        pname = spelling_features[:, :7].argmax(axis=1)
        alter = spelling_features[:, 7:].argmax(axis=1)
        pitch_names = ["pitch"]
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_div"] / note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]

    # find min pitch per unique onset and set to one
    min_max_pitch = np.zeros((len(note_array), 2))
    min_max_pitch_names = ["min_pitch", "max_pitch"]
    metrical_features = np.zeros((len(note_array), 4))
    # is first note of bar
    metrical_features[:, 0] = np.remainder(note_array["onset_div"], note_array["ts_beats"]) == 0
    # is integer beat
    metrical_features[:, 1] = np.remainder(note_array["onset_div"], 1) == 0
    # is half the ts_beats if ts_beats is even
    metrical_features[:, 2] = np.remainder(note_array["onset_div"], note_array["ts_beats"] / 2) == 0
    metrical_names = ["is_first_note_of_bar", "is_downbeat", "is_half_ts_beat", "time_until_next_onset"]
    # time until next onset
    unique_onsets = np.unique(note_array["onset_div"])
    time_until_next_onset = np.r_[np.diff(unique_onsets), 0.0]
    time_until_next_onset = time_until_next_onset[np.searchsorted(unique_onsets, note_array["onset_div"])]
    metrical_features[:, 3] = 1 - np.tanh(time_until_next_onset/note_array["ts_beats"])
    ca = np.zeros((len(note_array), len(NOTE_FEATURES)))
    for i, n in enumerate(note_array):
        n_onset = note_array[note_array["onset_div"] == n["onset_div"]]
        n_dur = note_array[np.where((note_array["onset_div"] < n["onset_div"]) & (
                    note_array["onset_div"] + note_array["duration_div"] > n["onset_div"]))]
        n_cons = note_array[note_array["onset_div"] + note_array["duration_div"] == n["onset_div"]]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        min_max_pitch[i, 0] = np.min(chord_pitch) == n["pitch"]
        min_max_pitch[i, 1] = np.max(chord_pitch) == n["pitch"]
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist(), return_pc_class=True)
        chords_features = {k: (1 if int_vec == v else 0) for k, v in CHORDS.items()}
        pc_class_recentered = sorted(list(map(lambda x: x - min(pc_class), pc_class)))
        is_maj_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 4, 7], [0, 5, 9], [0, 3, 8]] else 0
        is_min_triad = 1 if chords_features["M/m"] and pc_class_recentered in [[0, 3, 7], [0, 5, 8], [0, 4, 9]] else 0
        is_pmaj_triad = 1 if is_maj_triad and 4 in (chord_pitch - chord_pitch.min()) % 12 and 7 in (
                    chord_pitch - chord_pitch.min()) % 12 else 0
        ped_note = 1 if n["duration_div"] > n["ts_beats"] else 0
        hv_7 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 10 else 0
        hv_5 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 7 else 0
        hv_3 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 in [3, 4] else 0
        hv_1 = 1 if (chord_pitch.max() - chord_pitch.min()) % 12 == 0 and chord_pitch.max() != chord_pitch.min() else 0
        chord_has_2m = 1 if n["pitch"] - chord_pitch.min() in [1, -1] else 0
        chord_has_2M = 1 if n["pitch"] - chord_pitch.min() in [2, -2] else 0
        intervals = {"interval" + str(i): (1 if i in (n_cons["pitch"] - n["pitch"]) or -i in (
                n_cons["pitch"] - n["pitch"]) else 0) for i in range(13)} if n_cons.size else {"interval" + str(i): 0
                                                                                               for i in range(13)}
        ca[i] = np.array(int_vec + list(intervals.values()) + list(chords_features.values()) +
                         [is_maj_triad, is_pmaj_triad, is_min_triad, ped_note, hv_7, hv_5, hv_3, hv_1, chord_has_2m,
                          chord_has_2M])
    out = np.hstack((pitch_features, spelling_features, duration_feature, ca, min_max_pitch, metrical_features))
    names = pitch_names + snames + dur_names + NOTE_FEATURES + min_max_pitch_names + metrical_names
    return out, names


def get_panalysis_features(part) -> Tuple[np.ndarray, List]:
    if isinstance(part, pt.performance.PerformedPart):
        raise TypeError("PerformedPart is not supported")
    elif isinstance(part, np.ndarray):
        note_array = part
        if not np.all(np.isin(["is_downbeat", "ts_beats", "ts_beat_type"], note_array.dtype.names)):
            raise ValueError("Not all field names are given")
    else:
        note_array = part.note_array(include_time_signature=True, include_metrical_position=True)
    octave_oh, octave_names = get_octave_one_hot(part, note_array)
    pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_div"]/note_array["ts_beats"])))-0.5)*2, 1)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_div"] / note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]
    voice = np.expand_dims(note_array["voice"], 1)
    voice_names = ["voice"]
    metrical = np.expand_dims(note_array["is_downbeat"], 1)
    metrical_names = ["is_downbeat"]

    names = dur_names + pc_names + octave_names + voice_names + metrical_names
    out = np.hstack((duration_feature, pc_oh, octave_oh, voice, metrical))
    return out, names
