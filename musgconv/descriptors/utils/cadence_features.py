from musgconv.utils import chord_to_intervalVector
import numpy as np
import partitura as pt


def get_cad_features(part):
    """
    Create cadence relevant features on the note level.

    Parameters
    ----------
    part : partitura.score.Part
        In this function a dummy variable. It can be given empty.
    note_array : numpy structured array
        A part note array. Attention part must contain time signature information.

    Returns
    -------
    feat_array : numpy structured array
        A structured array of features. Each line corresponds to a note in the note array.
    feature_fn : list
        A list of the feature names.
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_beat", "f4"), ("duration_beat", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    elif isinstance(part, np.ndarray):
        note_array = part
        part = None
    else:
        note_array = part.note_array(include_time_signature=True)

    features = list()
    bass_voice = note_array["voice"].max() if note_array["voice" == note_array["voice"].max()]["pitch"].mean() < note_array["voice" == note_array["voice"].min()]["pitch"].mean() else note_array["voice"].min()
    high_voice = note_array["voice"].min() if note_array["voice" == note_array["voice"].min()]["pitch"].mean() > \
                                              note_array["voice" == note_array["voice"].max()]["pitch"].mean() else note_array["voice"].max()
    for i, n in enumerate(note_array):
        d = {}
        n_onset = note_array[note_array["onset_beat"] == n["onset_beat"]]
        n_dur = note_array[np.where((note_array["onset_beat"] < n["onset_beat"]) & (note_array["onset_beat"] + note_array["duration_beat"] > n["onset_beat"]))]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist(), return_pc_class=True)
        pc_class_recentered = sorted(list(map(lambda x: x - min(pc_class), pc_class)))
        maj_int_vecs = [[0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
        prev_4beats = note_array[np.where((note_array["onset_beat"] < n["onset_beat"]) & (note_array["onset_beat"] > n["onset_beat"] - 4))][
                          "pitch"] % 12
        prev_8beats = note_array[np.where((note_array["onset_beat"] < n["onset_beat"]) & (note_array["onset_beat"] > n["onset_beat"] - 8))][
                          "pitch"] % 12
        maj_pcs = [[0, 4, 7], [0, 5, 9], [0, 3, 8], [0, 4], [0, 8], [0, 7], [0, 5]]
        scale = [2, 3, 5, 7, 8, 11] if (n["pitch"] + 3) in chord_pitch % 12 else [2, 4, 5, 7, 9, 11]
        v7 = [[0, 1, 2, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 0]]
        next_voice_notes = note_array[np.where((note_array["voice"] == n["voice"]) & (note_array["onset_beat"] > n["onset_beat"]))]
        prev_voice_notes = note_array[np.where((note_array["voice"] == n["voice"]) & (note_array["onset_beat"] < n["onset_beat"]))]
        prev_voice_pitch = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"] if prev_voice_notes.size else 0
        # start Z features
        d["perfect_triad"] = int_vec in maj_int_vecs
        d["perfect_major_triad"] = d["perfect_triad"] and pc_class_recentered in maj_pcs
        d["is_sus4"] = int_vec == [0, 1, 0, 0, 2, 0] or pc_class_recentered == [0, 5]
        d["in_perfect_triad_or_sus4"] = d["perfect_triad"] or d["is_sus4"]
        d["highest_is_3"] = (chord_pitch.max() - chord_pitch.min()) % 12 in [3, 4]
        d["highest_is_1"] = (chord_pitch.max() - chord_pitch.min()) % 12 == 0 and chord_pitch.max() != chord_pitch.min()

        d["bass_compatible_with_I"] = (n["pitch"] + 5) % 12 in prev_4beats and (n["pitch"] + 11) % 12 in prev_4beats if prev_4beats.size else False
        d["bass_compatible_with_I_scale"] = all([(n["pitch"] + ni) % 12 in prev_8beats for ni in scale]) if prev_8beats.size else False
        d["one_comes_from_7"] = 11 in (prev_voice_pitch - chord_pitch.min())%12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_1"] = 0 in (prev_voice_pitch - chord_pitch.min())%12 and (
                    n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_2"] = 2 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["three_comes_from_4"] = 5 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 in [3, 4] if prev_voice_notes.size else False
        d["five_comes_from_5"] = 7 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min()) % 12 == 7 if prev_voice_notes.size else False

        # Make R features
        d["strong_beat"] = (n["ts_beats"] == 4 and n["onset_beat"] % 2 == 0) or (n["onset_beat"] % n['ts_beats'] == 0) # to debug
        d["sustained_note"] = n_dur.size > 0
        if next_voice_notes.size:
            d["rest_highest"] = n["voice"] == high_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_lowest"] = n["voice"] == bass_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_middle"] = n["voice"] != high_voice and n["voice"] != bass_voice and next_voice_notes["onset_beat"].min() > n[
                "onset_beat"] + n["duration_beat"]
            d["voice_ends"] = False
        else:
            d["rest_highest"] = False
            d["rest_lowest"] = False
            d["rest_middle"] = False
            d["voice_ends"] = True

        # start Y features
        d["v7"] = int_vec in v7
        d["v7-3"] = int_vec in v7 and 4 in pc_class_recentered
        d["has_7"] = 10 in pc_class_recentered
        d["has_9"] = 1 in pc_class_recentered or 2 in pc_class_recentered
        d["bass_voice"] = n["voice"] == bass_voice
        if prev_voice_notes.size:
            x = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"]
            d["bass_moves_chromatic"] = n["voice"] == bass_voice and (1 in x - n["pitch"] or -1 in x-n["pitch"])
            d["bass_moves_octave"] = n["voice"] == bass_voice and (12 in x - n["pitch"] or -12 in x - n["pitch"])
            d["bass_compatible_v-i"] = n["voice"] == bass_voice and (7 in x - n["pitch"] or -5 in x - n["pitch"])
            d["bass_compatible_i-v"] = n["voice"] == bass_voice and (-7 in x - n["pitch"] or 5 in x - n["pitch"])
        # X features
            d["bass_moves_2M"] = n["voice"] == bass_voice and (2 in x - n["pitch"] or -2 in x - n["pitch"])
        else:
            d["bass_moves_chromatic"] = d["bass_moves_octave"] = d["bass_compatible_v-i"] = d["bass_compatible_i-v"] = d["bass_moves_2M"] = False
        features.append(tuple(d.values()))
    feat_array = np.array(features)
    feature_fn = list(d.keys())
    return feat_array, feature_fn

