# TODO: move this in another file
from functools import partial
import partitura
from typing import Union, Dict
from musgconv.utils.general import exit_after
import numpy as np
import torch
import os


@exit_after(60)
def pianorolls_from_part(
    x: Union[Union[partitura.score.Part, partitura.score.PartGroup, str], np.ndarray],
    time_unit: str = "beat",
    time_div: int = 12,
    musical_beat: bool = True,
    path: str = "",
) -> Dict:
    if (
        isinstance(x, partitura.score.Part)
        or isinstance(x, partitura.score.PartGroup)
        or isinstance(x, list)
    ):
        parts = list(partitura.score.iter_parts(x))
        # set musical beat if requested
        [part.use_musical_beat() for part in parts]
        # get the maximum length of all parts to avoid shorter pianorolls
        end_time = max([int(part.beat_map([part._points[-1].t])) for part in parts])
        # define the parameters of the compute_pianoroll function
        get_pianoroll = partial(
            partitura.utils.compute_pianoroll,
            time_unit=time_unit,
            time_div=time_div,
            piano_range=True,
            remove_silence=False,
            end_time=end_time,
        )
        # compute pianorolls for all separated voices
        separated_prs = np.array([get_pianoroll(part) for part in parts])
        if not all([pr.shape == separated_prs[0].shape for pr in separated_prs]):
            raise Exception(f"Pianorolls of different lenght in {path}")
        # compute mixed pianoroll
        part = partitura.score.merge_parts(parts, reassign="voice")
        # mixed_pr = get_pianoroll(part)
        # compute mixed note array
        mixed_notearray = part.note_array()

        ## compute the voice_pianoroll, that will have the number of the voice in the bin where there is a note, and -1 where there is not
        dense_pianoroll = np.array([pr.todense() for pr in separated_prs])
        negative_pianoroll = np.zeros(dense_pianoroll.shape[1:])
        negative_pianoroll[np.sum(dense_pianoroll, axis=0) == 0] = 1
        voice_pianoroll = (
            np.concatenate(
                [np.expand_dims(negative_pianoroll, axis=0), dense_pianoroll]
            ).argmax(axis=0)
            - 1
        )
        # return {"separated_pianorolls": separated_prs, "mixed_pianoroll": mixed_pr, "mixed_notearray": mixed_notearray, "path": path}
        return {
            "voice_pianoroll": voice_pianoroll,
            "notearray_pitch": mixed_notearray["pitch"].astype(int),
            "notearray_onset_beat": mixed_notearray["onset_beat"].astype(float),
            "notearray_duration_beat": mixed_notearray["duration_beat"].astype(float),
            "notearray_voice": mixed_notearray["voice"].astype(int),
            "path": path,
        }
    elif isinstance(x, str):
        # print(f"Processing {x}")
        return pianorolls_from_part(
            partitura.load_score(x),
            time_unit,
            time_div,
            musical_beat,
            x.split(os.path.sep)[-1],
        )
    else:
        raise TypeError(f"x must be a list of Parts, not {type(x)}")


def pr_to_voice_pred(
    pianoroll: np.ndarray,
    onset_beat: np.ndarray,
    duration_beat: np.ndarray,
    pitch: np.ndarray,
    piano_range: bool,
    time_div: int,
):
    """
    Take the predicted voices from a pianoroll and map them into the note_array.
    Returns a list with a voice for each note in the input note array.

    The input pianoroll has dimension Tx88xV where V is the maximum number of voices.
    For a fixed t and note, it should contains V log probabilities that the note belong to each voice.
    WARNING: this does not work with normal or unnormalized probabilities. Only with log probabilities

    Returns a voice array, one for each note in the same order as the input parameters.
    Voices start from 1.
    """

    # shift in case the first time position is negative (pickup measure)
    positive_onset_beats = onset_beat
    if onset_beat[0] < 0:
        positive_onset_beats = positive_onset_beats - positive_onset_beats[0]
    pr_onset_idxs = torch.round(time_div * positive_onset_beats).int()
    pr_durations = torch.clip(
        torch.round(time_div * duration_beat).int(), min=1, max=None
    )
    pr_offset_idxs = pr_onset_idxs + pr_durations

    pitch_idxs = pitch
    if piano_range:
        pitch_idxs = pitch_idxs - 21  # pianorolls are with 88 only notes

    pred_voice = torch.zeros(pitch.shape, dtype=torch.int64)
    for i, (p, ons, offs) in enumerate(zip(pitch_idxs, pr_onset_idxs, pr_offset_idxs)):
        # get predictions from the pianoroll
        voice = pianoroll[:, ons : offs + 1, p]
        # sum the log probs to get a unique probability for the entire note, and take the max
        pred_voice[i] = torch.sum(voice, axis=1).argmax() + 1

    return pred_voice
