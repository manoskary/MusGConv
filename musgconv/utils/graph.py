import os
import random, string
import pickle
import warnings
import partitura.io.exportparangonada
import torch_geometric as pyg
from musgconv.utils.general import exit_after
from musgconv.descriptors.general import *
import torch
from typing import Union
from musgconv.utils.hgraph import HeteroScoreGraph


class ScoreGraph(object):
    def __init__(
        self,
        note_features,
        edges,
        name=None,
        note_array=None,
        edge_weights=None,
        labels=None,
        mask=None,
        info={},
    ):
        self.node_features = note_array.dtype.names if note_array.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [
                feat
                for feat in self.node_features
                if note_features.dtype.fields[feat][0] != np.dtype("U256")
            ]
            self.features = self.features[self.node_features]
        self.x = torch.from_numpy(
            np.asarray(
                rfn.structured_to_unstructured(self.features)
                if self.node_features
                else self.features
            )
        )
        self.note_array = note_array
        self.edge_index = torch.from_numpy(edges).long()
        self.edge_weights = (
            edge_weights if edge_weights is None else torch.from_numpy(edge_weights)
        )
        self.name = name
        self.mask = mask
        self.info = info
        self.y = labels if labels is None else torch.from_numpy(labels)

    def adj(self):
        # ones = np.ones(len(self.edge_index[0]), np.uint32)
        # matrix = sp.coo_matrix((ones, (self.edge_index[0], self.edge_index[1])))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(
            self.edge_index, ones, (len(self.x), len(self.x))
        )
        return matrix

    def save(self, save_dir):
        save_name = (
            self.name
            if self.name
            else "".join(random.choice(string.ascii_letters) for i in range(10))
        )
        (
            os.makedirs(os.path.join(save_dir, save_name))
            if not os.path.exists(os.path.join(save_dir, save_name))
            else None
        )
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, self.edge_index.numpy())
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(
                open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"),
                self.edge_weights.numpy(),
            )
        if isinstance(self.note_array, np.ndarray):
            np.save(
                open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"),
                self.note_array,
            )
        with open(os.path.join(save_dir, save_name, "graph_info.pkl"), "wb") as handle:
            pickle.dump(
                {
                    "node_features": self.node_features,
                    "mask": self.mask,
                    "info": self.info,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


@exit_after(10)
def load_score_graph(load_dir, name=None):
    path = (
        os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    )
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = (
        np.load(open(os.path.join(path, "y.npy"), "rb"))
        if os.path.exists(os.path.join(path, "y.npy"))
        else None
    )
    edge_weights = (
        np.load(open(os.path.join(path, "edge_weights.npy"), "rb"))
        if os.path.exists(os.path.join(path, "edge_weights.npy"))
        else None
    )
    name = name if name else os.path.basename(path)
    return ScoreGraph(
        note_array=x,
        edges=edge_index,
        name=name,
        labels=y,
        edge_weights=edge_weights,
        mask=graph_info["mask"],
        info=graph_info["info"],
    )


def check_note_array(na):
    dtypes = na.dtype.names
    if not all(
        [
            x in dtypes
            for x in ["onset_beat", "duration_beat", "ts_beats", "ts_beat_type"]
        ]
    ):
        raise (TypeError("The given Note array is missing necessary fields."))


def graph_from_note_array(note_array, rest_array=None, norm2bar=True):
    """Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    """

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where(
            (
                np.isclose(
                    note_array["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04
                )
                == True
            )
            & (note_array["pitch"] != x["pitch"])
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(
            np.isclose(
                note_array["onset_beat"],
                x["onset_beat"] + x["duration_beat"],
                rtol=1e-04,
                atol=1e-04,
            )
            == True
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            for j in np.where(
                np.isclose(
                    rest_array["onset_beat"],
                    x["onset_beat"] + x["duration_beat"],
                    rtol=1e-04,
                    atol=1e-04,
                )
                == True
            )[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)

        for j in np.where(
            (x["onset_beat"] < note_array["onset_beat"])
            & (x["onset_beat"] + x["duration_beat"] > note_array["onset_beat"])
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

    if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
        for i, r in enumerate(rest_array):
            for j in np.where(
                np.isclose(
                    note_array["onset_beat"],
                    r["onset_beat"] + r["duration_beat"],
                    rtol=1e-04,
                    atol=1e-04,
                )
                == True
            )[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)

        feature_fn = [
            dname
            for dname in note_array.dtype.names
            if dname not in rest_array.dtype.names
        ]
        if feature_fn:
            rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
            rest_feature_zeros = rfn.unstructured_to_structured(
                rest_feature_zeros, dtype=list(map(lambda x: (x, "<4f"), feature_fn))
            )
            rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
    else:
        end_times = note_array["onset_beat"] + note_array["duration_beat"]
        for et in np.sort(np.unique(end_times))[:-1]:
            if et not in note_array["onset_beat"]:
                scr = np.where(end_times == et)[0]
                diffs = note_array["onset_beat"] - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        edg_src.append(i)
                        edg_dst.append(j)

    edges = np.array([edg_src, edg_dst])
    # Resize Onset Beat to bar
    if norm2bar:
        note_array["onset_beat"] = np.mod(
            note_array["onset_beat"], note_array["ts_beats"]
        )
        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            rest_array["onset_beat"] = np.mod(
                rest_array["onset_beat"], rest_array["ts_beats"]
            )

    nodes = np.hstack((note_array, rest_array))
    return nodes, edges


@exit_after(60)
def graph_from_part(
    x: Union[Union[partitura.score.Part, partitura.score.PartGroup], np.ndarray],
    name=None,
    norm2bar=True,
    include_rests=False,
    labels=None,
) -> ScoreGraph:
    if (
        isinstance(x, partitura.score.Score)
        or isinstance(x, partitura.score.Part)
        or isinstance(x, partitura.score.PartGroup)
        or isinstance(x, list)
    ):
        part = x
        part = partitura.score.merge_parts(part)
        # TODO: discuss, do we need to unfold?
        part = partitura.score.unfold_part_maximal(part)
        note_array = part.note_array(
            include_time_signature=True, include_grace_notes=True, include_staff=True
        )
        note_features, _ = voice_separation_features(part)
        if include_rests:
            rest_array = part.rest_array(
                include_time_signature=True,
                include_grace_notes=True,
                collapse=True,
                include_staff=True,
            )
            if labels is None:
                labels = np.vstack(
                    (
                        rfn.structured_to_unstructured(note_array[["voice", "staff"]]),
                        rfn.structured_to_unstructured(rest_array[["voice", "staff"]]),
                    )
                )
        else:
            rest_array = None
            if labels is None:
                labels = rfn.structured_to_unstructured(note_array[["voice", "staff"]])
    elif isinstance(x, str):
        return graph_from_part(partitura.load_score(x), name, norm2bar)
    else:
        check_note_array(x)
        note_array = note_features = x
        rest_array = None
        if labels is None:
            labels = rfn.structured_to_unstructured(note_array[["voice", "staff"]])
    nodes, edges = graph_from_note_array(note_array, rest_array, norm2bar=norm2bar)
    return ScoreGraph(note_features, edges, name=name, labels=labels)


def get_matched_performance_idx(part, ppart, alignment):
    # remove repetitions from aligment note ids
    for a in alignment:
        if a["label"] == "match":
            a["score_id"] = str(a["score_id"])

    part_by_id = dict((n.id, n) for n in part.notes_tied)
    ppart_by_id = dict((n["id"], n) for n in ppart.notes)

    # pair matched score and performance notes
    note_pairs = [
        (part_by_id[a["score_id"]], ppart_by_id[a["performance_id"]])
        for a in alignment
        if (a["label"] == "match" and a["score_id"] in part_by_id)
    ]

    # sort according to onset (primary) and pitch (secondary)
    pitch_onset = [(sn.midi_pitch, sn.start.t) for sn, _ in note_pairs]
    sort_order = np.lexsort(list(zip(*pitch_onset)))
    pnote_ids = [note_pairs[i][1].id for i in sort_order]
    return pnote_ids


@exit_after(60)
def pgraph_from_part(
    sfn, pfn, afn, features=None, name=None, norm2bar=True, include_rests=False
) -> ScoreGraph:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spart = partitura.load_score(sfn)
        spart = partitura.score.merge_parts(spart)
        spart = partitura.score.unfold_part_maximal(spart)
        note_array = spart.note_array(
            include_time_signature=True, include_grace_notes=True
        )
        nid_dict = dict((n, i) for i, n in enumerate(note_array["id"]))
        ppart = partitura.load_performance_midi(pfn)
        alignment = partitura.io.exportparangonada.load_alignment_from_ASAP(afn)
        perf_array, snote_idx = partitura.musicanalysis.encode_performance(
            spart, ppart[0], alignment
        )
        matched_score_idx = np.array([nid_dict[nid] for nid in snote_idx])
        note_features, _ = select_features(spart, features)
        rest_array = None
        labels = rfn.structured_to_unstructured(perf_array)
        nodes, edges = graph_from_note_array(note_array, rest_array, norm2bar=norm2bar)
        graph_info = {"sfn": sfn, "pfn": pfn}
        return ScoreGraph(
            note_features,
            edges,
            name=name,
            labels=labels,
            mask=matched_score_idx,
            info=graph_info,
        )


@exit_after(60)
def agraph_from_part(
    sfn, pfn, afn, features=None, name=None, norm2bar=True, include_rests=False
) -> ScoreGraph:
    spart = partitura.load_score(sfn)
    spart = partitura.score.merge_parts(spart)
    spart = partitura.score.unfold_part_maximal(spart)
    note_array = spart.note_array(include_time_signature=True, include_grace_notes=True)
    nid_dict = dict((n, i) for i, n in enumerate(note_array["id"]))
    ppart = partitura.load_performance_midi(pfn)
    alignment = partitura.io.exportparangonada.load_alignment_from_ASAP(afn)
    perf_array, snote_idx = partitura.musicanalysis.encode_performance(
        spart, ppart, alignment
    )
    pnote_array = ppart.note_array()
    pid_dict = dict((n, i) for i, n in enumerate(pnote_array["id"]))
    pnote_array = pnote_array[["onset_sec", "pitch", "velocity"]]
    matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_idx])
    pnote_idx = get_matched_performance_idx(spart, ppart, alignment)
    matched_performance_idx = np.array([pid_dict[nid] for nid in pnote_idx])
    note_features = select_features(spart, features)
    rest_array = None
    labels = rfn.structured_to_unstructured(pnote_array)
    nodes, edges = graph_from_note_array(note_array, rest_array, norm2bar=norm2bar)
    return ScoreGraph(
        note_features,
        edges,
        name=name,
        labels=labels,
        mask=(matched_subset_idxs, matched_performance_idx),
    )


def rec_dir_search(par_dir, doc_type, save_path, result=[]):
    for cdir in os.listdir(par_dir):
        path = os.path.join(par_dir, cdir)
        if os.path.isdir(path):
            result = rec_dir_search(path, doc_type, result)
        else:
            if path.endswith(doc_type):
                try:
                    name = (
                        os.path.basename(os.path.dirname(os.path.dirname(par_dir)))
                        + "_"
                        + os.path.basename(os.path.dirname(par_dir))
                        + "_"
                        + os.path.basename(par_dir)
                    )
                    graph = graph_from_part(path, name)
                    graph.save(save_path)
                except:
                    print("Graph Creation failed on {}".format(path))
    return result


def score_graph_to_pyg(
    score_graph: Union[ScoreGraph, HeteroScoreGraph]
):
    """
    Converts a ScoreGraph to a PyTorch Geometric graph.
    Parameters
    ----------
    score_graph : ScoreGraph
        The ScoreGraph to convert
    """
    # edge_index = score_graph.edge_index.clone().t().contiguous()
    # edge_attr = (
    #     score_graph.edge_weights.clone()
    #     if score_graph.edge_weights is not None
    #     else None
    # )
    if isinstance(score_graph, HeteroScoreGraph):
        data = pyg.data.HeteroData()
        data["note"].x = score_graph.x.clone()
        # data["note"].y = y
        # add edges
        for e_type in score_graph.etypes.keys():
            data["note", e_type, "note"].edge_index = score_graph.get_edges_of_type(
                e_type
            )
        # add pitch, onset, offset info in divs that is necessary for evaluation
        data["note"].pitch = torch.from_numpy(score_graph.note_array["pitch"].copy())
        data["note"].onset_div = torch.from_numpy(score_graph.note_array["onset_div"].copy())
        data["note"].duration_div = torch.from_numpy(score_graph.note_array["duration_div"].copy())
        data["note"].onset_beat = torch.from_numpy(score_graph.note_array["onset_beat"].copy())
        data["note"].duration_beat = torch.from_numpy(score_graph.note_array["duration_beat"].copy())
        data["note"].ts_beats = torch.from_numpy(score_graph.note_array["ts_beats"].copy())
        # # add edges that will be used during evaluation
        # data["pot_edges"] = score_graph.pot_edges.clone().contiguous()
        # data["truth_edges_mask"] = score_graph.truth_edges_mask.clone().contiguous()
        # data["truth_edges"] = score_graph.truth_edges.clone().contiguous()
    else:
        raise ValueError("Only HeteroScoreGraph is supported for now")
        # data = pyg.data.Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

    # adding various graph info
    for key, value in vars(score_graph).items():
        if key not in ["x", "edge_index", "y", "edge_attr", "edge_weights", "node_features", "etypes", "note_array"]:	
            if isinstance(value, (np.ndarray, np.generic) ):
                data[key] = torch.tensor(value)
            else:
                data[key] = value
    return data
