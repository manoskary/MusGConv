from tqdm.contrib.concurrent import process_map
from musgconv.utils.hgraph import *
import os
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured

class BachWTCDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/manoskary/bach-wtc-fugues"
        super(BachWTCDataset, self).__init__(
            name="BachWTCDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)
        self.scores = []
        self.process()

    def process(self):
        self.scores = []
        base_score_path = os.path.join(self.save_path, "musicxml")
        for fn in os.listdir(base_score_path):
            self.scores.append(os.path.join(base_score_path, fn))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class BachWTCGraphVoiceSeparationDataset(musgconvDataset):
    r"""The Bach 24 Well-Tempered Clavier Fugues Graph Voice Separation Dataset.


    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach WTC Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4):
        self.dataset_base = BachWTCDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Bach WTC Fugues Dataset Successfully, now processing...")
        self.graphs = list()
        self.n_jobs = nprocs
        super(BachWTCGraphVoiceSeparationDataset, self).__init__(
            name="BachWTCGraphVoiceSeparationDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):

        def gfunc(fn):
            if fn.endswith(".krn"):
                g = hetero_graph_from_part(x=os.path.join(path, fn), name=os.path.splitext(fn)[0], features="voice")
                return g

        path = os.path.join(self.raw_dir, "BachWTCDataset", "kern")
        self.graphs = Parallel(self.n_jobs)(delayed(gfunc)(fn) for fn in tqdm(os.listdir(path)))


    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        for g in self.graphs:
            g.save(self.save_path)

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path, fn)
            self.graphs.append(graph)

    def __getitem__(self, idx):
        return [[
            self.graphs[i].x,
            self.graphs[i].edge_index,
            self.graphs[i].y,
            self.graphs[i].get_edges_of_type("consecutive"),
            structured_to_unstructured(self.graphs[i].note_array[["pitch", "onset_div", "duration_div"]]),
            self.graphs[i].name] for i in idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def save_name(self):
        return self.name

    @property
    def features(self):
        if self.graphs[0].node_features:
            return self.graphs[0].node_features
        else:
            return list(range(self.graphs[0].x.shape[-1]))


class BachWTCCadenceGraphDataset(musgconvDataset):
    r"""The Bach 24 Well-Tempered Clavier Fugues Cadence Graph Dataset."""

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, max_size=1000, include_measures=False):
        self.dataset_base = BachWTCDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Bach WTC Fugues Dataset Successfully, now processing...")
        self.graphs = list()
        self.max_size = max_size
        self.include_measures = include_measures
        self.stage = "validate"
        self.n_jobs = nprocs
        self.cadence_to_label = {"nocad": 0, "pac": 1, "iac": 2, "hc": 3}
        self.label_to_cadence = {0: "nocad", 1: "pac", 2: "iac", 3: "hc"}
        super(BachWTCCadenceGraphDataset, self).__init__(
            name="BachWTCCadenceGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def metadata(self):
        return [
        'wtc1f01', 'wtc1f07', 'wtc1f15', 'wtc1f13',
        'wtc1f06', 'wtc1f03', 'wtc1f02', 'wtc1f18',
        'wtc1f17', 'wtc1f09', 'wtc1f24', 'wtc1f10',
        'wtc1f22', 'wtc1f16', 'wtc1f12', 'wtc1f23',
        'wtc1f19', 'wtc1f05', 'wtc1f14', 'wtc1f04',
        'wtc1f08', 'wtc1f20', 'wtc1f21',
        ]

    def process(self):
        self.scores, self.annotations = self.get_annotations()
        process_map(self._process_score, list(self.scores.keys()), max_workers=self.n_jobs)
        self.load()

    def _process_score(self, score_key):
        score_fn = self.scores[score_key]
        annotation = self.annotations[score_key]
        if self._force_reload or not os.path.exists(
                    os.path.join(self.save_path, score_key)):
            try:
                score = partitura.load_score(score_fn)
                if len(score.parts) == 0:
                    print("Something is wrong with the score", score_fn)
                    return
            except:
                print("Something is wrong with the score", score_fn)
                return
            quarter_divs = np.array([p._quarter_durations[0] for p in score.parts])
            new_score = score if np.all(quarter_divs == quarter_divs[0]) else partitura.score.Score(partlist=list(map(lambda x: change_quarter_divs(x, quarter_divs.max()), score.parts)))
            note_array = new_score.note_array(
                include_time_signature=True,
                include_grace_notes=True,
                include_staff=True,
                include_pitch_spelling=True,
            )
            # assert that the note array contains voice information and has no NaNs
            assert np.all(note_array["voice"] >= 0), "Voice information is missing for score {}.".format(score_fn)
            note_features = select_features(note_array, "cadence")
            nodes, edges = hetero_graph_from_note_array(note_array)
            labels = self.get_labels(annotation, new_score, note_array, score_key)
            hg = HeteroScoreGraph(
                note_features,
                edges,
                name=score_key,
                labels=labels,
                note_array=note_array,
            )
            hg.save(self.save_path)
            return

    def get_annotations(self):
        """Data loading for Bach Well Tempered Clavier Fugues.

        Parameters
        ----------
        score_dir : The score Directory.

        Returns
        -------
        scores : dict
            A dictionary with keys of score names and values of score paths.
        annotations : dict
            A dictionary with keys of score names and values Cadence positions.
        """
        import requests, yaml
        scores = dict()
        annotations = dict()
        url_base = "https://gitlab.com/algomus.fr/algomus-data/-/raw/master/fugues/bach-wtc-i/"
        for score_fn in self.dataset_base.scores:
            key = os.path.basename(os.path.splitext(score_fn)[0])
            if key not in self.metadata():
                continue
            scores[key] = score_fn
            fugue_num = key[-2:]
            fn = "{}-bwv{}-ref.dez".format(fugue_num, 845 + int(fugue_num))
            annotation_dir = os.path.join(url_base, fn)
            cond_pac = lambda x: "PAC" in x["tag"] if "tag" in x.keys() else False
            cond_riac = lambda x: "IAC" in x["tag"] if "tag" in x.keys() else False
            cond_hc = lambda x: "HC" in x["tag"] if "tag" in x.keys() else False
            # Retrieve the file content from the URL
            response = requests.get(annotation_dir, allow_redirects=True)
            # Convert bytes to string
            content = response.content.decode("utf-8")
            # Load the yaml
            content = yaml.safe_load(content)
            l = content["labels"]
            annotations[key] = {
                "pac": [dv["start"] for dv in l if
                        (dv['type'] == 'Cadence') and cond_pac(dv)],
                "iac": [dv["start"] for dv in l if
                         (dv['type'] == 'Cadence') and cond_riac(dv)],
                "hc": [dv["start"] for dv in l if
                       (dv['type'] == 'Cadence') and cond_hc(dv)],
                }
        return scores, annotations

    def get_labels(self, cadences, score, note_array, score_key):
        labels = np.zeros(len(note_array), dtype=int)
        p = score[0]
        time_signature = np.array(
            [
                (p.beat_map(ts.start.t), (p.beat_map(ts.end.t) if ts.end else -1), ts.beats, ts.beat_type) for ts in
                p.iter_all(partitura.score.TimeSignature)
            ],
            dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"), ("denominator", "<i4")]
        )
        for key, cad_pos in cadences.items():
            if key not in self.cadence_to_label.keys():
                raise ValueError("Cadence position {} is not valid.".format(cad_pos))
            if cad_pos == []:
                continue

            if (not np.all(note_array["onset_beat"] >= 0)):
                cad_pos += min(note_array["onset_beat"])

            # Corrections of annotations with respect to time signature.
            if time_signature["denominator"][0] == 2:
                cad_pos = list(map(lambda x: x / 2, cad_pos))
            elif time_signature["denominator"][0] == 8:
                if time_signature["nominator"][0] in [6, 9, 12]:
                    cad_pos = list(map(lambda x: 2 * x / 3, cad_pos))
                else:
                    cad_pos = list(map(lambda x: 2 * x, cad_pos))

            for cad_onset in cad_pos:
                labels[np.where(note_array["onset_beat"] == cad_onset)[0]] = self.cadence_to_label[key]
                # check for false annotation that does not have match
                if np.all((note_array["onset_beat"] == cad_onset) == False):
                    raise IndexError("Annotated beat {} does not match with any score beat of score {} for cadence type {}.".format(cad_onset, score_key, key))
        return labels

    def load(self):
        for fn in os.listdir(self.save_path):
            path_graph = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path_graph, fn)
            self.graphs.append(graph)

    def has_cache(self):
        if all([os.path.exists(os.path.join(self.save_path, fn)) for fn in self.metadata()]):
            return True
        return False

    def set_split(self, stage):
        self.stage = stage

    def __getitem__(self, idx):
        return [self.get_graph_attr(i, self.stage == "train") for i in idx]

    def get_graph_attr(self, idx, batch):
        out = dict()
        if self.graphs[idx].x.size(0) > self.max_size and batch:
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            truth_edge_indices = np.isin(self.graphs[idx].truth_edges[0], indices) & np.isin(
                self.graphs[idx].truth_edges[1], indices)
            pot_edge_indices = np.isin(self.graphs[idx].pot_edges[0], indices) & np.isin(
                self.graphs[idx].pot_edges[1], indices)
            out["x"] = self.graphs[idx].x[indices]
            out["edge_index"] = self.graphs[idx].edge_index[:, edge_indices] - random_idx
            out["y"] = self.graphs[idx].y[pot_edge_indices]
            out["edge_type"] = self.graphs[idx].edge_type[edge_indices]
            out["note_array"] = structured_to_unstructured(
                self.graphs[idx].note_array[
                    ["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
            )[indices]
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                measure_edges = torch.tensor(self.graphs[idx].measure_edges)
                measure_nodes = torch.tensor(self.graphs[idx].measure_nodes).squeeze()
                beat_edges = torch.tensor(self.graphs[idx].beat_edges)
                beat_nodes = torch.tensor(self.graphs[idx].beat_nodes).squeeze()
                beat_edge_indices = torch.isin(beat_edges[0], indices)
                beat_node_indices = torch.isin(beat_nodes, torch.unique(beat_edges[1][beat_edge_indices]))
                min_beat_idx = torch.where(beat_node_indices)[0].min()
                max_beat_idx = torch.where(beat_node_indices)[0].max()
                measure_edge_indices = torch.isin(measure_edges[0], indices)
                measure_node_indices = torch.isin(measure_nodes, torch.unique(measure_edges[1][measure_edge_indices]))
                if measure_node_indices.sum() == 0:
                    print("No measure edges in graph", self.graphs[idx].name)
                min_measure_idx = torch.where(measure_node_indices)[0].min()
                max_measure_idx = torch.where(measure_node_indices)[0].max()
                out["beat_nodes"] = beat_nodes[min_beat_idx:max_beat_idx + 1] - min_beat_idx
                out["beat_edges"] = torch.vstack(
                    (beat_edges[0, beat_edge_indices] - random_idx, beat_edges[1, beat_edge_indices] - min_beat_idx))
                out["measure_nodes"] = measure_nodes[min_measure_idx:max_measure_idx + 1] - min_measure_idx
                out["measure_edges"] = torch.vstack((measure_edges[0, measure_edge_indices] - random_idx,
                                                     measure_edges[1, measure_edge_indices] - min_measure_idx))
        else:
            out["x"] = self.graphs[idx].x
            out["edge_index"] = self.graphs[idx].edge_index
            out["y"] = self.graphs[idx].y
            out["edge_type"] = self.graphs[idx].edge_type
            out["note_array"] = structured_to_unstructured(
                self.graphs[idx].note_array[
                    ["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
            )
            out["name"] = self.graphs[idx].name
            if self.include_measures:
                out["beat_nodes"] = torch.tensor(self.graphs[idx].beat_nodes)
                out["beat_edges"] = torch.tensor(self.graphs[idx].beat_edges)
                out["measure_nodes"] = torch.tensor(self.graphs[idx].measure_nodes)
                out["measure_edges"] = torch.tensor(self.graphs[idx].measure_edges)
        return out

    def __len__(self):
        return len(self.graphs)

    @property
    def save_name(self):
        return self.name

    @property
    def features(self):
        return self.graphs[0].x.shape[-1]


def change_quarter_divs(part, quarter_divs):
    orig_quarter_div = part._quarter_durations[0]
    ratio = quarter_divs / orig_quarter_div
    if ratio == 1:
        return part
    elif not ratio.is_integer():
        raise ValueError("The ratio between the original quarter division and the new one is not an integer.")

    for el in part.iter_all():
        el.start.t = el.start.t * ratio
        if el.end is not None:
            el.end.t = el.end.t * ratio
    return part