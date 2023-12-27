import os
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
from musgconv.data.vocsep import GraphVoiceSeparationDataset
from numpy.lib.recfunctions import structured_to_unstructured
from tqdm.contrib.concurrent import process_map
from musgconv.utils.hgraph import *


class MozartStringQuartetDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/manoskary/humdrum-mozart-quartets"
        super(MozartStringQuartetDataset, self).__init__(
            name="MozartStringQuartets",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.scores = list()
        self.collections = list()
        self.process()

    def process(self):
        root = os.path.join(self.raw_path, "musicxml")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".musicxml")]
        self.collections = ["mozart"]*len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class MozartStringQuartetGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2, include_measures=False):
        r"""The Mozart String Quartet Graph Voice Separation Dataset.

    Four-part Mozart string quartets digital edition of the quartets composed by Wolfgang Amadeus Mozart,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Mozart String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = MozartStringQuartetDataset(raw_dir=raw_dir)
        super(MozartStringQuartetGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=False,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures,
        )


class MozartStringQuartetPGGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2):
        r"""The Mozart String Quartet Graph Voice Separation Dataset.

    Four-part Mozart string quartets digital edition of the quartets composed by Wolfgang Amadeus Mozart,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Mozart String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = MozartStringQuartetDataset(raw_dir=raw_dir)
        super().__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=True,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
        )



class MozartStringQuartetCadenceGraphDataset(musgconvDataset):
    r"""The Bach 24 Well-Tempered Clavier Fugues Cadence Graph Dataset."""

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, max_size=1000, include_measures=False):
        self.dataset_base = MozartStringQuartetDataset(raw_dir=raw_dir)
        if verbose:
            print("Loaded Mozart String Quartet Dataset Successfully, now processing...")
        self.graphs = list()
        self.max_size = max_size
        self.include_measures = include_measures
        self.n_jobs = nprocs
        self.stage = "validate"
        self.problematic_pieces = ['']
        self.cadence_to_label = {"nocad": 0, "pac": 1, "iac": 2, "hc": 3}
        self.label_to_cadence = {0: "nocad", 1: "pac", 2: "iac", 3: "hc"}
        super(MozartStringQuartetCadenceGraphDataset, self).__init__(
            name="MozartStringQuartetCadenceGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def metadata(self):
        return [
            'k590-01', 'k155-02', 'k156-01', 'k080-02', 'k172-01',
            'k171-01', 'k172-04', 'k157-01', 'k589-01', 'k458-01',
            'k169-01', 'k387-01', 'k158-01', 'k157-02', 'k171-03',
            'k159-02', 'k428-02', 'k173-01', 'k499-03', 'k156-02',
            'k168-01', 'k080-01', 'k421-01', 'k171-04', 'k168-02',
            'k428-01', 'k499-01', 'k172-02', 'k465-04', 'k155-01',
            'k465-01', 'k159-01'
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
            labels = self.get_labels(annotation, note_array, score_key)
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
        """Data loading for Mozart String quartets.

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
        url_base = "https://gitlab.com/algomus.fr/algomus-data/-/raw/master/quartets/mozart/"
        for score_fn in self.dataset_base.scores:
            key = os.path.basename(os.path.splitext(score_fn)[0])
            if key not in self.metadata():
                continue
            scores[key] = score_fn
            # split key between "k" and "-"
            quartet_number, movement_number = key.split("-")
            quartet_number = quartet_number.replace("k0", "k")
            movement_number = str(int(movement_number))
            fn = "{}.{}-ref.dez".format(quartet_number, movement_number)
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
                         (dv['type'] == 'Cadence') and cond_hc(dv)]
                }
        return scores, annotations

    def get_labels(self, cadences, note_array, score_key):
        labels = np.zeros(len(note_array), dtype=int)
        for key, cad_pos in cadences.items():
            if key not in self.cadence_to_label.keys():
                raise ValueError("Cadence position {} is not valid.".format(cad_pos))
            if cad_pos == []:
                continue

            if (not np.all(note_array["onset_quarter"] >= 0)):
                cad_pos += min(note_array["onset_quarter"])

            for cad_onset in cad_pos:
                labels[np.where(note_array["onset_quarter"] == cad_onset)[0]] = self.cadence_to_label[key]
                # check for false annotation that does not have match
                if np.all(note_array["onset_quarter"] != cad_onset):
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