from musgconv.utils.graph import *
from musgconv.utils.hgraph import *
import os
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
from numpy.lib.recfunctions import structured_to_unstructured
from musgconv.data.vocsep import GraphVoiceSeparationDataset
from tqdm.contrib.concurrent import process_map



class HaydnStringQuartetDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/manoskary/humdrum-haydn-quartets"
        super(HaydnStringQuartetDataset, self).__init__(
            name="HaydnStringQuartets",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.process()

    def process(self):
        root = os.path.join(self.raw_path, "kern")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".krn")]
        self.collections = ["haydn"]*len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False




class HaydnStringQuartetCadenceAnnotations(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://gitlab.cp.jku.at/datasets/haydn_string_quartets.git"
        super(HaydnStringQuartetCadenceAnnotations, self).__init__(
            name="HaydnStringQuartetCadenceAnnotations",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.process()

    def process(self):
        root = os.path.join(self.raw_path, "annotations", "cadences_keys")
        self.annotations = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".csv")]

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class HaydnStringQuartetGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2, include_measures=False):
        r"""The Haydn String Quartet Graph Voice Separation Dataset.

    Four-part Haydn string quartets digital edition of the quartets composed by Joseph Haydn,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Haydn String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
        super(HaydnStringQuartetGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=False,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures,
        )

class HaydnStringQuartetPGVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2):
        r"""The Haydn String Quartet Graph Voice Separation Dataset.

    Four-part Haydn string quartets digital edition of the quartets composed by Joseph Haydn,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Haydn String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
        super().__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=True,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
        )


class HaydnStringQuartetsCadenceGraphDataset(musgconvDataset):
    r"""The Haydn String Quartet Cadence Graph Dataset."""

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, max_size=1000, include_measures=False):
        self.dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
        self.annotations_base = HaydnStringQuartetCadenceAnnotations(raw_dir=raw_dir)
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
        super(HaydnStringQuartetsCadenceGraphDataset, self).__init__(
            name="HaydnStringQuartetsCadenceGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def metadata(self):
        return [
            'haydn_op064_no06_mv01', 'haydn_op050_no06_mv01',
            'haydn_op020_no06_mv02', 'haydn_op020_no01_mv04',
            'haydn_op076_no05_mv02', 'haydn_op020_no05_mv01',
            'haydn_op017_no02_mv01', 'haydn_op033_no02_mv01',
            'haydn_op054_no03_mv01', 'haydn_op050_no01_mv01',
            'haydn_op017_no06_mv01', 'haydn_op064_no04_mv01',
            'haydn_op064_no04_mv04', 'haydn_op017_no05_mv01',
            'haydn_op064_no03_mv04', 'haydn_op054_no02_mv01',
            'haydn_op055_no02_mv02', 'haydn_op064_no03_mv01',
            'haydn_op033_no03_mv03', 'haydn_op074_no01_mv01',
            'haydn_op054_no01_mv01', 'haydn_op076_no02_mv01',
            'haydn_op033_no05_mv02', 'haydn_op055_no01_mv02',
            'haydn_op054_no01_mv02', 'haydn_op050_no02_mv01',
            'haydn_op050_no03_mv04', 'haydn_op020_no04_mv04',
            'haydn_op033_no01_mv03', 'haydn_op033_no05_mv01',
            'haydn_op050_no06_mv02', 'haydn_op020_no03_mv04',
            'haydn_op076_no04_mv01', 'haydn_op050_no05_mv04',
            'haydn_op033_no01_mv01', 'haydn_op054_no03_mv04',
            'haydn_op050_no04_mv01', 'haydn_op050_no02_mv04',
            'haydn_op017_no01_mv01', 'haydn_op033_no04_mv01',
            'haydn_op017_no03_mv04', 'haydn_op050_no01_mv04',
            'haydn_op055_no03_mv01', 'haydn_op074_no01_mv02',
            'haydn_op020_no03_mv03'
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

            score = partitura.load_score(score_fn)
            if len(score.parts) == 0:
                print("Something is wrong with the score {} it has no parts.".format(score_fn))
                return

            note_array = score.note_array(
                include_time_signature=True,
                include_grace_notes=True,
                include_staff=True,
                include_pitch_spelling=True,
            )
            # assert that the note array contains voice information and has no NaNs
            assert np.all(note_array["voice"] >= 0), "Voice information is missing for score {}.".format(score_fn)
            note_features = select_features(note_array, "cadence")
            nodes, edges = hetero_graph_from_note_array(note_array)
            labels = self.get_labels(annotation, score, note_array)
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
        import pandas as pd

        scores = {}
        for score_fn in self.dataset_base.scores:
            score_key = os.path.basename(os.path.splitext(score_fn)[0])

            op_num, rest = score_key.split("n") if "n" in score_key else (score_key[:4], score_key[4:])
            no_num, mov_num = rest.split("-")
            if no_num == "" or mov_num == "":
                continue
            new_key = "haydn_op{:03d}_no{:02d}_mv{:02d}".format(int(op_num[2:]), int(no_num), int(mov_num))
            if new_key not in self.metadata():
                continue
            scores[new_key] = score_fn
        annotations = {os.path.basename(os.path.splitext(annotation_fn)[0])[:-5] : pd.read_csv(annotation_fn, sep=",", encoding='cp1252') for annotation_fn in self.annotations_base.annotations}

        return scores, annotations

    def get_labels(self, cadences, score, note_array):
        import pandas as pd
        measures = {int(m.number): score[0].beat_map(m.start.t) for m in score[0].measures}
        cadence_pos = dict()
        sub_table = cadences[np.where(cadences["Descriptive Information"] == "Cad Cat.")[0][0]:].to_numpy()
        new_df_keys = sub_table[0, :].tolist()
        new_df_values = [sub_table[1:, i].tolist() for i in range(len(new_df_keys))]
        new_df = pd.DataFrame(data=dict(zip(new_df_keys, new_df_values))).dropna(how="all", axis=1)
        new_df = new_df.dropna(how="all", axis=0)
        for cad_type in self.cadence_to_label.keys():
            if cad_type == "nocad":
                continue
            idx = np.where(new_df["Cad Cat."].apply(lambda x: cad_type.upper() in x))[0]
            bars = list(map(lambda x: int(x), new_df["Bar #"][idx]))
            beats = list(map(lambda x: float(x) - 1, new_df["Pulse #"][idx]))
            cadence_pos[cad_type] = list(zip(bars, beats))

        labels = np.zeros(len(note_array), dtype=int)
        for key, cad_pos in cadence_pos.items():
            if key not in self.cadence_to_label.keys():
                raise ValueError("Cadence position {} is not valid.".format(cad_pos))
            if cad_pos == []:
                continue

            # if (not np.all(note_array["onset_quarter"] >= 0)):
            #     cad_pos += min(note_array["onset_quarter"])

            for cad_measure, cad_beat in cad_pos:
                cad_onset = measures[cad_measure] + cad_beat
                labels[np.where(note_array["onset_beat"] == cad_onset)[0]] = self.cadence_to_label[key]
                # check for false annotation that does not have match
                if np.all(note_array["onset_beat"] != cad_onset):
                    raise IndexError("Annotated measure {} and beat {} does not match with any score beat for cadence {}.".format(cad_measure, cad_beat, key))
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

