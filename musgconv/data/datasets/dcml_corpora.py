import os, random
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
from tqdm.contrib.concurrent import process_map
import partitura as pt
from musgconv.utils.hgraph import HeteroScoreGraph, hetero_graph_from_note_array, load_score_hgraph, select_features
from musgconv.utils import score_graph_to_pyg
from numpy.lib.recfunctions import structured_to_unstructured
import numpy as np
import torch


class DCMLPianoCorporaDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/fosfrancesco/piano_corpora_dcml.git"
        super(DCMLPianoCorporaDataset, self).__init__(
            name="DCMLPianoCorporaDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            is_zip=False)
        self.scores = []
        self.collections = []
        self.composers = []
        self.score_names = []
        self.period = "Unknown"
        self.type = "Unknown"
        self.process()

    def process(self):
        self.scores = []
        self.collections = []
        self.composers = []
        self.score_names = []
        for fn in os.listdir(os.path.join(self.save_path, "scores")):
            for score_fn in os.listdir(os.path.join(self.save_path, "scores", fn)):
                if score_fn.endswith(".musicxml"):
                    composer = fn.split("_")[0]
                    rest = "_".join(fn.split("_")[1:])
                    self.composers.append(composer)
                    self.score_names.append(rest + "_" + os.path.splitext(score_fn)[0])
                    self.scores.append(os.path.join(self.save_path, "scores", fn, score_fn))
                    self.collections.append(fn)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class DCMLGraphDataset(musgconvDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, n_jobs=12, include_measures=False, max_size=200):
        self.dcml_dataset = DCMLPianoCorporaDataset(raw_dir=raw_dir)
        self.n_jobs = n_jobs
        self.graphs = []
        self.max_size = max_size
        self.stage = "validate"
        self._force_reload = force_reload
        self.include_measures = include_measures
        self.prob_scores = ["l000_soirs", "l099_cahier", "l111-01_images_cloches", "l111-02_images_lune", "l111-03_images_poissons", "l117-11_preludes_danse", "l123-01_preludes_brouillards", "l123-02_preludes_feuilles", "l123-03_preludes_puerta", "l123-04_preludes_fees", "l123-05_preludes_bruyeres", "l123-06_preludes_general", "l123-07_preludes_terrasse", "l123-08_preludes_ondine", "l123-09_preludes_hommage", "l123-10_preludes_canope", "l123-11_preludes_tierces", "l123-12_preludes_feux", "l136-10_etudes_sonorites", "op43n06", "op34n03", "op35n04"]
        self.composer_to_label = dict()
        self.label_to_composer = dict()
        if verbose:
            print("Loaded DCML Piano Corpus Successfully, now processing...")
        super(DCMLGraphDataset, self).__init__(
            name="DCMLGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        zfiles = zip(self.dcml_dataset.scores, self.dcml_dataset.composers, self.dcml_dataset.score_names)
        process_map(self._process_score, zfiles, max_workers=self.n_jobs)
        self.load()

    def _process_score(self, data):
        score_fn, composer, score_name = data
        if score_name in self.prob_scores:
            return
        if self._force_reload or not (os.path.exists(os.path.join(self.save_path, score_name + ".pt"))):
            score = pt.load_score(score_fn)
            note_array = score.note_array(include_time_signature=True)
            note_features = select_features(note_array, "voice")
            nodes, edges = hetero_graph_from_note_array(note_array)
            hg = HeteroScoreGraph(
                note_features,
                edges,
                name=score_name,
                labels=None,
                note_array=note_array,
            )
            hg.y = composer
            pg_graph = score_graph_to_pyg(hg)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_path = os.path.join(self.save_path, pg_graph["name"] + ".pt")
            # add global composer label to the graph
            pg_graph["y"] = composer
            torch.save(pg_graph, file_path)
        return

    def save(self):
        pass

    def load(self):
        # Filter for composers
        composers, counts = np.unique(np.array(self.dcml_dataset.composers), return_counts=True)
        rejected_composers = composers[np.where(counts < 4)]
        for fn in os.listdir(self.save_path):
            if fn in self.prob_scores:
                continue
            path_graph = os.path.join(self.save_path, fn)
            graph = torch.load(path_graph)
            composer = graph.y
            # filter composer list
            if composer in rejected_composers:
                continue
            if composer not in self.composer_to_label.keys():
                max_i = max(self.composer_to_label.values()) if len(self.composer_to_label) > 0 else -1
                self.composer_to_label[composer] = max_i + 1
                self.label_to_composer[max_i + 1] = composer
            self.graphs.append(graph)

    def has_cache(self):
        if all(
                [os.path.exists(os.path.join(self.save_path, path))
                 for path in self.dcml_dataset.score_names]
        ):
            return True
        return False

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        out = []
        for i in idx:
            y = self.composer_to_label[self.graphs[i].y]  if not isinstance(self.graphs[i].y, int) else self.graphs[i].y
            self.graphs[i].y = y
            out.append([self.graphs[i]])

        return [[self.graphs[i]] for i in idx]
        # if isinstance(idx, int):
        #     return self.get_graph_attr(idx, self.stage == "train")
        # return [self.get_graph_attr(i, self.stage == "train") for i in idx]

    def set_split(self, stage="train"):
        self.stage = stage

    def get_graph_attr(self, idx, batch=True):
        out = dict()
        if self.graphs[idx].x.size(0) > self.max_size and batch:
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            out["x"] = self.graphs[idx].x[indices]
            out["edge_index"] = self.graphs[idx].edge_index[:, edge_indices] - random_idx
            out["y"] = self.composer_to_label[self.graphs[idx].y]
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
            out["y"] = self.composer_to_label[self.graphs[idx].y]
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

    @property
    def features(self):
        return self.graphs[0]["note"].x.shape[-1]

    @property
    def n_classes(self):
        return len(self.composer_to_label)