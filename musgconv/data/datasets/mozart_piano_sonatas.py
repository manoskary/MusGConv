from musgconv.utils.graph import *
import os
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
import partitura as pt
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map
from fractions import Fraction as frac
import pandas as pd
from musgconv.utils import hetero_graph_from_note_array, load_score_hgraph
from numpy.lib.recfunctions import structured_to_unstructured

################################################################################
# Converters
################################################################################
str2inttuple = lambda l: tuple() if l == '' else tuple(int(s) for s in l.split(', '))
str2strtuple = lambda l: tuple() if l == '' else tuple(str(s) for s in l.split(', '))
def iterable2str(iterable):
    try:
        return ', '.join(str(s) for s in iterable)
    except:
        return iterable

def int2bool(s):
    try:
        return bool(int(s))
    except:
        return s

def safe_frac(s):
    try:
        return frac(s)
    except:
        return s




################################################################################
# Constants
################################################################################
CONVERTERS = {
    'added_tones': str2inttuple,
    'act_dur': safe_frac,
    'chord_tones': str2inttuple,
    'globalkey_is_minor': int2bool,
    'localkey_is_minor': int2bool,
    'next': str2inttuple,
    'nominal_duration': safe_frac,
    'offset': safe_frac,
    'onset': safe_frac,
    'duration': safe_frac,
    'scalar': safe_frac,}

DTYPES = {
    'alt_label': str,
    'barline': str,
    'bass_note': 'Int64',
    'cadence': str,
    'cadences_id': 'Int64',
    'changes': str,
    'chord': str,
    'chord_type': str,
    'dont_count': 'Int64',
    'figbass': str,
    'form': str,
    'globalkey': str,
    'gracenote': str,
    'harmonies_id': 'Int64',
    'keysig': 'Int64',
    'label': str,
    'localkey': str,
    'mc': 'Int64',
    'midi': 'Int64',
    'mn': 'Int64',
    'notes_id': 'Int64',
    'numbering_offset': 'Int64',
    'numeral': str,
    'pedal': str,
    'playthrough': 'Int64',
    'phraseend': str,
    'relativeroot': str,
    'repeats': str,
    'root': 'Int64',
    'special': str,
    'staff': 'Int64',
    'tied': 'Int64',
    'timesig': str,
    'tpc': 'Int64',
    'voice': 'Int64',
    'voices': 'Int64',
    'volta': 'Int64'
}


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        if isinstance(frac_str, str):
            if frac_str == "":
                return 0.0
            num, denom = frac_str.split('/')
            try:
                leading, num = num.split('.')
                whole = float(leading)
            except ValueError:
                whole = 0
            frac = float(num) / float(denom) if num != "" else 0.
            return whole - frac if whole < 0 else whole + frac
        else:
            return 0.0


def load_tsv(path, index_col=[0, 1], converters={}, dtypes={}, stringtype=False, **kwargs):
    """ Loads the TSV file `path` while applying correct type conversion and parsing tuples.

    Parameters
    ----------
    path : :obj:`str`
        Path to a TSV file as output by format_data().
    index_col : :obj:`list`, optional
        By default, the first two columns are loaded as MultiIndex.
        The first level distinguishes pieces and the second level the elements within.
    converters, dtypes : :obj:`dict`, optional
        Enhances or overwrites the mapping from column names to types included the constants.
    stringtype : :obj:`bool`, optional
        If you're using pandas >= 1.0.0 you might want to set this to True in order
        to be using the new `string` datatype that includes the new null type `pd.NA`.
    """
    conv = dict(CONVERTERS)
    types = dict(DTYPES)
    types.update(dtypes)
    conv.update(converters)
    if stringtype:
        types = {col: 'string' if typ == str else typ for col, typ in types.items()}
    return pd.read_csv(path, sep='\t', index_col=index_col,
                                dtype=types,
                                converters=conv, **kwargs)


def frac_str_to_tuple(x):
    if isinstance(x, str):
        num, denom = x.split("/")
        return [float(num), float(denom)]
    else:
        return [0.0, 0.0]


class MozartPianoSonatas(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/DCMLab/mozart_piano_sonatas/"
        super(MozartPianoSonatas, self).__init__(
            name="MozartPianoSonatas",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)
        self.scores = list()
        self.process()

    def process(self):
        # import git
        #
        # repo = git.Repo(os.path.join(self.raw_dir, self.name))
        # git = repo.git
        # if 'main' in git.branch():
        #     o = repo.remotes.origin
        #     o.fetch()
        #     git.checkout('main')
        #     o.pull()
        #
        # path = os.path.join(self.raw_dir, self.name, "mozart_loader.py")
        # if not os.path.exists(os.path.join(self.raw_dir, self.name, "formatted", "-NTHCEpj_joined.tsv")):
        #     os.chdir(os.path.join(self.raw_dir, self.name))
        #     os.system("python {} -NTHCEpj".format(path))

        self.scores = list()
        for fn in os.listdir(os.path.join(self.save_path, "MS3")):
            if fn.endswith(".mscx"):
                self.scores.append(os.path.join(self.save_path, "MS3", fn))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


# class MozartPianoSonatasCadenceDataset(musgconvDataset):
#     r"""The MozartPianoSonatasCadenceDataset
#
#     Parameters
#     -----------
#     raw_dir : str
#         Raw file directory to download/contains the input data directory.
#         Dataset will search if Asap Dataset contining the scores is already available otherwise it will download it.
#         Default: ~/.musgconv/
#     force_reload : bool
#         Whether to reload the dataset. Default: False
#     verbose : bool
#         Whether to print out progress information. Default: True.
#     nprocs : Number of parallel processes for loading fast.
#     """
#
#     def __init__(self, raw_dir=None, force_reload=False,
#                  verbose=True, nprocs=4):
#         self.base_dataset = MozartPianoSonatas(raw_dir=raw_dir)
#         self.nprocs = nprocs
#         if verbose:
#             print("Loaded MozartPianoSonatas Dataset Successfully, now processing...")
#         self.graphs = list()
#         super(MozartPianoSonatasCadenceDataset, self).__init__(
#             name="MozartPianoSonatasCadenceDataset",
#             raw_dir=raw_dir,
#             force_reload=force_reload,
#             verbose=verbose)
#
#     def process(self):
#         def gcreate(df, pn):
#             fields = ["onset_beat", "duration_beat", "pitch", "ts_beats", "ts_beat_type"]
#             fields = np.dtype(list(map(lambda x: (x, 'f'), fields)))
#             beats = df["beat"].apply(convert_to_float).to_numpy()
#             durs = df["duration"].apply(convert_to_float).to_numpy()
#             pitch = df["midi"].to_numpy().astype(float)
#             ts_beats = df["timesig"].apply(lambda x : frac_str_to_tuple(x)[0]).to_numpy()
#             ts_beat_type = df["timesig"].apply(lambda x: frac_str_to_tuple(x)[1]).to_numpy()
#             na = np.transpose(np.vstack((beats, durs, pitch, ts_beats, ts_beat_type)))
#             na = rfn.unstructured_to_structured(na, dtype=fields)
#             nodes, edges = graph_from_note_array(na)
#             # TODO part from note array.
#             part = pt.musicanalysis.note_array_to_part(na)
#             note_features = select_features(part, "chord")
#             g = ScoreGraph(note_features, edges, name=pn)
#             return g
#
#         path = os.path.join(self.raw_dir, self.base_dataset.name, "scores")
#         df = load_tsv(os.path.join(self.raw_dir, self.base_dataset.name, "formatted", "-NTHCEpj_joined.tsv"))
#         df = df[["beat", "duration", "midi", "timesig", "chord"]]
#         df.dropna(how="any", inplace= True)
#         pieces = np.unique(df.index.get_level_values(0).to_numpy())
#         self.graphs = Parallel(n_jobs=self.nprocs)(
#             delayed(gcreate)(df.loc[pn], pn) for pn in tqdm(pieces, position=0, leave=True))
#
#
#     def has_cache(self):
#         if os.path.exists(self.save_path):
#             return True
#         return False
#
#     def save(self):
#         """save the graph list and the labels"""
#         for g in self.graphs:
#             g.save(os.path.join(self.save_dir, self.name))
#
#     def load(self):
#         self.graphs = list()
#         sdir = os.path.join(self.save_dir, self.name)
#         for fn in os.listdir(sdir):
#             path = os.path.join(sdir, fn)
#             graph = load_score_graph(sdir, fn)
#             self.graphs.append(graph)
#
#     def __getitem__(self, idx):
#         return [[self.graphs[i].x, self.graphs[i].edge_index, self.graphs[i].y, self.graphs[i].mask, self.graphs[i].info["sfn"]] for i in idx]
#
#     def __len__(self):
#         return len(self.graphs)
#
#     @property
#     def save_name(self):
#         return self.name
#
#     @property
#     def features(self):
#         if self.graphs[0].node_features:
#             return self.graphs[0].node_features
#         else:
#             return list(range(self.graphs[0].x.shape[-1]))


class MozartPianoSonatasCadenceGraphDataset(musgconvDataset):
    r"""The Mozart Piano Sonatas Cadence Graph Dataset."""

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, max_size=1000, include_measures=False):
        self.dataset_base = MozartPianoSonatas(raw_dir=raw_dir)
        if verbose:
            print("Loaded Mozart String Quartet Dataset Successfully, now processing...")
        self.graphs = list()
        self.max_size = max_size
        self.include_measures = include_measures
        self.n_jobs = nprocs
        self.problematic_pieces = ['']
        self.stage = "validate"
        self.cadence_to_label = {"nocad": 0, "pac": 1, "iac": 2, "hc": 3}
        self.label_to_cadence = {0: "nocad", 1: "pac", 2: "iac", 3: "hc"}
        super(MozartPianoSonatasCadenceGraphDataset, self).__init__(
            name="MozartPianoSonatasCadenceGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def metadata(self):
        return [
            'K279-1', 'K279-2', 'K279-3', 'K280-1', 'K280-2',
            'K280-3', 'K281-1', 'K281-2', 'K281-3', 'K282-1',
            'K282-2', 'K282-3', 'K283-1', 'K283-2', 'K283-3',
            'K284-1', 'K284-2', 'K284-3', 'K309-1', 'K309-2',
            'K309-3', 'K310-1', 'K310-2', 'K310-3', 'K311-1',
            'K311-2', 'K311-3', 'K330-1', 'K330-2', 'K330-3',
            'K331-1', 'K331-3', 'K332-1', 'K332-2',
            'K332-3', 'K333-1', 'K333-2', 'K333-3', 'K457-1',
            'K457-2', 'K457-3', 'K533-1', 'K533-2', 'K533-3',
            'K545-1', 'K545-2', 'K545-3', 'K570-1', 'K570-2',
            'K570-3', 'K576-1', 'K576-2', 'K576-3'
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

            score = partitura.load_via_musescore(score_fn)
            score = unfold_part_minimal(score)
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
        scores = dict()
        annotations = dict()
        url_base = "https://raw.githubusercontent.com/DCMLab/mozart_piano_sonatas/main/harmonies/"
        for score_fn in self.dataset_base.scores:
            key = os.path.basename(os.path.splitext(score_fn)[0])
            if key not in self.metadata():
                continue
            scores[key] = score_fn
            fn = "{}.harmonies.tsv".format(key)
            annotation_dir = os.path.join(url_base, fn)
            annotation = pd.read_csv(annotation_dir, sep="\t")
            annotations[key] = dict()
            for cadence in self.cadence_to_label.keys():
                if cadence == "nocad":
                    continue
                pre_annotations = annotation["quarterbeats"][annotation["label"].apply(lambda x: cadence.upper() in x)].tolist()
                annotations[key][cadence] = [(eval(pre_ann) if isinstance(pre_ann, str) else pre_ann) for pre_ann in pre_annotations if pre_ann is not np.nan]
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
                    raise IndexError(
                        "Annotated beat {} of does not match with any score beat on score {} for cadence type {}.".format(cad_onset, score_key, key))
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


def unfold_part_minimal(score):
    """Return the "minimally" unfolded part, that is, a copy of the
    part where all segments marked with repeat signs are only inserted once.
Note this might not be musically valid, e.g. a passing a "fine"
        even a first time will stop this unfolding.


    Parameters
    ----------
    part : :class:`Part`
        The Part to unfold.


    Returns
    -------
    unfolded_part : :class:`Part`
        The unfolded Part

    """
    new_partlist = []
    for part in score.parts:
        paths = partitura.score.get_paths(
            part, no_repeats=True, all_repeats=True, ignore_leap_info=True
        )

        unfolded_part = partitura.score.new_part_from_path(paths[0], part)
        new_partlist.append(unfolded_part)
    score.parts = new_partlist
    return score