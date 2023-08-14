from musgconv.utils.graph import *
from musgconv.utils.hgraph import *
from musgconv.utils.pianoroll import pianorolls_from_part
import os
from musgconv.data.dataset import BuiltinDataset, musgconvDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from musgconv.models.core import positional_encoding
from musgconv.data.datasets.mcma import preprocess_na_to_monophonic, get_mcma_truth_edges, get_edges_mask
from musgconv.data.vocsep import GraphVoiceSeparationDataset

class Bach370ChoralesDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/craigsapp/bach-370-chorales"
        super(Bach370ChoralesDataset, self).__init__(
            name="Bach370ChoralesDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        root = os.path.join(self.raw_path, "kern")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".krn")]
        self.collections = ["chorales"] * len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


# TODO : to complete
class Bach370ChoralesPianorollVoiceSeparationDataset(musgconvDataset):
    r"""The Bach 370 Chorales Graph Voice Separation Dataset.

    Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
    (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
    Ordered by Breitkopf & Härtel numbers.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        time_unit="beat",
        time_div=12,
        musical_beat=True,
        nprocs=4,
    ):
        self.dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
        self.dataset_base.process()
        if verbose:
            print("Loaded 370 Bach Chorales Dataset Successfully, now processing...")
        self.pianorolls_dicts = list()
        self.n_jobs = nprocs
        self.time_unit = time_unit
        self.time_div = time_div
        self.musical_beat = musical_beat
        super(Bach370ChoralesPianorollVoiceSeparationDataset, self).__init__(
            name="Bach370ChoralesPianorollVoiceSeparationDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        # sys.setrecursionlimit(5000)
        def gfunc(fn, path):
            if fn.endswith(".krn"):
                return pianorolls_from_part(
                    os.path.join(path, fn),
                    self.time_unit,
                    self.time_div,
                    self.musical_beat,
                )

        path = os.path.join(self.raw_dir, "Bach370ChoralesDataset", "kern")
        self.pianorolls_dicts = Parallel(self.n_jobs)(
            delayed(gfunc)(fn, path) for fn in tqdm(os.listdir(path))
        )

        # for fn in tqdm(os.listdir(path)):
        #     out = pianorolls_from_part(os.path.join(path, fn), self.time_unit, self.time_div, self.musical_beat)
        #     self.pianorolls_dicts.append(out)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False

    # TODO save as numpy arrays for faster loading.
    def save(self):
        """save the pianorolls dicts"""
        if not os.path.exists(os.path.join(self.save_path)):
            os.mkdir(os.path.join(self.save_path))
        for i, prdicts in tqdm(enumerate(self.pianorolls_dicts)):
            file_path = os.path.join(
                self.save_path, os.path.splitext(prdicts["path"])[0] + ".pkl"
            )
            pickle.dump(prdicts, open(file_path, "wb"))

    def load(self):
        for fn in os.listdir(self.save_path):
            path_prdict = os.path.join(self.save_path, fn)
            prdict = pickle.load(open(path_prdict, "rb"))
            self.pianorolls_dicts.append(prdict)

    # Return opnly dense matrices to avoid conflicts with torch dataloader.
    def __getitem__(self, idx):
        return [[self.pianorolls_dicts[i]] for i in idx]

    def __len__(self):
        return len(self.pianorolls_dicts)

    @property
    def save_name(self):
        return self.name

    # @property
    # def features(self):
    #     if self.graphs[0].node_features:
    #         return self.graphs[0].node_features
    #     else:
    #         return list(range(self.graphs[0].x.shape[-1]))


class Bach370ChoralesGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    r"""The Bach 370 Chorales Graph Voice Separation Dataset.

    Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
    (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
    Ordered by Breitkopf & Härtel numbers.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=1, include_measures=False):
        dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
        super(Bach370ChoralesGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            is_pyg=False,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures
        )

class Bach370ChoralesPGVoiceSeparationDataset(GraphVoiceSeparationDataset):
    r"""The Bach 370 Chorales Graph Voice Separation Dataset for pytorch geometric.

    Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
    (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
    Ordered by Breitkopf & Härtel numbers.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
        Default: ~/.musgconv/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2):
        dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
        super().__init__(
            dataset_base=dataset_base,
            is_pyg=True,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist)



# class Bach370ChoralesPGVoiceSeparationDataset(musgconvDataset):
#     r"""The Bach 370 Chorales Graph Voice Separation Dataset for pytorch geometric.

#     Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
#     (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
#     Ordered by Breitkopf & Härtel numbers.

#     Parameters
#     -----------
#     raw_dir : str
#         Raw file directory to download/contains the input data directory.
#         Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
#         Default: ~/.musgconv/
#     force_reload : bool
#         Whether to reload the dataset. Default: False
#     verbose : bool
#         Whether to print out progress information. Default: True.
#     """

#     def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_max_dist=8):
#         self.dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
#         if verbose:
#             print("Loaded 370 Bach Chorales Dataset Successfully, now processing...")
#         self.graphs = list()
#         self.n_jobs = nprocs
#         self.pot_edges_max_dist = pot_edges_max_dist
#         print("pot_edges_max_dist", self.pot_edges_max_dist)
#         super(Bach370ChoralesPGVoiceSeparationDataset, self).__init__(
#             name="Bach370ChoralesPGVoiceSeparationDataset",
#             raw_dir=raw_dir,
#             force_reload=force_reload,
#             verbose=verbose,
#         )

#     def process(self):
#         def gfunc(score_fn):
#             if score_fn.endswith(".krn"):
#                 score = partitura.load_kern(score_fn)
#                 note_array = score.note_array(
#                     include_time_signature=True,
#                     include_grace_notes=True,
#                     include_staff=True,
#                 )
#                 # preprocess to remove extra voices and chords
#                 note_array = preprocess_na_to_monophonic(note_array, score_fn)
#                 # build the HeteroGraph
#                 nodes, edges, pot_edges = hetero_graph_from_note_array(note_array, pot_edge_dist=2)
#                 note_features = select_features(note_array, "voice")
#                 hg = HeteroScoreGraph(
#                     note_features,
#                     edges,
#                     name=os.path.splitext(os.path.basename(score_fn))[0],
#                     labels=None,
#                     note_array=note_array,
#                 )
#                 # Adding positional encoding to the graph features.
#                 pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
#                 hg.x = torch.cat((hg.x, pos_enc), dim=1)
#                 # Compute the truth edges
#                 truth_edges = get_mcma_truth_edges(note_array)
#                 setattr(hg, "truth_edges", truth_edges)
#                 # Compute the potential edges to use for prediction.
#                 # pot_edges = get_mcma_potential_edges(hg, max_dist=self.pot_edges_max_dist)
#                 setattr(hg, "pot_edges", torch.tensor(pot_edges))
#                 # compute the truth edges mask over potential edges
#                 truth_edges_mask, dropped_truth_edges = get_edges_mask(truth_edges, pot_edges, check_strict_subset=True)
#                 setattr(hg, "truth_edges_mask", truth_edges_mask)
#                 setattr(hg, "dropped_truth_edges", dropped_truth_edges)
#                 setattr(hg, "collection", "chorales")
#                 pg_graph = score_graph_to_pyg(hg)
#                 return pg_graph

#         path = os.path.join(self.raw_dir, "Bach370ChoralesDataset", "kern")
#         self.graphs = Parallel(self.n_jobs)(
#             delayed(gfunc)(os.path.join(path, fn)) for fn in tqdm(os.listdir(path))
#         )

#     def has_cache(self):
#         if os.path.exists(self.save_path):
#             return True

#         return False

#     def load(self):
#         for fn in os.listdir(self.save_path):
#             path_graph = os.path.join(self.save_path, fn)
#             graph = torch.load(path_graph)
#             self.graphs.append(graph)

#     def __getitem__(self, idx):
#         return [[self.graphs[i]] for i in idx]

#     def __len__(self):
#         return len(self.graphs)

#     def save(self):
#         """save the torch graphs"""
#         if not os.path.exists(os.path.join(self.save_path)):
#             os.mkdir(os.path.join(self.save_path))
#         for i, graph in tqdm(enumerate(self.graphs)):
#             file_path = os.path.join(self.save_path, graph["name"] + ".pt")
#             torch.save(graph, file_path)

#     @property
#     def metadata(self):
#         return self.graphs[0].metadata()

#     @property
#     def features(self):
#         return self.graphs[0]["note"].x.shape[-1]

#     def num_dropped_truth_edges(self):
#         return sum([len(graph["dropped_truth_edges"]) for graph in self.graphs])
