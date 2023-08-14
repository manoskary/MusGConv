from musgconv.data import musgconvDataset, BuiltinDataset
from musgconv.utils.graph import *
import pandas as pd

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

BASIS_FN = [
	'onset_feature.score_position', 'duration_feature.duration', 'fermata_feature.fermata',
	'grace_feature.n_grace', 'grace_feature.grace_pos', 'onset_feature.onset',
	'polynomial_pitch_feature.pitch', 'grace_feature.grace_note',
	'relative_score_position_feature.score_position', 'slur_feature.slur_incr',
	'slur_feature.slur_decr', 'time_signature_feature.time_signature_num_1',
	'time_signature_feature.time_signature_num_2', 'time_signature_feature.time_signature_num_3',
	'time_signature_feature.time_signature_num_4', 'time_signature_feature.time_signature_num_5',
	'time_signature_feature.time_signature_num_6', 'time_signature_feature.time_signature_num_7',
	'time_signature_feature.time_signature_num_8', 'time_signature_feature.time_signature_num_9',
	'time_signature_feature.time_signature_num_10', 'time_signature_feature.time_signature_num_11',
	'time_signature_feature.time_signature_num_12', 'time_signature_feature.time_signature_num_other',
	'time_signature_feature.time_signature_den_1', 'time_signature_feature.time_signature_den_2',
	'time_signature_feature.time_signature_den_4', 'time_signature_feature.time_signature_den_8',
	'time_signature_feature.time_signature_den_16', 'time_signature_feature.time_signature_den_other',
	'vertical_neighbor_feature.n_total', 'vertical_neighbor_feature.n_above', 'vertical_neighbor_feature.n_below',
	'vertical_neighbor_feature.highest_pitch', 'vertical_neighbor_feature.lowest_pitch',
	'vertical_neighbor_feature.pitch_range'
	]

NOTE_FEATURES = ["int_vec1", "int_vec2", "int_vec3", "int_vec4", "int_vec5", "int_vec6"] + \
    ["interval"+str(i) for i in range(13)] + list(CHORDS.keys()) + \
    ["is_maj_triad", "is_pmaj_triad", "is_min_triad", 'ped_note',
     'hv_7', "hv_5", "hv_3", "hv_1", "chord_has_2m", "chord_has_2M"]

CAD_FEATURES = [
	'perfect_triad', 'perfect_major_triad','is_sus4', 'in_perfect_triad_or_sus4',
	'highest_is_3', 'highest_is_1', 'bass_compatible_with_I', 'bass_compatible_with_I_scale',
	'one_comes_from_7', 'one_comes_from_1', 'one_comes_from_2', 'three_comes_from_4',
	'five_comes_from_5', 'strong_beat', 'sustained_note', 'rest_highest',
	'rest_lowest', 'rest_middle', 'voice_ends', 'v7',
	'v7-3', 'has_7', 'has_9', 'bass_voice',
	'bass_moves_chromatic', 'bass_moves_octave', 'bass_compatible_v-i', 'bass_compatible_i-v',
	'bass_moves_2M']


class TonnetzCad(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/manoskary/tonnetzcad"
        super(TonnetzCad, self).__init__(
            name="TonnetzCad",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        pass

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class WTC1GraphCadenceDataset(musgconvDataset):
	def __init__(self, cad_type="PAC", raw_dir=None, verbose=True):
		self.features = BASIS_FN + NOTE_FEATURES + CAD_FEATURES
		self.cad_type = cad_type
		self.dataset_base = TonnetzCad(raw_dir=raw_dir)
		if verbose:
			print("Loaded Dependency Repository for Cadences Successfully, now processing...")
		super(WTC1GraphCadenceDataset, self).__init__(name="WTC1GraphCadenceDataset")


	def process(self):
		self.graphs = []
		self.cad_path = os.path.join(self.raw_dir, "TonnetzCad", "node_classification", "cad-feature-wtc")
		for fn in os.listdir(self.cad_path):
			print(fn)
			notes = os.path.join(self.cad_path, fn, "nodes.csv")
			edges_data = os.path.join(self.cad_path, fn, "edges.csv")

			a = notes[self.features].to_numpy()
			note_node_features = torch.from_numpy(a)
			note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
			edges_src = torch.from_numpy(edges_data['src'].to_numpy())
			edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

			src = torch.cat((edges_src, edges_dst))
			dst = torch.cat((edges_dst, edges_src))
			edges_src = src
			edges_dst = dst
			edges = (edges_src, edges_dst)

			self.graphs.append(ScoreGraph(note_node_features, edges, labels=note_node_labels, name=fn))



	def __getitem__(self, idx):
		return [[self.graphs[i].x, self.graphs[i].edge_index, self.graphs[i].y] for i in idx]

	def __len__(self):
		return len(self.graphs)

	def save(self):
		# save graphs and labels
		for graph in self.graphs:
			graph.save(save_dir=os.path.join(self.save_dir, self.name))

	def load(self):
		# load processed data from directory `self.save_path`
		self.graphs = list()
		for fn in os.listdir(self.save_path):
			path = os.path.join(self.save_path, fn)
			graph = load_score_graph(path, fn)
			self.graphs.append(graph)

	def has_cache(self):
		# check whether there are processed data in `self.save_path`
		graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		return os.path.exists(graph_path) and os.path.exists(info_path)

	@property
	def save_name(self):
		return self.name

	@property
	def features(self):
		if self.graphs[0].node_features:
			return self.graphs[0].node_features
		else:
			return list(range(self.graphs[0].x.shape[-1]))