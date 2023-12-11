from musgconv.utils.chord_representations import chord_to_intervalVector, time_divided_tsv_to_note_array, time_divided_tsv_to_part
from .graph import *
from .hgraph import *
from .globals import *

node_types = ["note"]
edge_types = [
    ("note", "onset", "note"),
    ("note", "consecutive", "note"),
    ("note", "during", "note"),
    ("note", "rests", "note"),
    ("note", "consecutive_rev", "note"),
    ("note", "during_rev", "note"),
    ("note", "rests_rev", "note"),
]
METADATA = (node_types, edge_types)