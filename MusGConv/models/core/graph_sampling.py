import numpy as np


class GraphSampler:
    """
    This class is just to showcase how you can write the graph sampler in pure python.
    The simplest and most basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.
    """
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def par_sample(self, stage, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        pass

    def par_sample(self, stage, **kwargs):
        pass

    def _helper_extract_subgraph(self, node_ids):
        """
        ONLY used for serial Python sampler (NOT for the parallel cython sampler).
        Return adj of node-induced subgraph and other corresponding data struct.
        Inputs:
            node_ids        1D np array, each element is the ID in the original
                            training graph.
        Outputs:
            indptr          np array, indptr of the subg adj CSR
            indices         np array, indices of the subg adj CSR
            data            np array, data of the subg adj CSR. Since we have aggregator
                            normalization, we can simply set all data values to be 1
            subg_nodes      np array, i-th element stores the node ID of the original graph
                            for the i-th node in the subgraph. Used to index the full feats
                            and label matrices.
            subg_edge_index np array, i-th element stores the edge ID of the original graph
                            for the i-th edge in the subgraph. Used to index the full array
                            of aggregation normalization.
        """
        node_ids = np.unique(node_ids)
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(node_ids.size + 1)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids
        for nid in node_ids:
            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        return indptr, indices, data, subg_nodes, subg_edge_index

