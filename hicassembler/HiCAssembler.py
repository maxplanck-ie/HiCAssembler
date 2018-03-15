import numpy as np
import networkx as nx
import time
import os.path
import copy
import sys

import hicexplorer.HiCMatrix as HiCMatrix
import hicexplorer.hicMergeMatrixBins
import hicexplorer.hicFindTADs as hicFindTADs
from functools import wraps
from hicassembler.Scaffolds import Scaffolds

import logging
log = logging.getLogger("HiCAssembler")
log.setLevel(logging.DEBUG)

POWER_LAW_DECAY = 2**(-1.08)  # expected exponential decay at 2*distance

MIN_LENGTH = 400000  # minimum contig or PE_scaffold length to consider
ZSCORE_THRESHOLD = -1  # zscore threshold to declare a boundary a misassembly
MIN_MAD = -0.5  # minimum zscore row contacts to filter low scoring bins
MAX_MAD = 50  # maximum zscore row contacts


def timeit(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        log.info("{} took {}".format(fn.__name__, elapsed_time))
        return ret

    return with_profiling


class HiCAssembler:
    def __init__(self, hic_file_name, fasta_file, out_folder,
                 min_mad=MIN_MAD, max_mad=MAX_MAD, split_misassemblies=True,
                 split_positions_file=None,
                 min_scaffold_length=MIN_LENGTH, matrix_bin_size=25000, use_log=False,
                 num_processors=5, misassembly_zscore_threshold=ZSCORE_THRESHOLD,
                 num_iterations=2, scaffolds_to_ignore=None):
        """
        Prepares a hic matrix for assembly.
        It is expected that initial contigs or scaffolds contain bins
        of restriction fragment size.

        Parameters
        ----------
        hic_file_name : hic file name or a HiCMatrix object
        min_mad : minimum MAD score value per bin
        max_mad : maximum MAD score value per bin
        Returns
        -------

        """

        # The list is modified with each iteration replacing its members
        # by lists. After two iterations a scaffold
        # list could look like: [[0],[1,2,3]]
        # which means that there are two scaffolds
        # one of those is composed of the contigs 1, 2 and 3

        # replace the diagonal from the matrix by zeros
        # hic.diagflat(0)
        self.fasta_file = fasta_file
        self.out_folder = out_folder
        self.min_mad = min_mad
        self.max_mad = max_mad
        self.min_scaffold_length = min_scaffold_length
        self.num_processors = num_processors
        self.misassembly_threshold = misassembly_zscore_threshold
        self.merged_paths = None
        self.num_iterations = num_iterations
        self.iteration = 0

        if not isinstance(hic_file_name, str):
            # assume that the hic given is already a HiCMatrix object
            # this is normally used for testing
            self.hic = hic_file_name
        else:
            log.info("Loading Hi-C matrix ... ")
            # check if a lower resolution matrix is available
            self.load_hic_matrix(hic_file_name, split_misassemblies, split_positions_file, matrix_bin_size)

        if use_log:
            self.hic.matrix.data = np.log1p(self.hic.matrix.data)

        # build scaffolds graph. Bins on the same contig are
        # put together into a path (a type of graph with max degree = 2)
        self.scaffolds_graph = Scaffolds(copy.deepcopy(self.hic), self.out_folder)

        if scaffolds_to_ignore is not None:
            for scaffold in scaffolds_to_ignore:
                log.info("Removing scaffold {} from assembly".format(scaffold))
                if scaffold in self.scaffolds_graph.scaffold.node:
                    self.scaffolds_graph._remove_bin_path(self.scaffolds_graph.scaffold.node[scaffold]['path'],
                                                          split_scaffolds=True)
                else:
                    log.warn("Scaffold {} is not part of the assembly".format(scaffold))
        self.plot_matrix(self.out_folder + "/before_assembly.pdf",
                         title="After split mis-assemblies assembly", add_vlines=True)

        mat_size = self.hic.matrix.shape[:]
        # remove contigs that are too small
        self.scaffolds_graph.remove_small_paths(self.min_scaffold_length, split_scaffolds=True)
        assert mat_size == self.scaffolds_graph.hic.matrix.shape

        self.N50 = []

    def load_hic_matrix(self, hic_file_name, split_misassemblies, split_positions_file, matrix_bin_size):
        """
        Checks if a already processed matrix is present and loads it. If not
        the high resolution matrix is loaded, the misasemblies are
        split and the lower resolution matrix is saved.

        Parameters
        ----------
        hic_file_name name of a hic file or a HiCMatrix object
        split_misassemblies bool If true, the TAD calling algorithm is used to identify misassemblies
        split_positions_file file containing manual split positions in bed format
        matrix_bin_size bin size of matrix
        Returns
        -------

        """

        merged_bins_matrix_file = self.out_folder + "/hic_merged_bins_matrix.h5"
        if os.path.isfile(merged_bins_matrix_file):
            log.info("Found reduced matrix file {}".format(merged_bins_matrix_file))
            self.hic = HiCMatrix.hiCMatrix(merged_bins_matrix_file)
        else:
            self.hic = HiCMatrix.hiCMatrix(hic_file_name)

            if split_misassemblies:
                # try to find contigs that probably should be separated
                self.split_misassemblies(hic_file_name, split_positions_file)

            log.info("Merging bins of file to reduce resolution")
            binsize = self.hic.getBinSize()
            if binsize < matrix_bin_size:
                # make an smaller matrix having bins of around 25.000 bp
                num_bins = matrix_bin_size / binsize

                log.info("Reducing matrix size to {:,} bp (number of bins merged: {})".format(binsize, num_bins))
                self.hic = HiCAssembler.merge_bins(self.hic, num_bins)

            self.hic.save(merged_bins_matrix_file)
            self.hic = HiCMatrix.hiCMatrix(merged_bins_matrix_file)

    def assemble_contigs(self):
        """

        Returns
        -------

        """
        log.debug("Size of matrix is {}".format(self.scaffolds_graph.hic.matrix.shape[0]))
        for iteration in range(self.num_iterations):
            self.iteration = iteration
            self.scaffolds_graph.iteration = iteration
            n50 = self.scaffolds_graph.compute_N50()
            self.scaffolds_graph.get_paths_stats()

            log.debug("iteration: {}\tN50: {:,}".format(iteration, n50))
            self.N50.append(n50)

            # the first iteration is is more stringent
            if iteration < 3:
                target_size = int(min(2e6, self.scaffolds_graph.paths_min * (iteration + 1)))
                log.debug("Merging small bins in larger bins of size {} bp".format(target_size))
                self.scaffolds_graph.split_and_merge_contigs(num_splits=3,
                                                             target_size=target_size,
                                                             normalize_method='ice')
                stats = self.scaffolds_graph.get_stats_per_split()
                try:
                    # stats[2] contains the mean, median, max, min and len(number of samples)
                    # for bins whose start position is about the distance of two
                    # bins or in other words that are separated by one bin
                    conf_score = stats[2]['median'] * 0.9
                # if the scaffolds are all very small, the get_stats_per_split
                # many not have enough information to compute, thus a second
                # method to identify confidence score is used
                except KeyError:
                    conf_score = np.percentile(self.scaffolds_graph.matrix.data, 5)

                log.debug("Confidence score set to {}".format(conf_score))

            else:
                # self.scaffolds_graph.split_and_merge_contigs(num_splits=1, target_size=int(1e6),
                #                                              normalize_method='ice')
                self.scaffolds_graph.split_and_merge_contigs(num_splits=1, normalize_method='ice')
                conf_score = np.percentile(self.scaffolds_graph.matrix.data, 30)
                log.info("Confidence score set to: {}".format(conf_score))

            self.scaffolds_graph.join_paths_max_span_tree(conf_score, node_degree_threshold=2e3,
                                                          hub_solving_method='remove weakest')

            if iteration == 0:
                self.scaffolds_graph.remove_small_paths(self.min_scaffold_length, split_scaffolds=True)
            self.plot_matrix(self.out_folder + "/after_assembly_{}.pdf".format(iteration),
                             title="Assembly iteration {}".format(iteration), add_vlines=True)

        before_assembly_length, before_num_paths = self.scaffolds_graph.get_assembly_length()

        self.put_back_small_scaffolds()

        after_assembly_length, after_num_paths = self.scaffolds_graph.get_assembly_length()

        diff = after_assembly_length - before_assembly_length
        log.info('{:,} bp ({:.2%}) were added back to the assembly'.
                 format(diff, float(diff) / self.scaffolds_graph.total_length))

        log.info('Total assembly length: {:,} bp ({:.2%})'.
                 format(after_assembly_length, float(after_assembly_length) / self.scaffolds_graph.total_length))

        self.plot_matrix(self.out_folder + "/after_put_scaff_back.pdf".
                         format(iteration), title="After assembly", add_vlines=True)

        hic = self.reorder_matrix(max_num_bins=int(1e6), rename_scaffolds=True)
        hic.save(self.out_folder + "/final_matrix.h5")
        print self.N50

        return self.get_contig_order()

    def make_scaffold_network(self, orig_scaff, confidence_score=None):
        """

        Parameters
        ----------
        orig_scaff
        confidence_score minimum value in the matrix

        Returns
        -------

        Examples
        --------

        >>> import tempfile
        >>> dirpath = tempfile.mkdtemp(prefix="hicassembler_test_")
        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 30, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 0, 10, 1), ('c-2', 10, 30, 1)]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> H = HiCAssembler(hic, "", dirpath, split_misassemblies=False,
        ... min_scaffold_length=20, use_log=False)
        >>> H.scaffolds_graph.split_and_merge_contigs(num_splits=1, normalize_method='none')
        >>> H.scaffolds_graph.add_edge(0, 1)
        >>> list(H.scaffolds_graph.matrix_bins.get_all_paths())
        [[0, 1, 5, 4]]
        >>> orig_scaff = Scaffolds(H.hic)
        >>> orig_scaff.split_and_merge_contigs(num_splits=1, normalize_method='none')

        In orig_scaff only the scaffold paths are represented. It contains all scaffolds
        including deleted ones in self.scaffold_graph. No connections between scaffolds
        are present
        >>> list(orig_scaff.matrix_bins.get_all_paths())
        [[0, 1], [2, 3], [4, 5]]
        >>> G = H.make_scaffold_network(orig_scaff)
        >>> list(G.edges(data=True))
        [('c-2', 'c-1', {'weight': 16.0}), ('c-2', 'c-0', {'weight': 42.0}), \
('c-1', 'c-0', {'weight': 28.0})]

        >>> import shutil
        >>> shutil.rmtree(dirpath)

        """
        nxG = nx.Graph()
        for node_id, node in orig_scaff.scaffold.node.iteritems():
            nn = node.copy()
            for attr, value in nn.iteritems():
                if isinstance(value, np.int64):
                    nn[attr] = int(value)
                elif isinstance(value, np.float64):
                    nn[attr] = float(value)
                elif isinstance(value, list):
                    nn[attr] = ", ".join([str(x) for x in value])
                elif isinstance(value, np.string_):
                    nn[attr] = str(value)

            if node_id in self.scaffolds_graph.scaffold.node:
                nn['is_backbone'] = 1
            nxG.add_node(node_id, **nn)

        matrix = orig_scaff.matrix.tocoo()
        matrix.setdiag(0)

        max_weight = float(orig_scaff.matrix.max() * 1.5)
        for u, v, weight in zip(matrix.row, matrix.col, matrix.data):
            if u == v:
                continue
            if weight < confidence_score:
                continue
            scaff_u = orig_scaff.pg_base.node[u]['name']
            scaff_v = orig_scaff.pg_base.node[v]['name']
            nxG.add_edge(scaff_u, scaff_v, weight=float(weight))
            if scaff_u in self.scaffolds_graph.scaffold.node and \
               scaff_v in self.scaffolds_graph.scaffold.node and \
               scaff_u in self.scaffolds_graph.scaffold.adj[scaff_v]:
                # u and v are directly joined
                nxG.add_edge(scaff_u, scaff_v, weight=float(max_weight))

        # add all contacts between assembled nodes that may not have been
        # present in the graph
        for path in self.scaffolds_graph.scaffold.get_all_paths():
            for scaff_u, scaff_v in zip(path[:-1], path[1:]):
                nxG.add_edge(scaff_u, scaff_v, weight=float(max_weight))

        return nxG

    @staticmethod
    def _remove_weakest(G, exclude=[]):
        """
        Based on the maximum spanning tree graph hubs are resolved by removing the
        weakest links until only two edges are left

        For a maximum spanning tree like this:

        o---o---o---o---o---o--o
                          \
                           --o--o

        The algorithm works as follows:

        1. Sort the node degree in decreasing order
        2. For each node with degree > 0 leave only the two edges with the highest
           weight


        Parameters
        ----------
        G : maximum spanning tree networkx graph
        exclude : list of nodes to exclude from removing links.

        Returns
        -------
        G
        """
        node_degree_mst = dict(G.degree(G.node.keys()))
        for node, degree in sorted(node_degree_mst.iteritems(), key=lambda (k, v): v, reverse=True):
            if degree > 2 and node not in exclude:
                adj = sorted(G.adj[node].iteritems(), key=lambda (k, v): v['weight'])
                # remove the weakest edges but only if either of the nodes is not a hub
                for adj_node, attr in adj[:-2]:
                    log.debug("Removing weak edge {}-{} weight: {}".format(node, adj_node, attr['weight']))
                    G.remove_edge(node, adj_node)
            if degree <= 2:
                break
        return G

    @staticmethod
    def _find_backbone_node(graph):
        """
        Given a networkx graph, identifies the node (or nodes that are backgbone) that is labeled as backbone.
        This function is called as part of put_back_small_scaffolds

        Parameters
        ----------
        graph : networkX graph.

        Returns
        -------
        set of backbone nodes



        """
        backbones = set()
        for node_id, attr in graph.node.iteritems():
            if 'is_backbone' in attr:
                backbones.add(node_id)

        return backbones

    @staticmethod
    def _get_subgraph_containing_node(graph, target_node):
        """
        Returns the subgraph of `graph` containing the given node
        Parameters
        ----------
        graph : NetworkX graph
        target_node : node id

        Returns
        -------
        Networkx graph or None if the node is not in the graph
        """

        for sub_graph in nx.connected_component_subgraphs(graph):
            if target_node in sub_graph:
                return sub_graph
        return None

    @staticmethod
    def _get_paths_from_backbone(graph, backbone_node):
        """
        Returns all paths that contain the backbone

        The graph used should not contain nodes with degree > 2 except for the
        backbone node: eg.

         o--*--o--o
            |
            o

        but not:

            o--o--*--o
               |
               o

        Parameters
        ----------
        graph : Networkx graph
        backbone_node: node id

        Returns
        -------
        path list, where each path has the backbone as the first element

        Examples
        --------

        >>> G = nx.Graph()
        >>> G.add_edge('backbone', 1, weight=10)
        >>> G.add_edge(1, 2, weight=5)
        >>> G.add_edge(2, 3, weight=6)
        >>> G.add_edge('backbone', 4, weight=5)
        >>> G.add_edge(4, 5, weight=10)
        >>> HiCAssembler._get_paths_from_backbone(G, 'backbone')
        [['backbone', 1, 2, 3], ['backbone', 4, 5]]
        """

        # get backbone_id neighbors
        path_list = []
        seen = set([backbone_node])
        for adj, weight in sorted(graph.adj[backbone_node].iteritems(), key=lambda (k, v): v['weight'])[::-1]:
            path = [backbone_node]
            while True:
                path.append(adj)
                seen.add(adj)
                adj_list = [x for x in graph.adj[adj].keys() if x not in seen]
                if len(adj_list) == 0:
                    break
                adj = adj_list[0]
            path_list.append(path)

        return path_list

    def put_back_small_scaffolds(self, normalize_method='ice'):
        """
        Identifies scaffolds that were removed from the Hi-C assembly and
        tries to find their correct location.
        Returns
        -------


        Examples
        --------
        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 30, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 0, 10, 1), ('c-2', 10, 30, 1)]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> import tempfile
        >>> dirpath = tempfile.mkdtemp()
        >>> H = HiCAssembler(hic, "", dirpath, split_misassemblies=False, min_scaffold_length=20, use_log=False)
        >>> H.scaffolds_graph.split_and_merge_contigs(num_splits=1, normalize_method='none')
        >>> H.scaffolds_graph.add_edge(0, 1)
        >>> list(H.scaffolds_graph.matrix_bins.get_all_paths())
        [[0, 1, 5, 4]]

        >>> H.put_back_small_scaffolds()
        >>> list(H.scaffolds_graph.matrix_bins.get_all_paths())
        [[0, 1, 2, 3, 5, 4]]

        >>> import shutil
        >>> shutil.rmtree(dirpath)

        # larger test
        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix

        >>> cut_intervals = [('c-0', 0, 20, 1), ('c-0', 20, 40, 2),
        ... ('c-1', 10, 20, 1), ('c-1', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 10, 20, 1),
        ... ('c-3', 0, 10, 1), ('c-3', 10, 20, 1),
        ... ('c-4', 0, 20, 1), ('c-4', 20, 40, 1)]
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix(np.array(
        ... [[50,  19,  9,  8,  5,  3,  2,  1,  0,  0],
        ...  [ 0, 50,  20,  9,  8,  5,  3,  2,  1,  0],
        ...  [ 0,  0, 50,  19,  9,  8,  5,  3,  2,  1],
        ...  [ 0,  0,  0, 50,  19,  9,  8,  5,  3,  2],
        ...  [ 0,  0,  0,  0, 50,  19,  9,  8,  5,  3],
        ...  [ 0,  0,  0,  0,  0, 50,  19,  9,  8,  5],
        ...  [ 0,  0,  0,  0,  0,  0, 50,  19,  9,  8],
        ...  [ 0,  0,  0,  0,  0,  0,  0, 50,  19,  9],
        ...  [ 0,  0,  0,  0,  0,  0,  0,  0, 50,  19],
        ...  [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 50]]))

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)

        # scramble matrix
        # the scrambled order is [c-3 (inv), c-1(inv), c-0, c-4, c-2]
        >>> scrambled_bins = [7,6, 3,2, 0,1, 8,9, 5,4]
        >>> matrix = hic.matrix[scrambled_bins, :][:, scrambled_bins]

        # the intervals are shuffled but not the direction
        >>> scrambled_intervals = [6,7, 2,3, 0,1, 8,9, 5,4]
        >>> cut_intervals = [cut_intervals[x] for x in scrambled_intervals]
        >>> hic.setMatrix(matrix, cut_intervals)
        >>> hic.matrix.todense()
        matrix([[100,  19,   5,   3,   1,   2,  19,   9,   9,   8],
                [ 19, 100,   8,   5,   2,   3,   9,   8,  19,   9],
                [  5,   8, 100,  19,   8,   9,   3,   2,   9,  19],
                [  3,   5,  19, 100,   9,  20,   2,   1,   8,   9],
                [  1,   2,   8,   9, 100,  19,   0,   0,   3,   5],
                [  2,   3,   9,  20,  19, 100,   1,   0,   5,   8],
                [ 19,   9,   3,   2,   0,   1, 100,  19,   8,   5],
                [  9,   8,   2,   1,   0,   0,  19, 100,   5,   3],
                [  9,  19,   9,   8,   3,   5,   8,   5, 100,  19],
                [  8,   9,  19,   9,   5,   8,   5,   3,  19, 100]])

        >>> dirpath = tempfile.mkdtemp()
        >>> H = HiCAssembler(hic, "", dirpath, split_misassemblies=False, min_scaffold_length=20, use_log=False)
        >>> H.scaffolds_graph.split_and_merge_contigs(num_splits=1, normalize_method='none')

        the shorter scaffolds are removed: c-1, c-2, and c-4.
        add edge between remaining scaffolds
        >>> H.scaffolds_graph.add_edge(0, 1)


        After adding the edge the network looks like
             c-0  c-4
              o---o

              c-1, c-2, c-3 removed

        >>> list(H.scaffolds_graph.scaffold.get_all_paths())
        [['c-0', 'c-4']]
        >>> H.put_back_small_scaffolds(normalize_method='none')

        The intermediate maximum spanning tree that is created
        looks like:

                 c-0  c-4
                   o===o
                  /    \
             c-1 o     o c-3
                        \
                        o c-2
        (==) denotes `backbone` edge, i.e, a edge that was established using the larger scaffolds

        In the algorithm, the backbone edges are removed as well as all backbone nodes not attached to a removed
        scaffold. After this step, the previous network now looks like:

                 c-0   c-4
                  o     o
                 /      \
            c-1 o       o c-3
                         \
                         o c-2

        Next, each branch is considered independently (e.g. [c-0, c-1]). The orientation of the scaffolds is
        determined pairwise using the find_best_permutation method on the scaffold path. E.g. for c-0 the matrix
        nodes path is [4, 5]. Once the orientation is known, the edge of the scaffold backbone is deleted to
        allow the insertion of the branch at that position.

        >>> list(H.scaffolds_graph.scaffold.get_all_paths())
        [['c-0', 'c-1', 'c-2', 'c-3', 'c-4']]

        # The resulting matrix should be ordered
        >>> hic = H.reorder_matrix()
        >>> hic.matrix.todense()
        matrix([[100,  19,   9,   8,   5,   3,   2,   1,   0,   0],
                [ 19, 100,  20,   9,   8,   5,   3,   2,   1,   0],
                [  9,  20, 100,  19,   9,   8,   5,   3,   2,   1],
                [  8,   9,  19, 100,  19,   9,   8,   5,   3,   2],
                [  5,   8,   9,  19, 100,  19,   9,   8,   5,   3],
                [  3,   5,   8,   9,  19, 100,  19,   9,   8,   5],
                [  2,   3,   5,   8,   9,  19, 100,  19,   9,   8],
                [  1,   2,   3,   5,   8,   9,  19, 100,  19,   9],
                [  0,   1,   2,   3,   5,   8,   9,  19, 100,  19],
                [  0,   0,   1,   2,   3,   5,   8,   9,  19, 100]])

        >>> shutil.rmtree(dirpath)
       """
        log.info("Total assembly length before adding scaffolds back: {:,}".
                 format(self.scaffolds_graph.get_assembly_length()[0]))

        # create orig_scaff once using a min_scaffold length as size target to
        # compute confidence scores
        orig_scaff = Scaffolds(self.hic)
        orig_scaff.split_and_merge_contigs(num_splits=1, target_size=self.min_scaffold_length, normalize_method=normalize_method)
        orig_stats = orig_scaff.get_stats_per_split()
        conf_score = orig_stats[1]['median']

        # re make orig_scaff a second time without splitting the scaffolds
        # as this is the structure needed for rest of the program
        orig_scaff = Scaffolds(self.hic)
        orig_scaff.split_and_merge_contigs(num_splits=1, normalize_method=normalize_method)
        # reset pb_base
        self.scaffolds_graph.pg_base = copy.deepcopy(self.scaffolds_graph.matrix_bins)
        nxG = self.make_scaffold_network(orig_scaff, confidence_score=conf_score)

        nxG = nx.maximum_spanning_tree(nxG, weight='weight')
        nx.write_graphml(nxG, self.out_folder + "/mst_for_small_Scaff_integration.graphml".format())

        # 1. Identify branches

        # delete backbone nodes that are not adjacent to a removed scaffold (removed scaffolds are those
        # small contig/scaffolds removed at the beginning that are stored in the self.scaffolds_graph.remove_scaffolds.
        # Basically, all so called backbone nodes that are not connected to the scaffolds that we want to put back
        # are deleted.
        for node_id in self.scaffolds_graph.scaffold.node.keys():
            # check that the backbone node is not adjacent to a removed node.
            if len(set(nxG.adj[node_id].keys()).intersection(self.scaffolds_graph.removed_scaffolds.node.keys())) == 0:
                nxG.remove_node(node_id)

        # remove backbone edges. That is, if there is some edge between two backbone nodes, this is removed (see
        # example on the docstring).
        for u, v in list(nxG.edges()):
            if 'is_backbone' in nxG.node[u] and 'is_backbone' in nxG.node[v]:
                nxG.remove_edge(u, v)

        nx.write_graphml(nxG, "{}/backbone_put_back_scaffolds.graphml".format(self.out_folder))
        # now each connected component should only have a backbone node
        # and all the connected scaffolds that belong to that node.
        for branch in list(nx.connected_component_subgraphs(nxG)):
            branch_len = sum([branch.node[x]['length'] for x in branch])
            branch_nodes = [x for x in branch]
            log.debug("Checking branch for insertion in assembly.\nLength:{}\nScaffolds:{}".
                      format(branch_len, branch_nodes))
            if len(branch) > 20:
                log.info("Skipping the insertion of a branch that is too long. "
                         "The length of the branch is: {} (threshold is 20)".format(len(branch)))
                continue
            # after removing the hubs the branch may contain several connected components. Only the component
            # that contains a backbone node is used.
            backbone_list = HiCAssembler._find_backbone_node(branch)

            if len(backbone_list) == 0:
                if len(branch_nodes) == 1:
                    continue
                # this is a branch without a backbone and is inserted as a separated
                # hic-scaffold
                branch = HiCAssembler._remove_weakest(branch)
                path = Scaffolds._return_paths_from_graph(branch)[0]

                for scaff_name in path:
                    # restore all scaffolds except for path[0] which is the backbone node (and was not removed)
                    self.scaffolds_graph.restore_scaffold(scaff_name)

                # get the matrix bin paths for each scaffold
                bins_path = [self.scaffolds_graph.scaffold.node[x]['path'] for x in path]

                # a. find the best orientation of the scaffold paths with respect to each other
                #     the best path contains the bins_path in the best computed orientations
                # best_path[0] is the backbone scaffold path
                best_path = Scaffolds.find_best_permutation(orig_scaff.hic.matrix, bins_path,
                                                            only_expand_but_not_permute=True)
                # b. add edges in the path
                for path_u, path_v in zip(best_path[:-1], best_path[1:]):
                    self.scaffolds_graph.add_edge_matrix_bins(path_u[-1], path_v[0])

                log.debug("No backbone found for branch with nodes: {}".format(branch.node.keys()))
                log.info("Scaffolds without a backbone node were added: {}".format(path))
                continue

            # each branch should contain at most two backbone node
            if len(backbone_list) > 2:
                log.info("Branch contains more than one backbone. Backbones in branch: {}".format(backbone_list))
                log.info("Skipping this branch of length: {}".format(len(branch)))
                continue

            # if the branch contains two backbone nodes and is a path
            # that means that the path is connecting two different
            # hic-scaffolds. The solution is to break the path by the
            # weakest link instead of letting the path to join the
            # two different hic-scaffolds
            elif len(backbone_list) == 2:
                # check if the branch forms a path with the backbones at the two ends
                path = HiCAssembler._get_paths_from_backbone(branch, list(backbone_list)[0])
                if len(path) > 1:
                    continue

                path = path[0]
                if backbone_list.intersection([path[0], path[-1]]) != backbone_list:
                    # the two backbones are not on the sides of the path.
                    continue

                # check if the backbone nodes are adjacent in
                # the hic-scaffolds. That means the removed path
                # should be inserted between them.
                if path[0] in self.scaffolds_graph.scaffold.adj[path[-1]]:
                    # remove one of the back bones of the graph and continue
                    log.debug("Removing one backbone scaffold from branch with two backbones")
                    branch.remove_node(path[-1])
                    self.insert_path(path[:-1], orig_scaff)
                    continue
                else:
                    # the backbones belong to different hic-scaffolds
                    # the path is split by the weakest edge.
                    min_weight = np.Inf
                    for u, v, attr in branch.edges(data=True):
                        if attr['weight'] < min_weight:
                            min_edge = (u, v)
                            min_weight = attr['weight']

                    log.debug("Removing weak edge in path connecting two hic-scaffolds: "
                              "edge: {}, weight: {}".format(min_edge, min_weight))

                    idx_u = path.index(min_edge[0])
                    idx_v = path.index(min_edge[1])

                    if idx_u > idx_v:
                        idx_u, idx_v = idx_v, idx_u
                    assert idx_u + 1 == idx_v
                    path_a = path[:idx_v]
                    path_b = path[idx_v:]
                    if len(path_a) > 0:
                        self.insert_path(path_a, orig_scaff)

                    if len(path_b) > 0:
                        # path b must be inverted such that path[0]
                        # corresponds to the backbone node
                        self.insert_path(path_b[::-1], orig_scaff)

                    continue
            else:
                backbone_node = list(backbone_list)[0]

                # A this point a branch may look like this
                #                       o
                #                      /
                #               o--*--o--o
                #                   \
                #                    o--o--o
                #                        \
                #                         o
                #
                # where `*` is the backbone node.

                branch = HiCAssembler._remove_weakest(branch, exclude=[backbone_node])
                # after removing the weakest edges parts of the graph are no longer connected to the backbone
                # thus, the subgraph containing the backbone is selected

                branch = HiCAssembler._get_subgraph_containing_node(branch, backbone_node)
                if branch is None:
                    log.debug("Graph is emtpy")
                    continue

                for path in HiCAssembler._get_paths_from_backbone(branch, backbone_node):
                    self.insert_path(path, orig_scaff)

        log.info("Total assembly length after adding scaffolds back: {:,}".format(self.scaffolds_graph.get_assembly_length()[0]))
        return

    def insert_path(self, path, orig_scaff):
        """

        Parameters
        ----------
        path

        Returns
        -------

        """
        # in path, path[0] is always the backbone node.

        for scaff_name in path[1:]:
            # restore all scaffolds except for path[0] which is the backbone node (and was not removed)
            self.scaffolds_graph.restore_scaffold(scaff_name)

        # get the matrix bin paths for each scaffold
        bins_path = [self.scaffolds_graph.scaffold.node[x]['path'] for x in path]

        # a. find the best orientation of the scaffold paths with respect to each other
        #     the best path contains the bins_path in the best computed orientations
        # best_path[0] is the backbone scaffold path
        best_path = Scaffolds.find_best_permutation(orig_scaff.hic.matrix, bins_path,
                                                    only_expand_but_not_permute=True)

        # the backbone bin id that should be joined with the removed scaffold
        # corresponds to the the last bin_id in the first best_path, which is the backbone path
        backbone_bin = best_path[0][-1]

        # identify the neighbor bin of the backbone scaffold in adjacent scaffold (if any).
        # To insert the removed scaffolds an edge in the assembled scaffolds has to be removed. This
        # edge is the edge containing the backbone_bin.
        adjacent_backbone_bin = None
        for adj in self.scaffolds_graph.matrix_bins.adj[backbone_bin].keys():
            if self.scaffolds_graph.matrix_bins.node[adj]['name'] != self.scaffolds_graph.matrix_bins.node[backbone_bin]['name']:
                adjacent_backbone_bin = adj
                break

        if adjacent_backbone_bin is not None:
            # delete edge between backbone and adjacent scaffold
            self.scaffolds_graph.delete_edge_from_matrix_bins(backbone_bin, adjacent_backbone_bin)
            # add edge between last bin in best path and adjacent scaffold
            self.scaffolds_graph.add_edge_matrix_bins(best_path[-1][-1], adjacent_backbone_bin)

        # b. add the other edges in the path
        for path_u, path_v in zip(best_path[:-1], best_path[1:]):
            self.scaffolds_graph.add_edge_matrix_bins(path_u[-1], path_v[0])

        integrated_paths = [(x, self.scaffolds_graph.scaffold.node[x]['length']) for x in path[1:]]
        log.info("Scaffolds {} successfully integrated into the network".format(integrated_paths))

    def split_misassemblies(self, hic_file_name, split_positions_file=None):
        """
        Mis assemblies are commonly found in the data. To remove them, we use
        a simple metric to identify empty contacts.

        Parameters
        ----------
        hic_file_name : Name of the file

        Returns
        -------

        """
        log.info("Detecting misassemblies")

        tad_score_file = self.out_folder + "/misassembly_score.txt"
        zscore_matrix_file = self.out_folder + "/zscore_matrix.h5"
        # check if the computation for the misassembly score was already done
        if not os.path.isfile(tad_score_file) or not os.path.isfile(zscore_matrix_file):
            ft = hicFindTADs.HicFindTads(hic_file_name, num_processors=self.num_processors, use_zscore=False)
            # adjust window sizes to compute misassembly score (aka tad-score)
            ft.max_depth = max(800000, ft.binsize * 500)
            ft.min_depth = min(200000, ft.binsize * 200)
            ft.step = ft.binsize * 50
            log.debug("zscore window sizes set by hicassembler: ")
            log.debug("max depth:\t{}".format(ft.max_depth))
            log.debug("min depth:\t{}".format(ft.min_depth))
            log.debug("step:\t{}".format(ft.step))
            log.debug("bin size:\t{}".format(ft.binsize))

            ft.hic_ma.matrix.data = np.log1p(ft.hic_ma.matrix.data)
            ft.hic_ma.matrix = ft.hic_ma.convert_to_obs_exp_matrix(perchr=True)
            ft.hic_ma.matrix.data = np.log2(ft.hic_ma.matrix.data)
            ft.compute_spectra_matrix(perchr=True)
            ft.save_bedgraph_matrix(tad_score_file)
            ft.hic_ma.save(zscore_matrix_file)

        log.info("Using previously computed scores: {}\t{}".format(tad_score_file, zscore_matrix_file))
        # TODO here the hic_file is loaded unnecessarily. A way to remove this step would be good
        ft = hicFindTADs.HicFindTads(hic_file_name, num_processors=self.num_processors, use_zscore=False)
        ft.hic_ma = HiCMatrix.hiCMatrix(zscore_matrix_file)
        ft.load_bedgraph_matrix(tad_score_file)
        ft.find_boundaries()

        tuple_ = []
        # find the tad score and position of boundaries with significant pvalues
        for idx, pval in ft.boundaries['pvalues'].iteritems():
            tuple_.append((ft.bedgraph_matrix['chrom'][idx],
                           ft.bedgraph_matrix['chr_start'][idx],
                           ft.bedgraph_matrix['chr_end'][idx],
                           np.mean(ft.bedgraph_matrix['matrix'][idx])))

        scaffold, start, end, tad_score = zip(*tuple_)
        tad_score = np.array(tad_score)
        # compute a zscore of the tad_score to select the lowest ranking boundaries.
        zscore = (tad_score - np.mean(tad_score)) / np.std(tad_score)

        # select as misassemblies all boundaries that have a zscore lower than 1.64 (p-value 0.05)
        bin_ids = {}
        bins_to_remove = []
        log.info("Splitting scaffolds using threshold = {}".format(self.misassembly_threshold))
        for idx in np.flatnonzero(zscore < self.misassembly_threshold):
            # find the bins that overlap with the misassembly
            if scaffold[idx] not in self.hic.interval_trees:
                # the scaffold[idx] key may not be present
                # in the self.hic because of the reduction of the matrix, which removes some scaffolds
                continue
            if scaffold[idx] not in bin_ids:
                bin_ids[scaffold[idx]] = []
            to_split_intervals = sorted(self.hic.interval_trees[scaffold[idx]][start[idx]:end[idx]])
            bin_ids[scaffold[idx]].extend(sorted([interval_bin.data for interval_bin in to_split_intervals]))

        # split scaffolds based on input file from user
        if split_positions_file is not None:
            log.debug("loading positions to split from {}".format(split_positions_file))
            from hicexplorer import readBed

            bed_file_h = readBed.ReadBed(open(split_positions_file, 'r'))
            for bed in bed_file_h:
                # find the bins that overlap with the misassembly
                if bed.chromosome not in self.hic.interval_trees:
                    log.info("split position {} not found in hic matrix".format(bed))
                    continue
                if bed.chromosome not in bin_ids:
                    bin_ids[bed.chromosome] = []
                to_split_intervals = sorted(self.hic.interval_trees[bed.chromosome][bed.start:bed.end])
                if len(to_split_intervals) == 0:
                    # it could be that there is not bin nearby so the nearest bin is taken
                    log.info('split position from split list {} does not match any bin. Using nearest bin'.format(bed))
                    to_split_intervals = [sorted(self.hic.interval_trees[bed.chromosome][0:bed.end])[-1]]
                    log.info('split position used is {}.'.format(to_split_intervals[0]))

                to_split_intervals = sorted([interval_bin.data for interval_bin in to_split_intervals])
                if len(to_split_intervals) > 1:
                    # if the split contains several bins, the region should be removed from the matrix.
                    # All the bins, except the last one, are marked for deletion. The last one is marked
                    # for split.
                    bins_to_remove.extend(to_split_intervals[:-1])
                    to_split_intervals = [to_split_intervals[-1]]

                bin_ids[bed.chromosome].extend(to_split_intervals)

        # rename cut intervals
        num_removed_misassemblies = 0
        new_cut_intervals = self.hic.cut_intervals[:]
        for scaff_name in bin_ids:
            scaff_bins = self.hic.getChrBinRange(scaff_name)
            # remove splits at the start or end of chromosome as they are most likely
            # false positives
            id_list = set(sorted([x for x in bin_ids[scaff_name] if x not in [scaff_bins[0], scaff_bins[1] - 1]]))
            part_number = 1
            if len(id_list) > 0:
                log.info("Removing {} misassemblies for {} ".format(len(id_list), scaff_name))
                for matrix_bin in range(scaff_bins[0], scaff_bins[1]):
                    name, cut_start, cut_end, extra = new_cut_intervals[matrix_bin]
                    new_name = "{}/{}".format(name, part_number)
                    new_cut_intervals[matrix_bin] = (new_name, cut_start, cut_end, extra)
                    if matrix_bin in id_list:
                        part_number += 1
                        num_removed_misassemblies += 1

        self.hic.setCutIntervals(new_cut_intervals)
        log.info("{} misassemblies were removed".format(num_removed_misassemblies))

        if len(bins_to_remove) > 0:
            log.info("{} bins will be removed from the matrix because they are contained within the split regions.".
                     format(len(bins_to_remove)))
            self.hic.removeBins(bins_to_remove)

    def plot_matrix(self, filename, title='Assembly results',
                    cmap='RdYlBu_r', log1p=True, add_vlines=False, vmax=None, vmin=None):
        """
        Plots the resolved paths on a matrix

        Parameters
        ----------
        filename
        title
        cmap
        log1p
        add_vlines
        vmax
        vmin

        Returns
        -------
        None
        """

        log.debug("plotting matrix")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig = plt.figure(figsize=(10, 10))
        hic = self.reorder_matrix()

        axHeat2 = fig.add_subplot(111)
        axHeat2.set_title(title)

        chrbin_boundaries = hic.chrBinBoundaries
        ma = hic.matrix.todense()
        norm = None
        if log1p:
            ma += 1
            norm = LogNorm()

        img3 = axHeat2.imshow(ma, interpolation='nearest', vmax=vmax, vmin=vmin, cmap=cmap, norm=norm)

        img3.set_rasterized(True)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axHeat2)
        cax = divider.append_axes("right", size="2.5%", pad=0.09)
        cbar = fig.colorbar(img3, cax=cax)
        cbar.solids.set_edgecolor("face")  # to avoid white lines in the color bar in pdf plots

        ticks = [pos[0] for pos in chrbin_boundaries.values()]
        labels = chrbin_boundaries.keys()
        axHeat2.set_xticks(ticks)
        if len(labels) < 40:
            axHeat2.set_xticklabels(labels, size=3, rotation=90)
        else:
            axHeat2.set_xticklabels(labels, size=1, rotation=90)

        if add_vlines:
            # add lines to demarcate 'super scaffolds'
            vlines = [x[0] for x in hic.chromosomeBinBoundaries.values()]
            axHeat2.vlines(vlines, 1, ma.shape[0], linewidth=0.1)
            axHeat2.set_ylim(ma.shape[0], 0)
        axHeat2.get_yaxis().set_visible(False)
        log.debug("saving matrix {}".format(filename))
        plt.savefig(filename, dpi=300)
        plt.close()

    def remove_noise_from_matrix(self):
        """
        set noise level at the value found in up to 70% of the sparse matrix data.
        The noise is removed from the hic matrix.

        Returns
        -------
        None
        """
        noise_level = np.percentile(self.hic.matrix.data, 70)
        log.debug("noise level set to {}".format(noise_level))

        self.hic.matrix.data = self.hic.matrix.data - noise_level
        self.hic.matrix.data[self.hic.matrix.data < 0] = 0
        self.hic.matrix.eliminate_zeros()

    def get_contig_order(self, add_split_contig_name=False):
        """

        Parameters
        ----------
        add_split_contig_name

        Returns
        -------

        Examples
        --------
        >>> import tempfile
        >>> dirpath = tempfile.mkdtemp(prefix="hicassembler_test_")
        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 30, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2/1', 0, 10, 1), ('c-2/2', 10, 30, 1)]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> H = HiCAssembler(hic, "", dirpath, split_misassemblies=False, min_scaffold_length=0)
        >>> H.scaffolds_graph.add_edge_matrix_bins(1,3)
        >>> H.get_contig_order(add_split_contig_name=False)
        [[('c-0', 0, 30, '+'), ('c-1', 0, 20, '-')], [('c-2', 0, 10, '+')], [('c-2', 10, 30, '+')]]

        >>> H.get_contig_order(add_split_contig_name=True)
        [[('c-0', 0, 30, '+'), ('c-1', 0, 20, '-')], [('c-2/1', 0, 10, '+')], [('c-2/2', 10, 30, '+')]]
        >>> import shutil
        >>> shutil.rmtree(dirpath)

        """
        import re
        super_scaffolds = []

        for path in self.scaffolds_graph.scaffold.get_all_paths():
            scaffold = []
            for scaff_name in path:
                scaff_data = self.scaffolds_graph.scaffold.node[scaff_name]
                if add_split_contig_name is False:
                    # check if node name has an indication that it was split (by ending in '/n')
                    res = re.search("(.*?)/(\d+)$", scaff_name)
                    if res is not None:
                        scaff_name = res.group(1)

                scaffold.append((scaff_name, scaff_data['start'],
                                 scaff_data['end'], scaff_data['direction']))

            super_scaffolds.append(scaffold)

        # sanity check
        def get_start_end_direction(_scaff_name, _bin_list, _start_list, _end_list):
            # check that the path in the scaffold is the same the bin_list
            assert self.scaffolds_graph.scaffold.node[_scaff_name]['path'] == _bin_list
            # check direction of scaffold. If the bin id's are decreasing
            # then the direction is "-"
            _scaff_start = _scaff_end = _direction = None
            if len(_bin_list) == 1:
                _direction = "+"
                _scaff_start = _start_list[0]
                _scaff_end = _end_list[0]

            elif all(y - x == -1 for x, y in zip(_bin_list, _bin_list[1:])):
                _direction = "-"
                _scaff_start = min(_start_list)
                _scaff_end = max(_end_list)
                assert _scaff_start == _start_list[-1]
                assert _scaff_end == _end_list[0]

            elif all(y - x == 1 for x, y in zip(_bin_list, _bin_list[1:])):
                _direction = "+"
                _scaff_start = min(_start_list)
                _scaff_end = max(_end_list)
                assert _scaff_start == _start_list[0]
                assert _scaff_end == _end_list[-1]

            else:
                # in this case the bins are not continuous. How did that happen?
                sys.stderr.write('Bins are not continuous. How did that happened?')
            return _scaff_start, _scaff_end, _direction

        scaff_order = {}
        gaps = {}
        for idx, matrix_bin_path in enumerate(self.scaffolds_graph.matrix_bins.get_all_paths()):
            scaff_order[idx] = []
            prev_scaff_name = None
            gaps[idx] = []
            start_list = []
            end_list = []
            bin_list = []
            # matrix_bin path is as list, containing all the bins that form a path
            # those bins are part of scaffolds
            for bin_id in matrix_bin_path:
                # get the scaffold name
                scaff_name, start, end, extra = self.hic.getBinPos(bin_id)
                bin_data = self.scaffolds_graph.matrix_bins.node[bin_id]
                assert bin_data['name'] == scaff_name
                assert bin_data['start'] == start
                assert bin_data['end'] == end

                if scaff_name != prev_scaff_name and prev_scaff_name is not None:
                    scaff_start, scaff_end, direction = get_start_end_direction(prev_scaff_name,
                                                                                bin_list, start_list, end_list)
                    scaff_order[idx].append((prev_scaff_name, scaff_start, scaff_end, direction))
                    start_list = []
                    end_list = []
                    bin_list = []

                start_list.append(start)
                end_list.append(end)
                bin_list.append(bin_id)
                prev_scaff_name = scaff_name

            scaff_start, scaff_end, direction = get_start_end_direction(scaff_name, bin_list, start_list, end_list)
            scaff_order[idx].append((scaff_name, scaff_start, scaff_end, direction))

        # scaffolds that were removed and could not be put back need to be returned as well
        for scaff in self.scaffolds_graph.removed_scaffolds.node.values():
            idx += 1
            scaff_order[idx] = [(scaff['name'], scaff['start'], scaff['end'], '+')]

        if add_split_contig_name is False:
            scaff_order_renamed = {}
            for idx, scaff_path in scaff_order.iteritems():
                scaff_order_renamed[idx] = []
                for scaff_name, scaff_start, scaff_end, scaff_direction in scaff_path:
                    # check if node name has an indication that it was split (by ending in '/n')
                    res = re.search("(.*?)/(\d+)$", scaff_name)
                    if res is not None:
                        scaff_name = res.group(1)
                    scaff_order_renamed[idx].append((scaff_name, scaff_start, scaff_end, scaff_direction))
            scaff_order = scaff_order_renamed

        # brute force comparison
        not_found_list = {}
        for idx, hic_scaff_order in scaff_order.iteritems():
            not_found_list[idx] = []
            for super_scaff in super_scaffolds:
                match = False
                if hic_scaff_order == super_scaff:
                    match = True
                    break
            if match is False:
                not_found_list[idx].append(hic_scaff_order)

        log.debug(not_found_list)
        return scaff_order.values()

    def reorder_matrix(self, max_num_bins=4000, rename_scaffolds=False):
        """
        Reorders the matrix using the assembled paths

        max_num_bins: since reorder is used mostly for plotting, it is required that the matrices are not too large
                      thus, a maximum number of bins can be set.
        rename_scaffolds: Set to true if the original scaffold names that are already merged should be renamed as
                          hic_scaffold_{n} where n is a counter
        Returns
        -------
        """
        import re

        def sorted_nicely(list_to_order):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(list_to_order, key=alphanum_key)

        log.debug("reordering matrix")
        hic = copy.deepcopy(self.scaffolds_graph.hic)
        order_list = []
        from collections import OrderedDict
        scaff_boundaries = OrderedDict()
        start_bin = 0
        end_bin = 0
        num_bins_to_merge = hic.matrix.shape[0] / max_num_bins

        # reduce the density of the matrix if this one is too big
        if hic.matrix.shape[0] > max_num_bins and num_bins_to_merge > 1:
            # compute number of bins required to reduce resolution to desired
            # goal
            num_bins_to_merge = hic.matrix.shape[0] / max_num_bins
            log.debug("Matrix size is too large for printing. Reducing the matrix by merging {} bins".
                      format(num_bins_to_merge))
            hic, map_old_to_merged = HiCAssembler.merge_bins(hic, num_bins_to_merge, skip_small=False,
                                                             return_bin_id_mapping=True)
        else:
            map_old_to_merged = None

        # check if scaffolds are already merged, and if not
        # sort the names alphanumerically.
        if self.scaffolds_graph.scaffold.path == {}:
            scaffold_order = sorted_nicely(list([x for x in self.scaffolds_graph.scaffold]))
            # after merging, small scaffolds will be removed from the matrix. They need
            # to be removed from scaffold_order before reordering the chromosomes to avoid an error
            scaffold_order = [x for x in scaffold_order if x in hic.chrBinBoundaries.keys()]
            hic.reorderChromosomes(scaffold_order)
            hic.chromosomeBinBoundaries = hic.chrBinBoundaries
        else:
            path_list_test = {}
            for idx, scaff_path in enumerate(self.scaffolds_graph.scaffold.get_all_paths()):
                # scaff_path looks like:
                # ['scaffold_12970/3', 'scaffold_12472/3', 'scaffold_12932/3', 'scaffold_12726/3', 'scaffold_12726/1']
                path_list_test[idx] = []
                for scaffold_name in scaff_path:
                    bin_path = self.scaffolds_graph.scaffold.node[scaffold_name]['path']
                    if map_old_to_merged is not None:
                        new_bin_path = []
                        seen = set()
                        for bin_id in bin_path:
                            map_old_to_merged[bin_id]
                            if map_old_to_merged[bin_id] not in seen:
                                new_bin_path.append(map_old_to_merged[bin_id])
                            seen.add(map_old_to_merged[bin_id])
                    else:
                        new_bin_path = bin_path
                    order_list.extend(new_bin_path)
                    path_list_test[idx].extend(new_bin_path)
                    end_bin += len(new_bin_path)

                # assert path_list_test[idx] == self.scaffolds_graph.matrix_bins[path_list_test[idx][0]]
                scaff_boundaries["scaff_{}".format(idx)] = (start_bin, end_bin)
                start_bin = end_bin

            hic.reorderBins(order_list)
            hic.chromosomeBinBoundaries = scaff_boundaries

        if rename_scaffolds is True:
            new_intervals = []
            start_list = []
            for idx, scaff_id in enumerate(hic.chromosomeBinBoundaries):
                start_bin, end_bin = hic.chromosomeBinBoundaries[scaff_id]
                start = 0
                for interval in hic.cut_intervals[start_bin:end_bin]:
                    scaff_name, int_start, int_end, cov = interval
                    end = start + (int_end - int_start)
                    new_intervals.append(("hic_scaffold_{}".format(idx + 1), start, end, cov))
                    start_list.append((start, end, int_start, int_end, int_end - int_start))

                    start = end

            hic.setCutIntervals(new_intervals)

        return hic

    @staticmethod
    def merge_bins(hic, num_bins, skip_small=True, return_bin_id_mapping=False):
        """
        Merge the bins using the specified number of bins. This
        functions takes care to make new intervals

        Parameters
        ----------

        hic : HiCMatrix object

        num_bins : number of consecutive bins to merge.

        Returns
        -------

        A sparse matrix.

        Set up a Hi-C test matrix
        >>> from scipy.sparse import csr_matrix
        >>> row, col = np.triu_indices(5)
        >>> cut_intervals = [('a', 0, 10, 0.5), ('a', 10, 20, 1),
        ... ('a', 20, 30, 1), ('a', 30, 40, 0.1), ('b', 40, 50, 1)]
        >>> hic = HiCMatrix.hiCMatrix()
        >>> hic.nan_bins = []
        >>> matrix = np.array([
        ... [ 50, 10,  5,  3,   0],
        ... [  0, 60, 15,  5,   1],
        ... [  0,  0, 80,  7,   3],
        ... [  0,  0,  0, 90,   1],
        ... [  0,  0,  0,  0, 100]], dtype=np.int32)

        make the matrix symmetric:
        >>> from scipy.sparse import dia_matrix

        >>> dia = dia_matrix(([matrix.diagonal()], [0]), shape=matrix.shape)
        >>> hic.matrix = csr_matrix(matrix + matrix.T - dia)
        >>> hic.setMatrix(hic.matrix, cut_intervals)

        run merge_matrix
        >>> merge_matrix, map_id = HiCAssembler.merge_bins(hic, 2, return_bin_id_mapping=True)
        >>> merge_matrix.cut_intervals
        [('a', 0, 20, 0.75), ('a', 20, 40, 0.55000000000000004), ('b', 40, 50, 1.0)]
        >>> merge_matrix.matrix.todense()
        matrix([[120,  28,   1],
                [ 28, 177,   4],
                [  1,   4, 100]], dtype=int32)
        >>> map_id
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
        """

        hic = hicexplorer.hicMergeMatrixBins.remove_nans_if_needed(hic)
        # get the bins to merge
        ref_name_list, start_list, end_list, coverage_list = zip(*hic.cut_intervals)
        new_bins = []
        bins_to_merge = []
        prev_ref = ref_name_list[0]

        # prepare new intervals
        idx_start = 0
        new_start = start_list[0]
        count = 0
        merge_bin_id = 0
        mapping_old_to_merged_bin_ids = {}
        for idx, ref in enumerate(ref_name_list):
            if (count > 0 and count % num_bins == 0) or ref != prev_ref:
                if skip_small is True and count < num_bins / 2:
                    sys.stderr.write("{} has few bins ({}). Skipping it\n".format(prev_ref, count))
                else:
                    coverage = np.mean(coverage_list[idx_start:idx])
                    new_end = end_list[idx - 1]
                    if new_start > new_end:
                        sys.stderr.write("end of new merged bin is smaller than start")
                    new_bins.append((ref_name_list[idx_start], new_start, end_list[idx - 1], coverage))
                    bins_to_merge.append(list(range(idx_start, idx)))
                    for old_bin_id in list(range(idx_start, idx)):
                        mapping_old_to_merged_bin_ids[old_bin_id] = merge_bin_id
                    merge_bin_id += 1
                idx_start = idx
                new_start = start_list[idx]
                count = 0

            prev_ref = ref
            count += 1

        if skip_small is True and count < num_bins / 2:
            sys.stderr.write("{} has few bins ({}). Skipping it\n".format(prev_ref, count))
        else:
            coverage = np.mean(coverage_list[idx_start:])
            new_end = end_list[idx - 1]
            if new_start > new_end:
                sys.stderr.write("end of new merged bin is smaller than start")
            new_bins.append((ref, new_start, end_list[idx], coverage))
            bins_to_merge.append(list(range(idx_start, idx + 1)))
            for old_bin_id in list(range(idx_start, idx + 1)):
                mapping_old_to_merged_bin_ids[old_bin_id] = merge_bin_id
            merge_bin_id += 1

        hic.matrix = hicexplorer.hicMergeMatrixBins.reduce_matrix(hic.matrix, bins_to_merge, diagonal=True)
        hic.matrix.eliminate_zeros()
        hic.setCutIntervals(new_bins)
        hic.nan_bins = np.flatnonzero(hic.matrix.sum(0).A == 0)

        if return_bin_id_mapping is True:
            return hic, mapping_old_to_merged_bin_ids
        else:
            return hic


class HiCAssemblerException(Exception):
        """Base class for exceptions in HiCAssembler."""
