import numpy as np
from scipy.sparse import triu, lil_matrix
import networkx as nx
import inspect
import time
import os.path
import copy


import hicexplorer.HiCMatrix as HiCMatrix
from hicexplorer.iterativeCorrection import iterativeCorrection
from hicexplorer.reduceMatrix import reduce_matrix
import hicexplorer.hicFindTADs as hicFindTADs
from functools import wraps
from hicassembler.Scaffolds import Scaffolds

import logging
log = logging.getLogger("HiCAssembler")
log.setLevel(logging.DEBUG)
# log.setLevel(logging.INFO)

#log.basicConfig(format='%(levelname)s[%(funcName)s]:%(message)s', level=logging.DEBUG)

POWER_LAW_DECAY = 2**(-1.08)  # expected exponential decay at 2*distance

MIN_LENGTH = 400000  # minimum contig or PE_scaffold length to consider
ZSCORE_THRESHOLD = -1 # zscore threshold to declare a boundary a misassembly
MIN_MAD = -0.5  # minimum zscore row contacts to filter low scoring bins
MAX_MAD = 50  # maximum zscore row contacts
MAX_INT_PER_LENGTH = 100  # maximum number of HiC pairs per length of contig
MIN_COVERAGE = 0.7
TEMP_FOLDER = '/tmp/'
ITER = 40
MIN_MATRIX_VALUE = 5

SIM = False
EXP = True
debug = 1


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
                 num_processors=5, misassembly_zscore_threshold=ZSCORE_THRESHOLD):
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

        if not isinstance(hic_file_name, str):
            # assume that the hic given is already a HiCMatrix object
            # this is normally used for testing
            self.hic = hic_file_name
        else:
            log.info("Loading Hi-C matrix ... ")
            # check if a lower resolution matrix is available
            merged_bins_matrix_file = self.out_folder + "/hic_merged_bins_matrix.h5"            # check if
            if os.path.isfile(merged_bins_matrix_file):
                log.info("Found reduced matrix file {}".format(merged_bins_matrix_file))
                self.hic = HiCMatrix.hiCMatrix(merged_bins_matrix_file)
            else:
                self.hic = HiCMatrix.hiCMatrix(hic_file_name)
                log.info("Merging bins of file to reduce resolution")
                binsize = self.hic.getBinSize()
                if binsize < matrix_bin_size:
                    # make an smaller matrix having bins of around 25.000 bp
                    num_bins = matrix_bin_size / binsize

                    from hicexplorer.hicMergeMatrixBins import merge_bins
                    log.info("Reducing matrix size to {:,} bp (number of bins merged: {})".format(binsize, num_bins))
                    self.hic = merge_bins(self.hic, num_bins)

                self.hic.save(merged_bins_matrix_file)
                self.hic = HiCMatrix.hiCMatrix(merged_bins_matrix_file)

        if use_log:
            self.hic.matrix.data = np.log1p(self.hic.matrix.data)

        self.min_mad = min_mad
        self.max_mad = max_mad
        self.min_scaffold_length = min_scaffold_length
        self.num_processors = num_processors
        self.misassembly_threshold = misassembly_zscore_threshold
        self.merged_paths = None
        self.iteration = 0

        #self.remove_noise_from_matrix()

        # build scaffolds graph. Bins on the same contig are
        # put together into a path (a type of graph with max degree = 2)
        self.scaffolds_graph = Scaffolds(copy.deepcopy(self.hic), self.out_folder)

        self.plot_matrix(self.out_folder + "/before_assembly.pdf", title="Before assembly")
        if split_misassemblies:
            # try to find contigs that probably should be separated
            self.split_misassemblies(hic_file_name, split_positions_file)
            # build scaffolds graph. Bins on the same contig are
            # put together into a path (a type of graph with max degree = 2)
            self.scaffolds_graph = Scaffolds(copy.deepcopy(self.hic), self.out_folder)
            self.plot_matrix(self.out_folder + "/after_split_assembly.pdf",
                             title="After split mis-assemblies assembly", add_vlines=True)

        mat_size = self.hic.matrix.shape[:]
        # remove contigs that are too small
        self.scaffolds_graph.remove_small_paths(self.min_scaffold_length, split_scaffolds=True)
        assert mat_size == self.scaffolds_graph.hic.matrix.shape

        self.N50 = []

    def assemble_contigs(self):
        """

        Returns
        -------

        """
        log.debug("Size of matrix is {}".format(self.scaffolds_graph.hic.matrix.shape[0]))
        for iteration in range(4):
            self.iteration = iteration
            n50 = self.scaffolds_graph.compute_N50()
            self.scaffolds_graph.get_paths_stats()

            log.debug("iteration: {}\tN50: {:,}".format(iteration, n50))
            self.N50.append(n50)

            # the first iteration is is more stringent
            if iteration < 2:
                target_size = int(min(2e6, self.scaffolds_graph.paths_min * (iteration + 1)))
                log.debug("Merging small bins in larger bins of size {} bp".format(target_size))
                self.scaffolds_graph.split_and_merge_contigs(num_splits=3,
                                                             target_size=target_size,
                                                             normalize_method='ice')
                stats = self.scaffolds_graph.get_stats_per_split()
                if iteration == 1:
                    conf_score = stats[2]['median']
                else:
                    conf_score = stats[2]['median'] * (1.0 / (iteration + 1))

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

        after_assembly_length, afeger_num_paths = self.scaffolds_graph.get_assembly_length()

        diff = after_assembly_length - before_assembly_length
        log.info('{:,} bp ({:.2%}) were added back to the assembly'.format(diff,
                                                                           float(diff)/self.scaffolds_graph.total_length))

        log.info('Total assembly length: {:,} bp ({:.2%})'.format(after_assembly_length,
                                                                  float(after_assembly_length)/self.scaffolds_graph.total_length))

        self.plot_matrix(self.out_folder + "/after_put_scaff_back.pdf".format(iteration), title="After assembly", add_vlines=True)

        print self.N50
        return self.get_contig_order()

    def make_scaffold_network(self, orig_scaff):
        """

        Parameters
        ----------
        orig_scaff

        Returns
        -------

        Examples
        --------

        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 30, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 0, 10, 1), ('c-2', 10, 30, 1)]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> H = HiCAssembler(hic, "", "/tmp/test/", split_misassemblies=False,
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
        max_weight = float(orig_scaff.matrix.max() * 1.5)
        for u, v, weight in zip(matrix.row, matrix.col, matrix.data):
            if u == v:
                continue
            scaff_u = orig_scaff.pg_base.node[u]['name']
            scaff_v = orig_scaff.pg_base.node[v]['name']
            nxG.add_edge(scaff_u, scaff_v, weight=float(weight))
            if scaff_u in self.scaffolds_graph.scaffold.node and \
               scaff_v in self.scaffolds_graph.scaffold.node and \
               scaff_u in self.scaffolds_graph.scaffold.adj[scaff_v]:
                # u and v are directly joined
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
        None
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

        The graph used should not contains nodes with degree > 2 execpt for the
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
        Identifies scaffolds that where removed from the Hi-C assembly and
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

        orig_scaff = Scaffolds(self.hic)
        orig_scaff.split_and_merge_contigs(num_splits=1, normalize_method=normalize_method)

        # reset pb_base
        self.scaffolds_graph.pg_base = copy.deepcopy(self.scaffolds_graph.matrix_bins)
        nxG = self.make_scaffold_network(orig_scaff)

        nxG = nx.maximum_spanning_tree(nxG, weight='weight')
        nx.write_graphml(nxG, self.out_folder + "/mst_for_small_Scaff_integration.graphml".format())

        # 1. Identify branches

        # delete backbone nodes that are not adjacent to a removed scaffold
        for node_id in self.scaffolds_graph.scaffold.node.keys():
            # check that the backbone node is not adjacent to a removed node.
            if len(set(nxG.adj[node_id].keys()).intersection(self.scaffolds_graph.removed_scaffolds.node.keys())) == 0:
                nxG.remove_node(node_id)

        # remove backbone edges
        for u, v in list(nxG.edges()):
            if 'is_backbone' in nxG.node[u] and 'is_backbone' in nxG.node[v]:
                nxG.remove_edge(u, v)

        for branch in list(nx.connected_component_subgraphs(nxG)):
            if len(branch) > 10:
                log.info("Skipping the insertion of a branch of length: {}".format(len(branch)))
                continue
            # afer removing the hubs the branch may contain several connected components. Only the component
            # that contains a backbone node is used.
            backbone_list = HiCAssembler._find_backbone_node(branch)

            if len(backbone_list) == 0:
                log.debug("No backbone found for branch with nodes: {}".format(branch.node.keys()))

            # each branch should contain only one backbone node
            if len(backbone_list) > 1:
                log.info("Branch contains more than one backbone. Backbones in branch: {}".format(backbone_list))
                log.info("Skipping this branch of length: {}".format(len(branch)))
                continue

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

                log.info("Scaffolds {} successfully integrated into the network".format(path[1:]))

        return

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
        # TODO here the hic_file is loaded unnecesarily. A way to remove this step would be good
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

        log.info("Splitting scaffolds using threshold = {}".format(self.misassembly_threshold))
        for idx in np.flatnonzero(zscore < self.misassembly_threshold):
            # find the bins that overlap with the misassembly
            if scaffold[idx] not in bin_ids:
                bin_ids[scaffold[idx]] = []
            to_split_intervals = sorted(self.hic.interval_trees[scaffold[idx]][start[idx]:end[idx]])
            bin_ids[scaffold[idx]].extend(sorted([interval_bin.data for interval_bin in to_split_intervals]))

        if split_positions_file is not None:
            log.debug("loading positions to split")
            from hicexplorer import readBed

            bed_file_h = readBed.ReadBed(open(split_positions_file, 'r'))
            for bed in bed_file_h:
                # find the bins that overlap with the misassembly
                if bed.chromosome not in self.hic.interval_trees:
                    continue
                if bed.chromosome not in bin_ids:
                    bin_ids[bed.chromosome] = []
                to_split_intervals = sorted(self.hic.interval_trees[bed.chromosome][bed.start:bed.end])
                bin_ids[bed.chromosome].extend(sorted([interval_bin.data for interval_bin in to_split_intervals]))

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
        chrbin_boundaries = hic.chrBinBoundaries

        axHeat2 = fig.add_subplot(111)
        axHeat2.set_title(title)
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
        >>> from hicassembler.Scaffolds import get_test_matrix as get_test_matrix
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 30, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2/1', 0, 10, 1), ('c-2/2', 10, 30, 1)]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> H = HiCAssembler(hic, "", "/tmp/test/", split_misassemblies=False, min_scaffold_length=0)
        >>> H.scaffolds_graph.add_edge_matrix_bins(1,2)
        >>> H.get_contig_order()
        [[('c-2', 10, 30, '+')], [('c-2', 0, 10, '+')], [('c-0', 0, 30, '+'), ('c-1', 0, 20, '+')]]

        >>> H.get_contig_order(add_split_contig_name=True)
        [[('c-2/2', 10, 30, '+')], [('c-2/1', 0, 10, '+')], [('c-0', 0, 30, '+'), ('c-1', 0, 20, '+')]]
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

        return super_scaffolds

    def reorder_matrix(self):
        """
        Reorders the matrix using the assembled paths

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

        # check if scaffolds are already merged, and if not
        # sort the names alphanumerically.
        if self.scaffolds_graph.scaffold.path == {}:
            scaffold_order = sorted_nicely(list([x for x in self.scaffolds_graph.scaffold]))
            hic.reorderChromosomes(scaffold_order)
            hic.chromosomeBinBoundaries = hic.chrBinBoundaries
        else:
            path_list_test = {}
            for idx, scaff_path in enumerate(self.scaffolds_graph.scaffold.get_all_paths()):
                path_list_test[idx] = []
                for scaffold_name in scaff_path:
                    bin_path = self.scaffolds_graph.scaffold.node[scaffold_name]['path']
                    order_list.extend(bin_path)
                    path_list_test[idx].extend(bin_path)
                    end_bin += len(bin_path)

                assert path_list_test[idx] == self.scaffolds_graph.matrix_bins[path_list_test[idx][0]]
                scaff_boundaries["scaff_{}".format(idx)] = (start_bin, end_bin)
                start_bin = end_bin

            hic.reorderBins(order_list)
            hic.chromosomeBinBoundaries = scaff_boundaries
        return hic

    def get_nearest_neighbors_2(self, paths, min_neigh=1, trans=True, threshold=0,
                                max_int=None):
        """ Thu, 23 May 2013 16:14:51
        The idea is to take the matrix and order it
        such that neighbors are placed next to each other.

        The algorithm is as follows:

        1. identify the cell with  highest number of shared pairs and
        save the pair of rows that corresponds to that cell.
        2. remove that number from the matrix.
        3. Quit if all nodes already have 'min_neigh' neighbors,
           otherwise repeat.

        In theory, if everything is consistent, at the end of
        the for loop each node should have *only* two
        neighbors. This is rarely the case and some of the nodes
        end up with more than two neighbors. However, a large
        fraction of the nodes only have two neighbors and they
        can be chained one after the other to form a super
        contig.

        Parameters:
        ----------
        min_neigh: minimun number of neighbors to consider.
                   If set to two, the function exists when
                   all contigs have at least two neighbors.
        """
        print "[{}]".format(inspect.stack()[0][3])

        # consider only the upper triangle of the
        # matrix and convert it to COO for quick
        # operations
        try:
            ma = triu(self.cmatrix, k=1, format='coo')
        except:
            import pdb; pdb.set_trace()
        order_index = np.argsort(ma.data)[::-1]
        # neighbors dictionary
        # holds, for each node the list of neighbors
        # using a sparse matrix is much faster
        # than creating a network and populating it
        net = lil_matrix(ma.shape, dtype='float64')

        # initialize neighbors dict
        neighbors = {}
        for index in range(ma.shape[0]):
            neighbors[index] = (0,0)

        counter = 0
        for index in order_index:
            counter += 1
            if counter % 10000 == 0:
                print "[{}] {}".format(inspect.stack()[0][3], counter)
            row = ma.row[index]
            col = ma.col[index]
            if col == row:
                continue
            if ma.data[index] < threshold:
                break

            # do not add edges if the number of contacts
            # in the uncorrected matrix is below
            if self.matrix[row, col] < 10:
                continue


            # if a contig (or path when they are merged)
            # has already two neighbors and the second
            # neighbor has a number of contacts equal to max_int
            # it means that this node is already connected to
            # two other nodes that already have been decided
            # and further connections to such nodes are skipped.
            # It is only necesary to check if the last neighbor
            # added to the node is equal to max_int because
            # the first node, must have also been max_int.
            if neighbors[row][1] == 2 and neighbors[row][0] == max_int:
                continue
            if neighbors[col][1] == 2 and neighbors[col][0] == max_int:
                continue

            # add an edge if the number of neighbors
            # is below min_neigh:
            if neighbors[row][1] < min_neigh and \
                    neighbors[col][1] < min_neigh:
                neighbors[row] = (ma.data[index], neighbors[row][1]+1)
                neighbors[col] = (ma.data[index], neighbors[col][1]+1)
                net[row, col] = ma.data[index]

            # add further edges if a high value count exist
            # that is higher than the expected power law decay
            elif ma.data[index] > neighbors[row][0]*POWER_LAW_DECAY and \
                    ma.data[index] > neighbors[col][0]*POWER_LAW_DECAY:
                if neighbors[row][1] < min_neigh:
                    neighbors[row] = (ma.data[index], neighbors[row][1]+1)

                if neighbors[col][1] < min_neigh:
                    neighbors[col] = (ma.data[index], neighbors[col][1]+1)
                net[row, col] = ma.data[index]

        G = nx.from_scipy_sparse_matrix(net, create_using=nx.Graph())
        if trans:
            # remap ids
            mapping = dict([(x,paths[x][0]) for x in range(len(paths))])
            G = nx.relabel_nodes(G, mapping)

        # remove all edges not connected to flanks
        # the idea is that if a path already exist
        # no edges should point to it unless
        # is they are the flanks
        """
        if self.iteration==2:
            import pdb;pdb.set_trace()
        if self.merged_paths:
            flanks = set(HiCAssembler.flatten_list(
                    [[x[0],x[-1]] for x in self.merged_paths]))
            for edge in G.edges():
                if len(flanks.intersection(edge)) == 0:
                    G.remove_edge(*edge)
        """
        return G

    def assemble_super_contigs(self, G, paths, max_int, add_first_neighbors=True):
        """Mon, 14 Oct 2013 22:10:46 +0200
        Uses networkx to find shortest path
        joining connected - components

        Params:
        ------
        G: a networkx object
        paths
        max_int
        add_first_neighbors

        Returns:
        -------

        super_contigs: list of lists. Each list member
        contains the sequence of sequential contigs
        """
        log.info("Assembling contigs")
        for node in G.nodes(data=True):
            G.add_node(node[0], label="{}({})".format(paths[node[0]], node[0]), id=node[0])
        nx.write_gml(G, "./prepros_g_{}.gml".format(self.iteration))

        cen = nx.degree(G)
        high_deg, degree = zip(*cen.iteritems())
        # sort hubs by increasing degree and filter by degree > 2
        high_deg = [high_deg[x] for x in HiCAssembler.argsort(degree) if degree[x] > 2]
        while len(high_deg) > 0:
            node = high_deg.pop(0)
            if G.degree(node) > 3:
                # test if some of the neighbors can be kept because
                # of the overlap graph
                u = paths[x][0]
                to_keep = []
                weights, nodes = zip(*[(v['weight'], k)
                                       for k, v in G[node].iteritems()])

                # filter the nodes list to keep decided edges
                nodes = [nodes[idx] for idx, value in enumerate(weights) if value < max_int]
                for n in nodes:
                    v = paths[node][0]
                    if self.overlap.have_direct_overlap(u, v):
                        to_keep.append(n)
                if len(to_keep) > 2:
                    # suspicious
                    to_keep = []
                elif len(to_keep):
                    import pdb; pdb.set_trace()
                    log_m = "\n".join(["{}".format(G[node][x]) for x in to_keep])
                    log.debug("Found direct overlap for a neighbor of {}. "
                              "Keeping neighbor {}".format(node, log_m))

                log.debug("High degree node {} {} removed from graph".format(node, G[node]))
                G.remove_edges_from([(node, y) for y in G[node].keys() if y not in to_keep])

        # first pass
        # remove all hubs.
        # Hubs are defined as nodes having a degree of at least 4
        # whose edges weight do not show a power law decay.

        cen = nx.degree(G)
        high_deg, degree = zip(*cen.iteritems())
        # sort hubs by increassing degree and filter by
        # degree > 2
        high_deg = [high_deg[x] for x in HiCAssembler.argsort(degree) if degree[x] > 2]
        while len(high_deg) > 0:
            node = high_deg.pop(0)
            # while resolving other nodes,
            # the current node may have now
            # degree of 2 or less and is not
            # neccesary further remove any edge
            if G.degree(node) <= 2:
                continue
            """
            weights, nodes = zip(*[(v['weight'], k)
                                   for k, v in G[node].iteritems()])
            order = HiCAssembler.argsort(weights)[::-1]

            max_nodes = len([x for x in weights if x==max_int])
            nodes = list(nodes)
            # if one of the edges already has the maximun value
            # discard it.
            if max_nodes == 1:
                # to discard the max_node is only needed to
                # remove it from the order list.
                order.pop(0)

            # skip hubs
            if len(nodes) > 3 and weights[order[-1]] >  weights[order[0]] * POWER_LAW_DECAY:
                continue
            """
            if 2 == 2:
#            else:
                direct_neighbors = G[node].keys()
                if add_first_neighbors:
                    first_neighbors = HiCAssembler.flatten_list([G[x].keys() for x in direct_neighbors])
                    neighbors = np.unique(direct_neighbors + first_neighbors)
                else:
                    neighbors = direct_neighbors + [node]
                fixed_paths = HiCAssembler.get_fixed_paths(G, neighbors, max_int)
                if len(neighbors)-len(fixed_paths) > 5:
                    if debug:
                        log.debug("node {} has too many neighbors {}\n{}\n{}\n{}"
                                  "".format(node,
                                            neighbors,
                                            [self.paths[x] for x in neighbors],
                                            [int(self.cmatrix[node, x]) for x in neighbors],
                                            G[node]))
                    continue
#                    G.remove_edges_from([(node, y) for y in direct_neighbors])
                else:
                    bw_order = HiCAssembler.permute(self.cmatrix, neighbors,
                                                    fixed_paths)

                    # remove from G
                    node_idx = bw_order.index(node)
                    true_neigh = []
                    if node_idx > 0:
                        true_neigh.append(bw_order[node_idx-1])
                    if node_idx < len(bw_order) - 1:
                        true_neigh.append(bw_order[node_idx+1])
                    # check if the 'true neighbors' where in the list
                    # of direct neighbors. When this not happends
                    # this may indicate a spourius case and is better
                    # to avoid it.
                    if len(set(direct_neighbors).intersection(true_neigh)) != len(true_neigh):
                        true_neigh = []

                    if debug:
                        correct = True
                        for tn in true_neigh:
                            if abs(tn - node) != 1:
                                correct = False
                        if correct is False:
                            log.error("WRONG bw permutation of node: {} is: {}\n{}\n{}\n{}"
                                      "".format(node, bw_order,
                                                [self.paths[x] for x in bw_order],
                                                [int(self.cmatrix[bw_order[x], bw_order[x + 1]])
                                                 for x in range(len(bw_order)-1)],
                                                [int(self.matrix[bw_order[x], bw_order[x+1]])
                                                 for x in range(len(bw_order) - 1)]))
                            self.bad_bw += 1
                        else:
                            print "bw worked fine!"
                            self.good_bw += 1
                    G.remove_edges_from([(node, y) for y in direct_neighbors
                                         if y not in true_neigh])

        ## third pass:
        # remove any hub left in the graph
        cen = nx.degree(G)
        high_deg = [k for k, v in cen.iteritems() if v > 2]
        while len(high_deg) > 0:
            node = high_deg.pop(0)
            if G.degree(node) <=2:
                continue
            direct_neighbors = G[node].keys()
            if add_first_neighbors:
                first_neighbors =HiCAssembler.flatten_list([G[x].keys()
                                                           for x in direct_neighbors])
                neighbors = np.unique(direct_neighbors + first_neighbors)
            else:
                neighbors = direct_neighbors + [node]
            fixed_paths = HiCAssembler.get_fixed_paths(G, neighbors, max_int)
            if len(neighbors)-len(fixed_paths) > 5:
                if debug:
                    print "node {} has too many neighbors {}\n{}\n{}\n{}".format(
                    node,neighbors,
                    [self.paths[x] for x in neighbors],
                    [int(self.cmatrix[node, x])
                     for x in neighbors], G[node])
                G.remove_edges_from([(node, y) for y in direct_neighbors])
            else:
                log.debug("High degree node {} {} removed "
                              "from graph".format(node,
                                                  G[node]))
                G.remove_edges_from([(node, y) for y in G[node].keys()])

        ### just for debugging purposes
        cen = nx.degree(G)
        high_deg = [k for k, v in cen.iteritems() if v > 2]
        if len(high_deg):
            print "not all hubs were flattened {}".format(high_deg)
            import pdb;pdb.set_trace()
        #####

        # break self loops
        conn = nx.connected_components(G)
        for nodes in conn:
            if len(nodes)==1:
                continue
            # get all nodes with degree 1.
            # This should be the nodes at the
            # opposite sides. All other nodes
            # in between should have a degree equal to 2
            deg_one = [x for x in nodes if G.degree(x) == 1]
            if len(deg_one) != 2:
                if len(nodes) > 5:
                    fixed_paths = HiCAssembler.get_fixed_paths(G, nodes, max_int)
                else:
                    fixed_paths = []
                if len(nodes) - len(fixed_paths) <= 6:
                    bw_order = HiCAssembler.permute(self.matrix, nodes,
                                                    fixed_paths)
                    # break the closed loop by setting the
                    # bw_order as path
                    for edge in G.edges(nodes):
                        G.remove_edge(*edge)
                    G.add_path(bw_order)
                    print '[{}] The nodes form a closed loop' \
                        'because no opposite sides exist' \
                        'bw order {}'.format(inspect.stack()[0][3], bw_order)
                else:
                    print "closed loop found but is too large for bw "\
                        "permutation. Removing weakest link"

        # add a label to the nodes
        for node in G.nodes(data=True):
            G.add_node(node[0], label="{}({})".format(paths[node[0]],node[0]),
                       id=node[0])
        nx.write_gml(G,
                     "./net_g_{}.gml".format(self.iteration))

        # after the while loop
        # only paths should be found and
        # all edges can be added
        for edge in G.edges_iter():
            u, v = [paths[x][0] for x in edge]

            if self.iteration <= 2:
                # check that other direct neighbors also share
                # interactions
                try:
                    direct_u = self.scaffolds.get_neighbors(u)
                    direct_v = self.scaffolds.get_neighbors(v)
                    if len(direct_u) ==  1 and len(direct_v) == 1:
                        if set([direct_u[0], direct_v[0]]) != set([u,v]) and \
                               self.hic.matrix[u, direct_v[0]] < 5 and self.hic.matrix[v, direct_u[0]] < 5:
                            log.debug('dropping interaction between {} {} ({}, {} (neighbors: {}, {})) '
                                          'because neighbors do not interact: {}'.format(
                                    edge[0], edge[1], u, v, direct_u[0], direct_v[0], self.hic.matrix[direct_u[0], direct_v[0]]))
                            if debug:
                                if abs(u-v) == 1:
                                    log.debug("error dropping previous edge")
    #                                import pdb;pdb.set_trace()
                        continue
                except IndexError:
                    import pdb;pdb.set_trace()

            if self.matrix[edge[0],edge[1]] < 10:
                log.debug('dropping interaction between {} {} ({}, {}) '
                              'because raw number of edges is too low: {}'.format(
                        edge[0], edge[1], u, v, self.matrix[edge[0],edge[1]]))
                continue
            self.add_edge(u,v,
                          iteration=self.iteration,
                          merged_pairs = self.matrix[edge[0], edge[1]],
                          normalized_merged_pairs = self.cmatrix[edge[0], edge[1]],
                          shared_pairs=self.hic.matrix[u,v],
                          normalized_contacts=self.cmatrix_orig[u,v])

        return self.scaffolds.get_all_paths()

    @staticmethod
    def flatten_list(alist):
        """
        given a list of list of list, returns a list
        For example: given [1,[[2]], [3,4]]
        returns [1, 2, 3 4]
        This is a recursive function.

        Parameters
        ----------
        alist : list of list of list from subsequent merges.

        Returns
        -------
        list flattened

        >>> HiCAssembler.flatten_list([1,[[2]], [3,4]])
        [1, 2, 3, 4]
        """
        ret = []
        for values in alist:
            if type(values) != list:
                ret.append(values)
            else:
                ret.extend(HiCAssembler.flatten_list(values))
        return ret

    @staticmethod
    def get_flanks(path, flank_length, contig_len, recursive_repetitions, counter=0):
        """
        Takes a path and returns the flanking regions plus the inside. This is a
        recursive function and will split the inside as many times as possible,
        stopping when 'recursive_repetitions' have been reached

        Parameters
        ----------
        path : list of ids
        flank_length : length in bp of the flank lengths that want to be kept
        contig_len : list with the lengths of the contig/bis in the path. The
                     index in the list are considered to be the length for bin
                     id.
        recursive_repetitions : If recursive, then the flanks of the inside
                    part of the path (the interior part) are also returned

        counter : internal counter to keep track of recursive repetitions

        Returns
        -------

        Examples
        --------

        The flank length is set to 2000, thus, groups of two should be
        selected
        >>> path = [1,2,5,3,4]
        >>> flank_length = 2000

        the index in the contig_len array matches
        the path ids. Thus, in the following example
        the value 2000 is asigned to id 5 in the path
        >>> contig_len = np.array([1000,1000,1000,1000,1000,2000])
        >>> HiCAssembler.get_flanks(path, flank_length, contig_len, 3)
        [[1, 2], [5], [4, 3]]

        Same as before, but now I set id 5 to be smaller.
        Now 5 should be skipped as is not at least flank_length * 0.75
        >>> contig_len = np.array([1000,1000,1000,1000,1000,800])
        >>> HiCAssembler.get_flanks(path, flank_length, contig_len, 3)
        [[1, 2], [4, 3]]

        Get the flanks, and do not recursively iterate
        >>> HiCAssembler.get_flanks(path, 1000, np.array([1000,1000,1000,1000,1000,1000]), 1)
        [[1], [4]]

        Get the flanks, and iterate twice
        >>> path = [1,2,3,4,5,6,7,8,9]
        >>> contig_len = np.array([1,1,1,1,1,1,1,1,1,1,1])
        >>> HiCAssembler.get_flanks(path, 2, contig_len, 2)
        [[1, 2], [3, 4], [7, 6], [9, 8]]
            """
        counter += 1
        if counter > recursive_repetitions:
            return []

        tolerance_max = flank_length * 1.25
        tolerance_min = flank_length * 0.75
        path_length_sum = sum(contig_len[path])
        flanks = []

        def _get_path_flank(_path):
            """
            returns the first k ids in a path such that the sum of the lengths for _path[k] is
            between tolerance_min and tolerance_max

            Parameters
            ----------
            _path : list of ids

            Returns
            -------
            _path[0:k] such that the length of the bins in _path[0:k] is between tolerance_min and tolerance_max
            """
            flank = []
            for x in _path:
                flank_sum = sum(contig_len[x] for x in flank)
                if flank_sum > tolerance_max:
                    break
                elif tolerance_min <= flank_sum <= tolerance_max:
                    break
                flank.append(x)
            return flank

        if len(path) == 1:
            if contig_len[path[0]] > tolerance_min or counter == 1:
                flanks = [path]
        else:
            if path_length_sum < 2*flank_length*0.75:
                # if the total path length is shorter than twice the flank_lengh *.75
                # then split the path into two
                log.debug("path {} is being divided into two, although is quite small {}".format(path, path_length_sum))
                path_half = len(path)/2
                left_flank = path[0:path_half]
                right_flank = path[path_half:][::-1]
                flanks.extend([left_flank, right_flank])
            else:
                left_flank = _get_path_flank(path)
                right_flank = _get_path_flank(path[::-1])

                # check if the flanks overlap
                over = set(left_flank).intersection(right_flank)
                if len(over):
                    # remove overlap
                    left_flank = [x for x in left_flank if x not in over]

                if len(left_flank) == 0 or len(right_flank) == 0:
                    path_half = len(path)/2
                    left_flank = path[0:path_half]
                    right_flank = path[path_half:][::-1]

                interior = [x for x in path if x not in HiCAssembler.flatten_list(left_flank + right_flank)]

                if len(interior):
                    interior = HiCAssembler.get_flanks(interior, flank_length, contig_len,
                                                       recursive_repetitions, counter=counter)
                if len(left_flank):
                    flanks.append(left_flank)
                if len(interior):
                    flanks.extend(interior)
                if len(right_flank):
                    flanks.append(right_flank)

            try:
                if len(left_flank) == 0 or len(right_flank) == 0:
                    import pdb;pdb.set_trace()
            except:
                pass

        return flanks


class HiCAssemblerException(Exception):
        """Base class for exceptions in HiCAssembler."""

