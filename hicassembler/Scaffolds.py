from collections import OrderedDict
import copy
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, triu
import logging
import time
import hicexplorer.HiCMatrix as HiCMatrix
from hicassembler.PathGraph import PathGraph, PathGraphEdgeNotPossible, PathGraphException

from hicexplorer.reduceMatrix import reduce_matrix
from hicexplorer.iterativeCorrection import iterativeCorrection
from functools import wraps
import itertools
import networkx as nx


logging.basicConfig()
log = logging.getLogger("Scaffolds")
log.setLevel(logging.INFO)


def logit(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        log.info("Entering " + func.__name__)
        start_time = time.time()
        f_result = func(*args, **kwds)
        elapsed_time = time.time() - start_time
        log.info("Exiting {} after {} seconds".format(func.__name__, elapsed_time))
        return f_result
    return wrapper


class Scaffolds(object):
    """
    This class is a place holder to keep track of the iterative scaffolding.
    The underlying data structure is a special directed graph that does
    not allow more than two edges per node.

    The list of paths in the graph (in the same order) is paired with
    the rows in the HiC matrix.

    Example:

    Init a small HiC matrix
    >>> hic = get_test_matrix()
    >>> S = Scaffolds(hic)

    the list [('c-0', 0, 1, 1), ... ] has the format of the HiCMatrix attribute cut_intervals
    That has the format (chromosome name or contig name, start position, end position). Each
    HiC bin is determined by this parameters


    """
    def __init__(self, hic_matrix, out_folder=None):
        """

        Parameters
        ----------
        cut_intervals

        Returns
        -------

        Examples
        -------
        >>> hic = get_test_matrix()
        >>> S = Scaffolds(hic)

        """
        # initialize the list of contigs as a graph with no edges
        self.hic = hic_matrix
        self.matrix = None  # will contain the reduced matrix
        self.total_length = None
        self.out_folder = '/tmp/' if out_folder is None else out_folder
        # three synchronized PathGraphs are used
        # 1. matrix_bins contains the bin id related to the hic matrix. This is the most lower level PathGraph
        # and the ids always match the ids in the self.hic.matrix
        self.matrix_bins = PathGraph()

        # 2. scaffold PathGraph, whose nodes are the scaffolds/contigs in the matrix. Each node in this graph contains
        # a reference to the matrix bins it contains
        self.scaffold = PathGraph()

        # 3. this PathGraph contains iterative merges of the self.hic.matrix. It is dynamic and changing in contrast to
        # the other matrices. The nodes in this matrix, match the nodes of the self.matrix after a merge round
        # occurs
        self.pg_base = None

        # keep track of removed scaffolds
        self.removed_bins = PathGraph()
        self.removed_scaffolds = PathGraph()

        # initialize the contigs directed graph
        self._init_path_graph()

        self.iteration = 0

    def _init_path_graph(self):
        """Uses the hic information for each row (cut_intervals)
        to initialize a path graph in which each node corresponds to
        a contig or a bin in a contig

        This method is called by the __init__ see example there

        Parameters
        ----------
        cut_intervals : the cut_intervals attribute of a HiCMatrix object

        Returns
        -------

        Example
        -------
        >>> cut_intervals = [('c-0', 0, 1, 1), ('c-1', 0, 1, 1), ('c-2', 0, 1, 1),
        ... ('c-4', 0, 1, 1), ('c-4', 0, 1, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)

        >>> S = Scaffolds(hic)
        >>> S.pg_base[0]
        [0]
        >>> S.pg_base[4]
        [3, 4]
        >>> S.pg_base.path == {'c-0': [0],
        ...                    'c-1': [1],
        ...                    'c-2': [2],
        ...                    'c-4': [3, 4]}
        True

        >>> S.scaffold.node['c-4']['path'] == [3, 4]
        True

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-1', 10, 20, 1),
        ... ('c-1', 20, 30, 1), ('c-2', 0, 10, 1), ('c-2', 10, 20, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)

        >>> S = Scaffolds(hic)
        >>> S.scaffold.node['c-0']
        {'direction': '+', 'end': 20, 'name': 'c-0', 'start': 0, 'length': 20, 'path': [0, 1]}
        """

        def add_scaffold_node(name, path, length):
            # prepare scaffold information
            scaff_start = self.hic.cut_intervals[contig_path[0]][1]
            scaff_end = self.hic.cut_intervals[contig_path[-1]][2]
            #length = scaff_end - scaff_start
            attr = {'name': prev_label,
                    'path': contig_path[:],
                    'length': length,
                    'start': scaff_start,
                    'end': scaff_end,
                    'direction': "+"}
            self.scaffold.add_node(prev_label, **attr)

        contig_path = []
        prev_label = None
        self.matrix = self.hic.matrix.copy()
        self.matrix_bins = PathGraph()
        self.scaffold = PathGraph()
        self.bin_id_to_scaff = OrderedDict()
        self.total_length = 0
        scaff_length = 0
        for idx, interval in enumerate(self.hic.cut_intervals):
            label, start, end, coverage = interval
            length = end - start
            if prev_label is not None and prev_label != label:
                self.matrix_bins.add_path(contig_path, name=prev_label)
                add_scaffold_node(prev_label, contig_path, scaff_length)
                contig_path = []
                scaff_length = 0

            scaff_length += length
            attr = {'name': label,
                    'start': start,
                    'end': end,
                    'coverage': coverage,
                    'length': length}

            self.matrix_bins.add_node(idx, **attr)
            self.bin_id_to_scaff[idx] = label
            self.total_length += length
            contig_path.append(idx)
            prev_label = label
        if prev_label is not None:
            add_scaffold_node(prev_label, contig_path, scaff_length)

        if len(contig_path) > 1:
            self.matrix_bins.add_path(contig_path, name=label)

        # before any merge is done, pg_base == pg_matrix.bins
        self.pg_base = copy.deepcopy(self.matrix_bins)

    def get_all_paths(self, pg_base=False):
        """Returns all paths in the graph.
        This is similar to get connected components in networkx
        but in this case, the order of the returned  paths
        represents scaffolds of contigs

        >>> cut_intervals = [('c-0', 0, 1, 1), ('c-0', 1, 2, 1), ('c-0', 2, 3, 1),
        ... ('c-2', 0, 1, 1), ('c-2', 1, 2, 1), ('c-3', 0, 1, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> [x for x in S.get_all_paths()]
        [[0, 1, 2], [3, 4], [5]]
        """
        seen = set()
        if pg_base and self.pg_base is not None:
            pathgraph = self.pg_base
        else:
            pathgraph = self.matrix_bins
        for v in pathgraph:
            # v in pathgraph returns all nodes in the pathgraph
            if v not in seen:
                # pathgraph[v] returns a path containing v
                yield pathgraph[v]
            seen.update(pathgraph[v])

    def get_assembly_length(self):
        """
        Computes the length of the assembly.

        Returns
        -------
        tuple: (assembly length, number of paths)

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.get_assembly_length()
        (60, 3)
        """
        assembly_length = 0
        paths_total = 0
        for path in self.scaffold.get_all_paths():
            paths_total += 1
            length = sum([self.scaffold.node[x]['length'] for x in path])
            assembly_length += length

        return assembly_length, paths_total

    def remove_small_paths(self, min_length, split_scaffolds=False):
        """
        Removes from HiC matrix all bins that are smaller than certain size.

        Parameters
        ----------
        min_length : minimum path length in bp

        Returns
        -------
        None

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4], [5]]
        >>> list(S.scaffold.get_all_paths())
        [['c-3'], ['c-2'], ['c-0']]

        >>> S.remove_small_paths(20)

        The paths that are smaller or equal to 20 are c-2 and c-3.
        thus, only the path of 'c-0' is kept
        >>> S.matrix_bins.path.values()
        [[0, 1, 2]]

        >>> list(S.scaffold.get_all_paths())
        [['c-0']]

        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3'], ['c-2']]

        >>> list(S.removed_bins.get_all_paths())
        [[3, 4], [5]]
        >>> S.removed_bins.node[5]
        {'end': 10, 'name': 'c-3', 'start': 0, 'length': 10, 'coverage': 1}

        Test removal of bins and scaffold when two scaffolds are already merged
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 50, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.add_edge(4, 5)
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> S.remove_small_paths(30)
        >>> list(S.scaffold.get_all_paths())
        [['c-0']]
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-2', 'c-3']]

        Test split_scaffolds
        >>> S.restore_scaffold('c-2')
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> S.remove_small_paths(30, split_scaffolds=True)
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3'], ['c-2']]
        """

        to_remove = []
        to_remove_paths = []
        paths_total = 0
        removed_length_total = 0
        paths_list = list(self.matrix_bins.get_all_paths())
        for path in paths_list:
            paths_total += 1
            length = (sum([self.matrix_bins.node[x]['length'] for x in path]))

            if length <= min_length:
                log.debug("Removing path {}, length {}".format(self.matrix_bins.get_path_name_of_node(x), length))
                self._remove_bin_path(path, split_scaffolds=split_scaffolds)
                to_remove.extend(path)
                to_remove_paths.append(path)
                removed_length_total += length

        if len(to_remove) and len(to_remove) < self.matrix.shape[0]:
            log.debug("Removing {num_scaffolds} scaffolds/contigs, containing {num_bins} bins "
                      "({fraction:.3f}% of total assembly length), because they "
                      "are shorter than {min_length} ".format(num_scaffolds=len(to_remove_paths),
                                                              num_bins=len(to_remove),
                                                              fraction=100 * float(removed_length_total) / self.total_length,
                                                              min_length=min_length))

    def _remove_bin_path(self, path, split_scaffolds=False):
        scaffold_id = self.matrix_bins.node[path[0]]['name']

        if path[0] in self.matrix_bins.path_id:
            path_name = self.matrix_bins.path_id[path[0]]
            self.removed_bins.add_path(path, name=path_name)
            self.removed_scaffolds.add_path(self.scaffold[scaffold_id])

        for bin_node in path:
            self.removed_bins.add_node(bin_node, **self.matrix_bins.node[bin_node])
            self.removed_bins.add_node(bin_node, **self.matrix_bins.node[bin_node])
        for scaffold_name in self.scaffold[scaffold_id]:
            self.removed_scaffolds.add_node(scaffold_name, **self.scaffold.node[scaffold_name])
        # remove scaffold links
        if split_scaffolds is True and len(self.scaffold[scaffold_id]) > 1:
            scaff_path = self.scaffold[scaffold_id]
            for scaff_u, scaff_v in zip(scaff_path[:-1], scaff_path[1:]):
                path_u = self.scaffold.node[scaff_u]['path']
                path_v = self.scaffold.node[scaff_v]['path']

                self.removed_bins.delete_edge(path_u[-1], path_v[0])
                self.removed_scaffolds.delete_edge(scaff_u, scaff_v)

        # remove matrix_bins path
        self.matrix_bins.delete_path_containing_node(path[0], delete_nodes=True)
        # remove scaffolds path
        self.scaffold.delete_path_containing_node(scaffold_id, delete_nodes=True)

    def restore_scaffold(self, scaffold_name):
        """

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.remove_small_paths(20)
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3'], ['c-2']]
        >>> S.restore_scaffold('c-2')
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3']]
        >>> list(S.scaffold.get_all_paths())
        [['c-2'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4]]
        >>> list(S.removed_bins.get_all_paths())
        [[5]]

        Test restore when the removed scaffold is already joined to other scaffold
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 50, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.add_edge(4, 5)
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4, 5]]
        >>> S.remove_small_paths(30)
        >>> S.restore_scaffold('c-2')
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4, 5]]
        """
        scaff_path = self.removed_scaffolds[scaffold_name]
        if scaffold_name in self.removed_scaffolds.path_id:
            path_id = self.removed_scaffolds.path_id[scaffold_name]
            self.scaffold.add_path(scaff_path, name=path_id)
        bin_0 = self.removed_scaffolds.node[scaffold_name]['path'][0]
        if bin_0 in self.removed_bins.path_id:
            matrix_bin_path = self.removed_bins[bin_0]
            matrix_path_id = self.removed_bins.path_id[bin_0]
            self.matrix_bins.add_path(matrix_bin_path, name=matrix_path_id)
        for scaff_name in scaff_path:
            self.scaffold.add_node(scaff_name, **self.removed_scaffolds.node[scaff_name])
            path = self.removed_scaffolds.node[scaff_name]['path']
            for bin_node in path:
                self.matrix_bins.add_node(bin_node, **self.removed_bins.node[bin_node])

        self.removed_bins.delete_path_containing_node(path[0], delete_nodes=True)
        self.removed_scaffolds.delete_path_containing_node(scaffold_name, delete_nodes=True)
    def restore_scaffold(self, scaffold_name):
        """

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.remove_small_paths(20)
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3'], ['c-2']]
        >>> S.restore_scaffold('c-2')
        >>> list(S.removed_scaffolds.get_all_paths())
        [['c-3']]
        >>> list(S.scaffold.get_all_paths())
        [['c-2'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4]]
        >>> list(S.removed_bins.get_all_paths())
        [[5]]

        Test restore when the removed scaffold is already joined to other scaffold
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 50, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.add_edge(4, 5)
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4, 5]]
        >>> S.remove_small_paths(30)
        >>> S.restore_scaffold('c-2')
        >>> list(S.scaffold.get_all_paths())
        [['c-2', 'c-3'], ['c-0']]
        >>> list(S.matrix_bins.get_all_paths())
        [[0, 1, 2], [3, 4, 5]]
        """
        scaff_path = self.removed_scaffolds[scaffold_name]
        if scaffold_name in self.removed_scaffolds.path_id:
            path_id = self.removed_scaffolds.path_id[scaffold_name]
            self.scaffold.add_path(scaff_path, name=path_id)
        bin_0 = self.removed_scaffolds.node[scaffold_name]['path'][0]
        if bin_0 in self.removed_bins.path_id:
            matrix_bin_path = self.removed_bins[bin_0]
            matrix_path_id = self.removed_bins.path_id[bin_0]
            self.matrix_bins.add_path(matrix_bin_path, name=matrix_path_id)
        for scaff_name in scaff_path:
            self.scaffold.add_node(scaff_name, **self.removed_scaffolds.node[scaff_name])
            path = self.removed_scaffolds.node[scaff_name]['path']
            for bin_node in path:
                self.matrix_bins.add_node(bin_node, **self.removed_bins.node[bin_node])

        self.removed_bins.delete_path_containing_node(path[0], delete_nodes=True)
        self.removed_scaffolds.delete_path_containing_node(scaffold_name, delete_nodes=True)

    def remove_small_paths_bk(self, min_length):
        """
        Removes from HiC matrix all bins that are smaller than certain size.

        1. the paths that need to be removed are identified.
        2. all the bins belonging to the paths to be removed are eliminated
           from the matrix
        3. The remaining paths are relabeled to match the new indices of the matrix

        Parameters
        ----------
        min_length : minimum path length in bp

        Returns
        -------
        None

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> [x for x in S.get_all_paths()]
        [[0, 1, 2], [3, 4], [5]]
        >>> S.remove_small_paths_bk(20)

        The paths that are smaller or equal to 20 are the one corresponding to c-2 and c-3.
        thus, only the path of 'c-0' is kept
        >>> list(S.get_all_paths())
        [[0, 1, 2]]

        The matrix is reduced
        >>> S.matrix.todense()
        matrix([[ 2,  8,  5],
                [ 8,  8, 15],
                [ 5, 15,  0]])
        """
        to_remove = []
        to_remove_paths = []

        paths_total = 0
        length_total = 0
        removed_length_total = 0

        for path in self.get_all_paths():
            paths_total += 1
            length = (sum([self.matrix_bins.node[x]['length'] for x in path]))
            length_total += length

            if length <= min_length:
                log.debug("Removing path {}, length {}".format(self.matrix_bins.get_path_name_of_node(x), length))
                to_remove.extend(path)
                to_remove_paths.append(path)
                removed_length_total += length

        if len(to_remove) and len(to_remove) < self.matrix.shape[0]:
            log.debug("Removing {num_scaffolds} scaffolds/contigs, containing {num_bins} bins "
                      "({fraction:.3f}% of total assembly length), because they "
                      "are shorter than {min_length} ".format(num_scaffolds=len(to_remove_paths),
                                                              num_bins=len(to_remove),
                                                              fraction=100 * float(removed_length_total) / length_total,
                                                              min_length=min_length))

            self.hic.removeBins(to_remove)
            self._init_path_graph()

    def get_paths_length(self):
        for path in self.get_all_paths():
            yield (sum([self.matrix_bins.node[x]['length'] for x in path]))

    def get_paths_stats(self):
        import matplotlib.pyplot as plt
        paths_length = np.fromiter(self.get_paths_length(), int)
        if len(paths_length) > 10:
            plt.hist(paths_length, 100)
        # TODO clear debug code that generates images
        file_name = "{}/stats_len_{}.pdf".format(self.out_folder, len(paths_length))
        log.debug("Saving histogram {} ".format(file_name))
        plt.savefig(file_name)
        plt.close()

        self.paths_len = len(paths_length)
        self.paths_min = paths_length.min()
        self.paths_max = paths_length.max()
        self.paths_mean = np.mean(paths_length)
        self.paths_median = np.median(paths_length)
        self.paths_p25 = np.percentile(paths_length, 25)
        self.paths_p75 = np.percentile(paths_length, 75)
        log.info("len:\t{:,}".format(self.paths_len))
        log.info("max:\t{:,}".format(self.paths_max))
        log.info("min:\t{:,}".format(self.paths_min))
        log.info("mean:\t{:,}".format(self.paths_mean))
        log.info("median:\t{:,}".format(self.paths_median))
        log.info("25th percentile:\t{:,}".format(self.paths_p25))
        log.info("75th percentile:\t{:,}".format(self.paths_p75))

    def compute_N50(self, min_length=200):
        """
        Computes the N50 based on the existing paths.

        Parameters
        ----------
        min_length : paths with a length smaller than this will be skiped

        Returns
        -------
        int : length of the N50 contig

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-2', 0, 30, 1),
        ... ('c-3', 0, 10, 1), ('c-3', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)

        The lengths for the paths in this matrix are:
        [20 30 10 10 10]

        The sorted cumulative sum is:
        [10, 20, 30, 50, 80]
        >>> S.compute_N50(min_length=2)
        30

        """
        length = np.sort(np.fromiter(self.get_paths_length(), int))
        if len(length) == 0:
            raise ScaffoldException ("No paths. Can't compute N50")
        length = length[length > min_length]
        if len(length) == 0:
            raise ScaffoldException ("No paths with length > {}. Can't compute N50".format(min_length))
        cumsum = np.cumsum(length)

        # find the index at which the cumsum length is half the total length
        half_length = float(cumsum[-1]) / 2
        for i in range(len(length)):
            if cumsum[i] >= half_length:
                break

        return length[i]

    def split_and_merge_contigs(self, num_splits=3, target_size=None,  normalize_method=['mean', 'ice', 'none'][0]):
        """
        Splits each contig/scaffold into `num_splits` parts and creates
        a new reduced matrix with the slitted paths. The merge data is kept
        in the pg_base PathGraph object

        Parameters
        ----------
        num_splits number of parts in which the contig is split.
        target_size overrides num_splits. Instead, the num_splits is computed based on the target size.
        normalize_method after the contigs are split, the individual bins that constitute each split are merged and
                        subsequently normalized using the given method. Default is 'mean'

        Returns
        -------
        None


        Examples
        -------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        >>> A = csr_matrix(np.array([[2,2,1,1,1,1],[2,2,1,1,1,1],
        ... [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]]))

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)
        >>> S = Scaffolds(hic)
        >>> S.matrix.todense()
        matrix([[4, 4, 2, 2, 2, 2],
                [4, 4, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2]])

        >>> S.matrix_bins.path == {'c-0': [0, 1, 2, 3, 4, 5]}
        True
        >>> S.split_and_merge_contigs(num_splits=3, normalize_method='none')
        >>> S.matrix.todense()
        matrix([[12,  8,  8],
                [ 8,  6,  8],
                [ 8,  8,  6]])

        Now the 'c-0' contig path is shorter
        >>> S.pg_base.path
        {'c-0': [0, 1, 2]}
        >>> len(S.pg_base.node)
        3
        >>> S.pg_base.node
        {0: {'initial_path': [0, 1], 'length': 20, 'name': 'c-0_0'}, \
1: {'initial_path': [2, 3], 'length': 20, 'name': 'c-0_1'}, \
2: {'initial_path': [4, 5], 'length': 20, 'name': 'c-0_2'}}

        Same matrix as before, but this time normalized by mean
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(num_splits=3, normalize_method='mean')
        >>> S.matrix.todense()
        matrix([[ 3. ,  2. ,  2. ],
                [ 2. ,  1.5,  2. ],
                [ 2. ,  2. ,  1.5]])

        Same matrix as before, but this time normalized by ice
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(num_splits=3, normalize_method='ice')
        >>> S.matrix.todense()
        matrix([[ 8.97254   ,  7.50474504,  7.50474504],
                [ 7.50474504,  7.06169578,  9.41559437],
                [ 7.50474504,  9.41559437,  7.06169578]])

        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(num_splits=1, normalize_method='none')
        >>> S.matrix.todense()
        matrix([[48]])

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 0, 10, 1), ('c-2', 10, 20, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)
        >>> S = Scaffolds(hic)
        >>> S.matrix.todense()
        matrix([[4, 4, 2, 2, 2, 2],
                [4, 4, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2]])
        >>> S.matrix_bins.path # == {'c-0': [0, 1, 2, 3, 4, 5]}
        {'c-2': [4, 5], 'c-1': [2, 3], 'c-0': [0, 1]}
        >>> S.split_and_merge_contigs(num_splits=1, normalize_method='none')

        >>> S.pg_base.node
        {0: {'initial_path': [0, 1], 'length': 20, 'name': 'c-0'}, \
1: {'initial_path': [2, 3], 'length': 20, 'name': 'c-1'}, \
2: {'initial_path': [4, 5], 'length': 20, 'name': 'c-2'}}

        >>> S.matrix_bins.node[0]['merged_path_id']
        0
        >>> S.matrix_bins.node[5]['merged_path_id']
        2
        >>> S.scaffold.node['c-1']['merged_path_id']
        1
        >>> S.matrix.todense()
        matrix([[12,  8,  8],
                [ 8,  6,  8],
                [ 8,  8,  6]])

        # do a second round of split_and_merge after joining some contigs
        >>> S.add_edge(0, 1)
        >>> S.add_edge(1, 2)
        >>> S.split_and_merge_contigs(num_splits=1, normalize_method='none')
        >>> S.matrix.todense()
        matrix([[48]])

        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(num_splits=1, normalize_method='mean')
        >>> S.matrix.todense()
        matrix([[ 3. ,  2. ,  2. ],
                [ 2. ,  1.5,  2. ],
                [ 2. ,  2. ,  1.5]])

        Now there are not paths.
        >>> S.pg_base.path
        {'c-2': [2], 'c-1': [1], 'c-0': [0]}
        """

        paths_flatten = []
        i = 0
        self.pg_base = PathGraph()
        for path in self.get_all_paths():
            if target_size is not None:
                # define the number of splits based on the target size
                length = sum([self.matrix_bins.node[x]['length'] for x in path])
                if target_size <= length:
                    num_splits = length / target_size
                else:
                    log.debug("path is too small ({:,}) to split for target size {:,}".format(length, target_size))
                    num_splits = 1
            # split_path has the form [[0, 1 ,2], [3, 4], [5, 6, 7]]
            split_path = Scaffolds.split_path(path, num_splits)
            # each sub path in the split_path list will become an index
            # in a new matrix after merging and correcting. To
            # keep track of the original path that give rise to the
            # bin, a new PathGraph node is created, having as id the future merged matrix
            # bin id, and having as attributes length and base path.

            # For example, for a split_path list e.g. [[0,1], [2,3]]]
            # after merging (that is matrix merging of the respective bins e.g 0 and 1)
            # the [0,1] becomes bin [0] and [2,3] becomes bin 1. Thus, a PathGraph node with
            # id 0 is created, having as length the sum of 0 and 1, and having as base_pat [0,1]

            # if reset_base_paths is True, then the merged paths become the
            # base paths. In this case node information contains extra values

            # reset_base_paths is used for the first merge_to_size when is irrelevant to
            # keep the original high resolution paths that form a contig. After this
            # merge, it is important to keep track of the original or base paths.
            merged_path = []
            path_name = self.matrix_bins.get_path_name_of_node(path[0])
            for index, sub_path in enumerate(split_path):
                length = sum([self.matrix_bins.node[x]['length'] for x in sub_path])

                # prepare new PathGraph nodes
                if num_splits == 1:
                    # by default the type of path_name is numpy.string which is not compatible with networkx when
                    # saving graphml
                    name = str(path_name)
                else:
                    name = "{}_{}".format(path_name, index)
                attr = {'length': length,
                        'name': name,
                        'initial_path': sub_path}
                scaffold_name_set = set()
                # update matrix_bin nodes to refer to the new merged path id
                for bin_id in sub_path:
                    self.matrix_bins.node[bin_id]['merged_path_id'] = i
                    scaffold_name_set.add(self.matrix_bins.node[bin_id]['name'])
                # update self.scaffold nodes to refer to the new merged path id
                for scaff_name in scaffold_name_set:
                    self.scaffold.node[scaff_name]['merged_path_id'] = i

                self.pg_base.add_node(i, attr_dict=attr)

                merged_path.append(i)
                i += 1

            self.pg_base.add_path(merged_path, name=path_name)
            paths_flatten.extend(split_path)

        if len(paths_flatten) == 0:
            log.warn("Nothing to reduce.")
            return None

        reduce_paths = paths_flatten[:]

        reduced_matrix = reduce_matrix(self.hic.matrix, reduce_paths, diagonal=True)

        if normalize_method == 'mean':
            self.matrix = Scaffolds.normalize_by_mean(reduced_matrix, reduce_paths)
        elif normalize_method == 'ice':
            self.matrix = Scaffolds.normalize_by_ice(reduced_matrix, reduce_paths)
        else:
            self.matrix = reduced_matrix

        assert len(self.pg_base.node.keys()) == self.matrix.shape[0], "inconsistency error"

    @staticmethod
    def normalize_by_mean(matrix, paths):

        matrix = matrix.tocoo()
        paths_len = [len(x) for x in paths]
        # compute mean values for reduce matrix
        new_data = np.zeros(len(matrix.data))
        for index, value in enumerate(matrix.data):
            row_len = paths_len[matrix.row[index]]
            col_len = paths_len[matrix.col[index]]
            new_data[index] = float(value) / (row_len * col_len)

        matrix.data = new_data
        return matrix.tocsr()

    @staticmethod
    def normalize_by_ice(matrix, paths):
        return iterativeCorrection(matrix, M=1000, verbose=False)[0]

#     @logit
#     def merge_to_size(self, target_length=20000, reset_base_paths=True, normalize_method='ice'):
#         """
#         finds groups of bins/node that have a sum length of about the `target_length` size.
#         The algorithm proceeds from the flanks of a path to the inside. If a bin/node
#         is too small it is skipped.
#
#
#         Parameters
#         ----------
#         target_length : in bp
#         reset_base_paths: boolean   Reset means that the merge information is kept
#                                     as the primary data, otherwise the original paths
#                                     and node data is kept and the merge refers to this data. Why is this done?
#                                     For contigs with dpnII based bins, the original paths can be hundreds of
#                                     bins long but this detailed information is not needed. Thus, merging by
#                                     size only the id of the contig and the shorter (merged) path is kept.
#                                     However, subsequent merges by size of scaffolds (union of contigs) need
#                                     to refer to the original contigs and all information should be kept. This
#                                     is achieved by using two PathGraph objects, one that holds the current
#                                     paths based on the current matrix bin ids and other PathGraph object
#                                     that holds the original contig bins.
#
#         Returns
#         -------
#
#         Examples
#         --------
#         >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
#         ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
#         >>> hic = get_test_matrix(cut_intervals=cut_intervals)
#         >>> S = Scaffolds(hic)
#         >>> S.matrix_bins.path == {'c-0': [0, 1, 2, 3, 4, 5]}
#         True
#         >>> S.merge_to_size(target_length=20, reset_base_paths=True)
#         >>> S.matrix.todense()
#         matrix([[ 11.26259299,  21.9378206 ,  17.42795074],
#                 [ 21.9378206 ,   6.86756935,  21.82307214],
#                 [ 17.42795074,  21.82307214,  11.37726956]])
#
#         Now the 'c-0' contig path is shorter
#         >>> S.pg_base.path == {'c-0': [0, 1, 2]}
#         True
#         >>> len(S.pg_base.node)
#         3
#         >>> S.pg_base.node
#         {0: {'start': 0, 'length': 20, 'end': 20, 'name': 'c-0', 'coverage': 1.5}, \
# 1: {'start': 20, 'length': 20, 'end': 40, 'name': 'c-0', 'coverage': 1.0}, \
# 2: {'start': 40, 'length': 20, 'end': 60, 'name': 'c-0', 'coverage': 1.0}}
#
#
#         """
#         log.info("merge_to_size. flank_length: {:,}".format(target_length))
#
#         # flattened list of merged_paths  e.g [[1,2],[3,4],[5,6],[7,8]].
#         # This is in contrast to a list containing split_path_envenly that may
#         # look like [ [[1,2],[3,4]], [[5,6]] ]
#         paths_flatten = []
#         log.debug('value of reset paths is {}'.format(reset_base_paths))
#         i = 0
#
#         pg_merge = PathGraph()
#
#         for path in self.get_all_paths():
#             # split_path_envenly has the form [[0, 1], [2, 3], [4, 5]]
#             # where each sub-list (e.g. [0,1]) has more or less the same size in bp
#             split_path = self.split_path_envenly(path, target_length, 100)
#
#             # each sub path in the split_path_envenly list will become an index
#             # in a new matrix after merging and correcting. To
#             # keep track of the original path that give rise to the
#             # bin, a new PathGraph node is created, having as id the future merged matrix
#             # bin id, and having as attributes length and base path.
#
#             # For example, for a split_path_envenly list e.g. [[0,1], [2,3]]]
#             # after merging (that is matrix merging of the respective bins e.g 0 and 1)
#             # the [0,1] becomes bin [0] and [2,3] becomes bin 1. Thus, a PathGraph node with
#             # id 0 is created, having as length the sum of 0 and 1, and having as base_pat [0,1]
#
#             # if reset_base_paths is True, then the merged paths become the
#             # base paths. In this case node information contains extra values
#
#             # reset_base_paths is used for the first merge_to_size when is irrelevant to
#             # keep the original high resolution paths that form a contig. After this
#             # merge, it is important to keep track of the original or base paths.
#
#             merged_path = []
#             path_name = self.pg_base.get_path_name_of_node(path[0])
#             for index, sub_path in enumerate(split_path):
#                 # prepare new PathGraph nodes
#                 attr = {'length': sum([self.pg_base.node[x]['length'] for x in sub_path]),
#                         'name': "{}/{}".format(path_name, index),
#                         'initial_path': sub_path}
#
#                 pg_merge.add_node(i, attr_dict=attr)
#
#                 merged_path.append(i)
#
#                 i += 1
#
#             pg_merge.add_path(merged_path, name=path_name)
#             paths_flatten.extend(split_path)
#
#         if len(paths_flatten) == 0:
#             log.warn("Nothing to reduce.")
#             return None
#
#         reduce_paths = paths_flatten[:]
#
#         if len(reduce_paths) < 2:
#             log.info("Reduce paths to small {}. Returning".format(len(reduce_paths)))
#             return None
#
#         reduced_matrix = reduce_matrix(self.matrix, reduce_paths, diagonal=True)
#
#         # correct reduced_matrix
#         start_time = time.time()
#         corrected_matrix = iterativeCorrection(reduced_matrix, M=1000, verbose=False)[0]
#         elapsed_time = time.time() - start_time
#         log.debug("time iterative_correction: {:.5f}".format(elapsed_time))
#
#         # update matrix
#         self.matrix = corrected_matrix
#
#         self.pg_base = pg_merge
#
#         if reset_base_paths is True:
#             self.reset_matrix_bins()
#
#         assert len(self.pg_base.node.keys()) == self.matrix.shape[0], "inconsistency error"

    def reset_matrix_bins(self):
        """
        This is a function called to reduce the information stored from original paths
        after merge_to_size is called for the first time.

        When contigs contains a lot of bins (e.g. when DpnII restriction enzyme was used)
        after the initial merge_to_size, there is no need to keep the detailed
        information of all the bins.

        Returns
        -------

        """

        if self.matrix_bins is None:
            log.debug("can't reset. matrix_bins is not set")
            return

        pg_base = PathGraph()
        # this variable is to keep the hic object matrix in sync
        cut_intervals = []
        for path in self.get_all_paths():
            path_name = self.pg_base.get_path_name_of_node(path[0])
            for node in path:
                base_attr = self.pg_base.node[node]
                initial_path = base_attr['initial_path']
                # prepare replacement for initial small nodes
                first_node = initial_path[0]
                last_node = initial_path[-1]
                attr = {'name': self.matrix_bins.node[first_node]['name'],
                        'start': self.matrix_bins.node[first_node]['start'],
                        'end': self.matrix_bins.node[last_node]['end'],
                        'coverage': float(self.matrix_bins.node[first_node]['coverage'] +
                                          self.matrix_bins.node[last_node]['coverage']) / 2}
                assert attr['start'] < attr['end']
                attr['length'] = attr['end'] - attr['start']

                base_attr['initial_path'] = [node]
                self.pg_base.add_node(node, attr_dict=base_attr)
                pg_base.add_node(node, attr_dict=attr)
                cut_intervals.append((attr['name'], attr['start'], attr['end'], attr['coverage']))
            pg_base.add_path(path, name=path_name)

        self.pg_base = pg_base
        self.matrix_bins = None

        # reset the hic matrix object
        self.hic.matrix = self.matrix
        self.hic.setCutIntervals(cut_intervals)

    @staticmethod
    def split_path(path, num_parts):
        """
        Splits a path into `num_parts`. For example a path of length 100, can be subdivided
        into three shorted paths of about the same number of members.

        Parameters
        ----------
        path : list of ids
        num_parts : number of parts to divide the path

        Returns
        -------

        Examples
        --------

        >>> Scaffolds.split_path([1,2,3,4,5], 2)
        [[1, 2, 3], [4, 5]]

        >>> Scaffolds.split_path([1,2,3,4,5,6,7,8,9,10], 4)
        [[1, 2, 3], [4, 5], [6, 7], [8, 9, 10]]
        >>> Scaffolds.split_path([1,2,3,4,5,6], 3)
        [[1, 2], [3, 4], [5, 6]]
         """
        def split_number(whole, parts):
            """
            Splits the integer (whole) into more or less equal parts.
            An array is returned such that the longest parts are on the start and
            end of the array.

            Parameters
            ----------
            whole : integer to be divided
            parts : numbeer of parts

            Returns
            -------
            numpy array

            >>> split_number(20, 7)
            array([3, 3, 3, 2, 3, 3, 3])
            >>> split_number(20, 3)
            array([7, 6, 7])
            """
            arr = np.zeros(parts, dtype=int)
            remain = whole
            parts_left = parts
            count = 0
            for i in range(parts):
                size = (remain + parts_left - 1) / parts_left
                if i % 2 == 0:
                    arr[count] = int(size)
                    count += 1
                else:
                    arr[-count] = int(size)
                remain -= size
                parts_left -= 1
            return arr

        divided_path = []

        if len(path) == 1:
            log.debug("Can't subdivide path of length = 1")
            divided_path = [path]
        elif len(path) < num_parts:
            log.debug("Can't subdivide path of length = {} into {} parts. Dividing into {} "
                     "parts".format(len(path), num_parts, len(path)))
            divided_path = [[x] for x in path]
        else:
            start_index = 0
            for sub_path_len in split_number(len(path), num_parts):
                divided_path.append(path[start_index:start_index+sub_path_len])
                start_index += sub_path_len

        return divided_path

    def split_path_envenly(self, path, flank_length, recursive_repetitions, counter=0):
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
        # make one matrix whith only one split contig c-0 whose
        # bins have all 10 bp
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> flank_length = 20
        >>> S.split_path_envenly(S.matrix_bins[0], flank_length, 30)
        [[0, 1], [2, 3], [4, 5]]

        # 5 bins, one internal is smaller than 20*75 (flank_length * tolerance) is skipped
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> flank_length = 20
        >>> S.split_path_envenly(S.matrix_bins[0], flank_length, 30)
        [[0, 1], [3, 4]]

        Get the flanks, and do not recursively iterate
        # 5 bins, one internal is smaller than 20*75 (flank_length * tolerance) is skipped
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> flank_length = 10
        >>> S.split_path_envenly(S.matrix_bins[0], flank_length, 1)
        [[0], [4]]

        """
        counter += 1
        if counter > recursive_repetitions:
            return []

        tolerance_max = flank_length * 1.25
        tolerance_min = flank_length * 0.75
        path_length = dict([(x, self.matrix_bins.node[x]['length']) for x in path])
        path_length_sum = sum(path_length.values())
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
            for n in _path:
                flank_sum = sum(path_length[x] for n in flank)
                if flank_sum > tolerance_max:
                    break
                elif tolerance_min <= flank_sum <= tolerance_max:
                    break
                flank.append(n)
            return flank

        if len(path) == 1:
            if path_length[path[0]] > tolerance_min or counter == 1:
                flanks = [path]
        else:
            if path_length_sum < 2*flank_length*0.75:
                if counter == 1:
                    # if the total path length is shorter than twice the flank_length *.75
                    # then split the path into two
                    log.debug("path {} is being divided into two, although is quite small {}".format(path, path_length_sum))
                    path_half = len(path)/2
                    left_flank = path[0:path_half]
                    right_flank = path[path_half:]
                    flanks.extend([left_flank, right_flank])
                else:
                    flanks.extend([path])
                    log.debug("The path: {} is small ({}) and has no flanks".format(path, path_length_sum))
            else:
                left_flank = _get_path_flank(path)
                right_flank = _get_path_flank(path[::-1])[::-1]

                # check if the flanks overlap
                over = set(left_flank).intersection(right_flank)
                if len(over):
                    # remove overlap
                    left_flank = [x for x in left_flank if x not in over]

                if len(left_flank) == 0 or len(right_flank) == 0:
                    path_half = len(path)/2
                    left_flank = path[0:path_half]
                    right_flank = path[path_half:]

                interior = [x for x in path if x not in left_flank + right_flank]
                if len(interior):
                    interior = self.split_path_envenly(interior, flank_length, recursive_repetitions, counter=counter)
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

    def get_stats_per_distance(self):
        """
        takes the information from all bins that are split
        or merged and returns two values and two vectors. The
        values are the average length used and the sd.
        The vectors are: one containing the number of contacts found for such
        distance and the third one containing the normalized
        contact counts for different distances.
        The distances are 'bin' distance. Thus,
        if two bins are next to each other, they are at distance 1

        Returns
        -------
        mean bin length, std bin length, dict containing as key the bin distance
        and as values a dict with mean, median, max, min and len

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> mean, sd, stats = S.get_stats_per_distance()
        >>> stats[2]['mean']
        4.25

        """
        log.info("Computing stats per distance")

        # get all paths (connected components) longer than 1
        if len(self.pg_base.path) == 0:
            raise ScaffoldException ("Print no paths found\n")

        # use all paths to estimate distances
        dist_dict = dict()
        path_length = []
        for count, path in enumerate(self.pg_base.path.values()):
            if count > 200:
                # otherwise too many computations will be made
                # but 200 cases are enough to get
                # an idea of the distribution of values
                break
            for n in path:
                path_length.append(self.pg_base.node[n]['length'])

            # take upper triangle of matrix containing selected path
            sub_m = triu(self.matrix[path, :][:, path], k=1, format='coo')
            # find counts that are one bin apart, two bins apart etc.
            dist_list = sub_m.col - sub_m.row
            # tabulate all values that correspond
            # to distances
            for distance in np.unique(dist_list):
                if distance not in dist_dict:
                    dist_dict[distance] = sub_m.data[dist_list == distance]
                else:
                    dist_dict[distance] = np.hstack([dist_dict[distance],
                                                     sub_m.data[dist_list == distance]])

        # get mean and sd of the bin lengths
        mean_path_length = np.mean(path_length)
        sd_path_length = np.std(path_length)
        log.info("Mean path length: {:.1f} sd: {:.1f}".format(mean_path_length, sd_path_length))

        # consolidate data:
        consolidated_dist_value = dict()
        for k, v in dist_dict.iteritems():
            consolidated_dist_value[k] = {'mean': np.mean(v),
                                          'median': np.median(v),
                                          'max': np.max(v),
                                          'min': np.min(v),
                                          'len': len(v)}
            if len(v) < 10 and k < 15:
                log.warn('stats for distance {} contain only {} samples'.format(k, len(v)))
        return mean_path_length, sd_path_length, consolidated_dist_value

    def get_stats_per_split(self):
        """
        takes the information from all bins that are split
        or merged and returns the a dictionary whose key is the distance between split bins
        being 1, bins that are consecutive (or that the start positions are 1 bins apart),
        2 bins that are separated by one bin etc. The values of the dictionary are itself a dictionary
        whose keys are: mean, median, max, min and len and reflect the mean, median etc number of contacts. The
        len is the number of samples that were used to derive the information.

        Returns
        -------
        dictionary as explained previously

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> stats = S.get_stats_per_split()
        >>> stats[2]['mean']
        4.25

        >>> stats.keys()
        [1, 2, 3, 4, 5]

        >>> stats[1]
        {'max': 15, 'min': 1, 'median': 7.0, 'len': 5, 'mean': 7.4000000000000004}

        """
        log.info("Computing stats per distance")

        # get all paths (connected components) longer than 1
        if len(self.pg_base.path) == 0:
            raise ScaffoldException ("Print no paths found\n")

        # use all paths to estimate distances
        dist_dict = dict()
        for count, path in enumerate(self.pg_base.path.values()):
            if count > 200:
                # otherwise too many computations will be made
                # but 200 cases are enough to get
                # an idea of the distribution of values
                break

            # take upper triangle of matrix containing selected path
            sub_m = triu(self.matrix[path, :][:, path], k=1, format='coo')
            # find counts that are one bin apart, two bins apart etc.
            dist_list = sub_m.col - sub_m.row
            # tabulate all values that correspond
            # to distances
            for distance in np.unique(dist_list):
                if distance not in dist_dict:
                    dist_dict[distance] = sub_m.data[dist_list == distance]
                else:
                    dist_dict[distance] = np.hstack([dist_dict[distance], sub_m.data[dist_list == distance]])

        # consolidate data:
        consolidated_dist_value = dict()
        for k, v in dist_dict.iteritems():
            consolidated_dist_value[k] = {'mean': np.mean(v),
                                          'median': np.median(v),
                                          'max': np.max(v),
                                          'min': np.min(v),
                                          'len': len(v)}
            if len(v) < 10 and k < 10:
                log.warn('stats for distance {} contain only {} samples'.format(k, len(v)))
        return consolidated_dist_value

    @staticmethod
    def find_best_permutation(ma, paths, return_all_sorted_best_paths=False, list_of_permutations=None,
                              only_expand_but_not_permute=False):
        """
        Computes de bandwidth(bw) for all permutations of rows (and, because
        the matrix is symmetric of cols as well).
        Returns the permutation having the minimum bw.

        The fixed pairs are not separated when considering
        the permutations to test

        Parameters
        ----------
        ma: HiCMatrix object
        paths: list of paths, containing paths that should not be reorder
        only_expand_but_not_permute : if false, a permutation of the path is done and a expansion that
                                      flips each path direction is also carried out. Setting to false, only
                                      the flipping is done but not the permutation
        Returns
        -------
        path

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix(np.array(
        ... [[12,5,3,2,0],
        ...  [0,11,4,1,1],
        ...  [0,0,9,6,0],
        ...  [0,0,0,10,0],
        ...  [0,0,0,0,0]]))
        >>> Scaffolds.find_best_permutation(A, [[2], [3,4]])
        [[2], [3, 4]]
        >>> Scaffolds.find_best_permutation(A, [[2],[3],[4]])
        [[3], [2], [4]]
        """
        indices = sum(paths, [])
        ma = ma[indices, :][:, indices]
        ma.setdiag(0)
        # mapping from 'indices' to new matrix id
        mapping = dict([(val, idx) for idx, val in enumerate(indices)])
        bw_value = []
        perm_list = []
        if list_of_permutations is None:
            if only_expand_but_not_permute:
                for expnd in Scaffolds.permute_paths(paths):
                    if expnd[::-1] in perm_list:
                        continue
                    expand_indices = sum(expnd, [])
                    mapped_perm = [mapping[x] for x in expand_indices]
                    bw_value.append(Scaffolds.bw(ma[mapped_perm, :][:, mapped_perm]))
                    perm_list.append(expnd)
            else:
                for perm in itertools.permutations(paths):
                    for expnd in Scaffolds.permute_paths(perm):
                        if expnd[::-1] in perm_list:
                            continue
                        expand_indices = sum(expnd, [])
                        mapped_perm = [mapping[x] for x in expand_indices]
                        bw_value.append(Scaffolds.bw(ma[mapped_perm, :][:, mapped_perm]))
                        perm_list.append(expnd)

        else:
            for expnd in list_of_permutations:
                if expnd[::-1] in perm_list:
                    continue
                expand_indices = sum(expnd, [])
                mapped_perm = [mapping[x] for x in expand_indices]
                bw_value.append(Scaffolds.bw(ma[mapped_perm, :][:, mapped_perm]))
                perm_list.append(expnd)

        min_val = min(bw_value)
        min_indx = bw_value.index(min_val)

        if return_all_sorted_best_paths is True:
            order = np.argsort(bw_value)
            return [(perm_list[x], bw_value[x]) for x in order]
        return perm_list[min_indx]

    @staticmethod
    def find_best_direction(ma, paths):
        """
        For a list of paths, uses the bandwidth measurement to identify the direction of the paths
        by flipping each path and evaluating if the bandwidth decreases.

        Parameters
        ----------
        ma: HiCMatrix object
        paths: list of paths, containing paths that should not be reorder

        Returns
        -------
        path : path with the lowest bw

        Examples
        --------
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

        >>> A = A + A.T
        >>> Scaffolds.find_best_direction(A, [[2, 1, 0], [3, 4, 5], [9, 8, 7, 6]])
        [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        >>> Scaffolds.find_best_direction(A, [[2, 1, 0], [5, 4, 3], [9, 8, 7, 6]])
        [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

        >>> Scaffolds.find_best_direction(A, [[9, 8, 7, 6], [5, 4, 3], [2, 1, 0] ])
        [[9, 8, 7, 6], [5, 4, 3], [2, 1, 0]]
        """
        indices = sum(paths, [])
        ma = ma[indices, :][:, indices]
        ma.setdiag(0)
        # mapping from 'indices' to new matrix id
        mapping = dict([(val, idx) for idx, val in enumerate(indices)])
        min_value = np.Inf
        best_path = paths[:]
        seen = set()
        # the algorithm takes each part of the path (sub_path),
        # and computes the bw without flipping and with flipping.
        # Then takes the direction that produces the smallest bw and
        # continues with the next part of the path.
        # This is faster than testing all possible combinations or orientations
        # For a path with three sub_paths, the number of possible combinations is
        # 8 (+++, ++-, +-+, +--, -++, -+-, --+, ---) while for this
        # algorithm then number of combinations is at most 6. In general, for the
        # exhaustive search the combinations are 2 ** len(path), while for this
        # algorithm the number of combinations are <= 2 * len(path)
        for idx, sub_path in enumerate(paths):
            if len(sub_path) == 1:
                continue
            for orientation in ['+', '-']:
                sub_path_oriented = sub_path[:] if orientation == '+' else sub_path[::-1]
                # best_path stores in each iteration the best orientation for the evaluated
                # sub_path
                path_to_test = best_path[:idx] + [sub_path_oriented] + best_path[idx+1:]
                path_indices = sum(path_to_test, [])
                if tuple(path_indices) in seen:
                    continue
                seen.add(tuple(path_indices))
                mapped_perm = [mapping[x] for x in path_indices]
                bw_value = Scaffolds.bw(ma[mapped_perm, :][:, mapped_perm])
                if bw_value < min_value:
                    best_path = path_to_test
                    min_value = bw_value
                prev_path = path_to_test
        return best_path

    @staticmethod
    def bw(ma):
        """
        Computes my version of the bandwidth of the matrix
        which is defined as \sum_i\sum_{j=i} M(i,j)*(j-i)
        The matrix that minimizes this function should have
        higher values next to the main diagonal and
        decreasing values far from the main diagonal
        """
        ma = triu(ma, k=1, format='coo').astype(float)
        ma.data *= (ma.col - ma.row)
        return ma.sum()

    @staticmethod
    def permute_paths(paths):
        """
        Flips a path if it contains more than one element
        and returns all possible combinations

        Parameters
        ----------
        paths : list of paths

        >>> Scaffolds.permute_paths([[1, 2]])
        [[[1, 2]], [[2, 1]]]
        >>> Scaffolds.permute_paths([[1,2], [3]])
        [[[1, 2], [3]], [[2, 1], [3]]]
        >>> Scaffolds.permute_paths([[1], [2], [3,4], [5]])
        [[[1], [2], [3, 4], [5]], [[1], [2], [4, 3], [5]]]
        >>> Scaffolds.permute_paths([[1,2], [3,4], [5,6]])[0:3]
        [[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [6, 5]], [[1, 2], [4, 3], [5, 6]]]
        >>> Scaffolds.permute_paths([[1], [2], [3]])
        [[[1], [2], [3]]]
        """
        #paths = [[1, 2], [3]]
        combinations = []
        len_paths_longer_one = sum([1 for x in paths if len(x) > 1])

        # `itertools.product(*[(0, 1)]*len_paths_longer_one)`
        # generates all binary combinations of the given length
        # thus, itertools.product(*[(0, 1)] *2:
        # [(0, 0), (0, 1), (1, 0), (1, 1)]
        #
        # itertools.product(*[(0, 1)] *3:
        # [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), .... ]

        for prod in itertools.product(*[(0, 1)]*len_paths_longer_one):
            combination = []
            bigger_path_pos = 0
            for path in paths:
                if len(path) == 1:
                    combination.append(path)
                else:
                    # decide if the path should be flipped
                    # using the prod tuple
                    combination.append(path if prod[bigger_path_pos] == 0 else path[::-1])
                    bigger_path_pos += 1

            combinations.append(combination)
        return combinations

    @logit
    def join_paths_max_span_tree(self, confidence_score,
                                 hub_solving_method=['remove weakest', 'bandwidth permute'][0],
                                 node_degree_threshold=None):
        """
        Uses the maximum spanning tree to identify paths to
        merge

        Parameters
        ---------
        confidence_score : Minimum contact threshold to consider. All other values are discarded
        hub_solving_method : Either 'remove weakest' or 'bandwidth permutation'

        Returns
        -------

        Examples
        --------

        The following matrix is used:

                      ______
                     |      |
                 a---b---c--d
                      \    /
                       --e--f

        The maximum spanning tree is:

                 a---b---c--d
                      \
                       --e--f

        After the computation of the maximum spaning tree, hubs are resolved and paths are merged

        >>> cut_intervals = [('a', 0, 1, 1), ('b', 1, 2, 1), ('c', 2, 3, 1),
        ... ('d', 0, 1, 1), ('e', 1, 2, 1), ('f', 0, 1, 1)]

                                0  1  2  3  4  5
                                a  b  c  d  e  f
        >>> matrix = np.array([[0, 3, 0, 0, 0, 0], # a
        ...                    [0, 0, 3, 2, 2.5, 0], # b
        ...                    [0, 0, 0, 3, 0, 0], # c
        ...                    [0, 0, 0, 0, 1, 0], # d
        ...                    [0, 0, 0, 0, 0, 3], # e
        ...                    [0, 0, 0, 0, 0, 0]])# f

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)
        >>> list(S.get_all_paths())
        [[0], [1], [2], [3], [4], [5]]

        by default the hub resolving algorithm prunes branches
        of length one that are connected to a hub.
        This only happens on the first iteration to remove
        single scaffolds that can be problematic.
        in the example, the node 'a' is a branch of length = 1
        and is removed from the graph
        >>> S.join_paths_max_span_tree(0, hub_solving_method='remove weakest', node_degree_threshold=5)
        >>> list(S.get_all_paths())
        [[5, 4, 1, 2, 3]]

        By changing the iteration to other value than 0,
        now the the weakest link for the hub 'b' (b-e) is removed
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)
        >>> S.iteration = 1
        >>> S.join_paths_max_span_tree(0, hub_solving_method='remove weakest', node_degree_threshold=5)
        >>> list(S.get_all_paths())
        [[3, 2, 1, 0], [5, 4]]
        
        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)

        By setting the node_degree threshold to 3 the
        'b' node (id 1) is skipped.
        Thus, the path [0, 1, 2, 3] can not be formed.
        but the path [2, 3, 4, 5] is formed
        >>> S.join_paths_max_span_tree(0, hub_solving_method='remove weakest',
        ...                            node_degree_threshold=4)
        >>> list(S.get_all_paths())
        [[0], [1], [5, 4, 3, 2]]

        Test bandwidth permutation
        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)
        >>> list(S.get_all_paths())
        [[0], [1], [2], [3], [4], [5]]

        >>> S.join_paths_max_span_tree(0, hub_solving_method='bandwidth permutation',
        ... node_degree_threshold=5)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3, 4, 5]]

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)
        >>> S.join_paths_max_span_tree(0, hub_solving_method='bandwidth permutation',
        ... node_degree_threshold=4)
        >>> list(S.get_all_paths())
        [[0], [1], [2, 3, 4, 5]]


        Links from nodes inside an established path are removed before computing the mst
        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(normalize_method=None)

        Add path joining a, b, and c. Before the computation of the max. spanning tree the links of 'b'
        are removed as this node is not a flanking node of the path. Because of this, the edge between
        'd' and 'e' will be kept during the mst forming a complete path from a to f.
        >>> S.add_path([0, 1, 2])
        >>> S.join_paths_max_span_tree(0, hub_solving_method='bandwidth permutation',
        ... node_degree_threshold=4)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3, 4, 5]]

        """
        matrix = self.matrix.copy()

        if confidence_score is not None:
            matrix.data[matrix.data <= confidence_score] = 0
        matrix.setdiag(0)
        matrix.eliminate_zeros()

        # remove all contacts that are within paths
        internal_nodes = []
        paths = self.get_all_paths(pg_base=True)
        for path in paths:
            if len(path) > 3:
                internal_nodes.extend(path[1:-1])

        # get the node degree from the source  matrix.
        node_degree = dict([(x, matrix[x, :].nnz) for x in range(matrix.shape[0])])

        # remove hubs from internal nodes
        for node in internal_nodes:
            if node_degree[node] > 2:
                # get the bin id of the other nodes that link with the query node
                adj_nodes = np.flatnonzero(matrix[node, :].todense().A)
                for adj_node in adj_nodes:
                    if adj_node not in self.pg_base.adj[node].keys():
                        # remove the contact between node and adj_node in the matrix
                        matrix[node, adj_node] = 0
                        matrix[adj_node, adj_node] = 0
        matrix.eliminate_zeros()

        # define node degree threshold as the degree for the 90th percentile
        if node_degree_threshold is None:
            node_degree_threshold = np.percentile(node_degree.values(), 99)
        len_nodes_to_remove = len([x for x in node_degree.values() if x >= node_degree_threshold])

        if len_nodes_to_remove > 0:
            log.info(" {} ({:.2f}%) nodes will be removed because they are above the degree "
                     "threshold".format(len_nodes_to_remove, 100*float(len_nodes_to_remove)/matrix.shape[0]))
            # remove from nxG nodes with degree > node_degree_threshold
            # nodes with high degree are problematic
            to_remove = []
            for node, degree in node_degree.iteritems():
                if degree >= node_degree_threshold:
                    # unset the rows and cols of hubs
                    # this is done in `matrix` as well
                    # as self.matrix because the self.matrix
                    # is used for permutation and must not contain
                    # the hub data.
                    log.debug("removing hub node {} with degree {} (thresh: {})"
                              "".format(node, degree, node_degree_threshold))

                    to_remove.append(node)

            to_remove = np.unique(to_remove)
            if len(to_remove) > 0:
                self.matrix[to_remove, :] = 0
                self.matrix[:, to_remove] = 0
                matrix[to_remove, :] = 0
                matrix[:, to_remove] = 0

                matrix.eliminate_zeros()
                self.matrix.eliminate_zeros()

        self.matrix = matrix

        if len(self.matrix.data) == 0:
            # if matrix is empty after removing intra-path contacts
            # then nothing is left to do.
            return
        nxG = self.make_nx_graph()
        nx.write_graphml(nxG, "{}/pre_mst_iter_{}.graphml".format(self.out_folder, self.iteration))
        # compute maximum spanning tree
        nxG = nx.maximum_spanning_tree(nxG, weight='weight')
        log.debug("saving maximum spanning tree network {}/mst_iter_{}.graphml".format(self.out_folder,
                                                                                      self.iteration))
        nx.write_graphml(nxG, "{}/mst_iter_{}.graphml".format(self.out_folder, self.iteration))
        degree = np.array(dict(nxG.degree()).values())
        if len(degree[degree > 2]):
            # count number of hubs
            log.info("{} hubs were found".format(len(degree[degree>2])))

        if hub_solving_method == 'remove weakest':
            self._remove_weakest(nxG)
        else:
            log.debug("degree: {}".format(node_degree))
            self._bandwidth_permute(nxG, node_degree, node_degree_threshold)

    def _bandwidth_permute(self, G, node_degree, node_degree_threshold):
        """
        Based on the maximum spanning tree graph hubs are resolved using the
        bandwidth permutation method.

        Parameters
        ----------
        G : maximum spanning tree networkx graph
        matrix :
        Returns
        -------

        """

        is_hub = set()
        # 1. based on the resulting maximum spanning tree, add the new edges to
        #    the paths graph unless any of the nodes is a hub
        for u, v, data in G.edges(data=True):
            edge_has_hub = False
            for node in [u, v]:
                if G.degree(node) > 2:
                    is_hub.add(node)
                    edge_has_hub = True
            if edge_has_hub is False:
                if u in self.pg_base[v]:
                    # skip same path nodes
                    continue
                else:
                    self.add_edge(u, v, weight=data['weight'])

        if len(is_hub) == 0:
            return

        # 2. Find nodes with degree > 2 and arrange them using the bandwidth permutation
        solved_paths = []
        seen = set()

        node_degree_mst = dict(G.degree(G.node.keys()))
        for node, degree in sorted(node_degree_mst.iteritems(), key=lambda (k,v): v, reverse=True):
            if node in seen:
                continue
            if degree > 2:
                paths_to_check = [self.pg_base[node]]
                seen.update(self.pg_base[node])
                for v in G.adj[node]:
                    # only add paths that are not the same path
                    # already added.
                    if v not in self.pg_base[node]:
                        paths_to_check.append(self.pg_base[v])
                        seen.update(self.pg_base[v])

                # check, that the paths_tho_check do not contain hubs
                if len(paths_to_check) > 1:
                    check = True

                    solved_paths.append(Scaffolds.find_best_permutation(self.matrix, paths_to_check))
                    log.debug("best permutation: {}".format(solved_paths[-1]))

        for s_path in solved_paths:
            # add new edges to the paths graph
            for index, path in enumerate(s_path[:-1]):
                # s_path has the form: [1, 2, 3], [4, 5, 6], [7, 8] ...
                # the for loops selects pairs as (3, 4), (6,7) as degest to add
                u = path[-1]
                v = s_path[index + 1][0]
                self.add_edge(u, v, weight=self.matrix[u, v])
    @staticmethod
    def _return_paths_from_graph(G):
        """
        Returns all paths from a networkX graph.
        The graph should only contain paths, this means that no node has a degree > 2


        Parameters
        ----------
        G

        Returns
        -------
        List of paths

        Examples
        --------

        >>> G = nx.path_graph(4)
        >>> G.add_edge(10, 11)
        >>> G.add_edge(11, 12)
        >>> G.add_edge(12, 13, weight=6)
        >>> Scaffolds._return_paths_from_graph(G)
        [[3, 2, 1, 0], [13, 12, 11, 10]]
        """
        path_list = []
        for conn_component in nx.connected_component_subgraphs(G):
            source = next(iter(conn_component))  # get one random node from the connected component
            path = [source]

            # the algorithm works by adding to path the adjacent nodes one after the other.
            # Because the source node could be in the middle of a path, basically two paths
            # are computed and merged. Eg. for the path [0, 1, 2, 3, 4] where `source` = 2
            # the algorithm finds the path for neighbor `1` which is [0, 1], and the path for neighbor `3`
            # which is [3, 4]. The path building starts by [2], then progress through [2, 1, 0], then is inverted
            # and to add the [3, 4] path to yield [0, 1, 2, 3, 4].
            for next_node in sorted(G[source]):
                seen = source
                i = 0
                while True:
                    i += 1
                    if i > len(conn_component):
                        raise ScaffoldException
                    adj_list = [x for x in G[next_node] if x != seen]
                    if len(adj_list) == 0:
                        path.append(next_node)
                        break
                    path.append(next_node)
                    seen = next_node
                    next_node = adj_list[0]
                path = path[::-1]
            path_list.append(path)
        return path_list

    def _remove_weakest(self, G):
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

        Returns
        -------
        None
        """
        node_degree_mst = dict(G.degree(G.node.keys()))
        for node, degree in sorted(node_degree_mst.iteritems(), key=lambda (k, v): v, reverse=True):
            if degree > 2:
                # check if node already is inside a path
                if len(self.pg_base.adj[node]) == 2:
                    # this could indicate a problematic case
                    log.info("Hub node is already inside a path {},  node_id:{}".format(G.node[node]['name'], node))

                # prune single nodes but only on first iteration. Single nodes are defined as:
                #            o  <- single node
                #           /
                #  o---o---o----o---o
                # a single node, is a node adj to a hub, whose other adj nodes have all degree 2
                # adj_degree looks like: [(90, 2), (57, 2), (59, 1)], where is tuple is (node_id, degree)
                adj_degree = sorted([(x, node_degree_mst[x]) for x in G.adj[node].keys()], key=lambda(k, v): v)[::-1]
                if self.iteration == 0 and len(adj_degree) == 3 and \
                   adj_degree[0][1] == 2 and adj_degree[1][1] == 2 and adj_degree[2][1] == 1:
                    node_to_prune = adj_degree[2][0]
                    # check that the node is not part of an split scaffold. If that
                    # is the case, do not remove it. To check if a node is part of an split scaffold
                    # what needs to be done is to check if it belongs to a path that is larger than 1.
                    # calling self.pg_base[x] will return the path containing node x
                    if len(self.pg_base[node_to_prune]) == 1:
                        log.debug("Pruning single node: {}".format(G.node[node_to_prune]['name']))
                        G.remove_node(node_to_prune)
                        self._remove_bin_path(self.pg_base.node[node_to_prune]['initial_path'], split_scaffolds=True)
                        continue
                # the adj variable looks like:
                # [(90, {'weight': 1771.3}), (57, {'weight': 2684.6}), (59, {'weight': 14943.6})]
                adj = sorted(G.adj[node].iteritems(), key=lambda (k, v): v['weight'])
                # remove the weakest edges but only if either of the nodes is not a hub
                for adj_node, attr in adj[:-2]:
                    if len(G.adj[adj_node]) > 3:
                        log.warn("\n\nHub-hub contact for bin_id:{}\tscaffold: {}\tdegree: {}\n"
                                 "with bin_id: {}\tscaffold: {}\tdegree:{}\n\n"
                                 "##############\n"
                                 "these cases could introduce problems in the assembly.\n"
                                 "Thus, node is being removed from the graph.\n"
                                 "##############\n\n".format(node, self.pg_base.node[node]['name'],
                                                             len(G.adj[node]), adj_node,
                                                             self.pg_base.node[adj_node]['name'],
                                                             len(G.adj[adj_node])))

                        # adj node is hub. In this case remove the node and the scaffold it belongs
                        # to from the graph
                        path  = len(self.pg_base[node])
                        # only remove if the path is not longer than 5. Otherwise
                        # a quite large scaffold can be removed.
                        if len(path) < 5:
                            for node_id in path:
                                G.remove_node(node_id)
                            self._remove_bin_path(self.pg_base.node[node]['initial_path'], split_scaffolds=True)
                            continue

                    log.debug("Removing weak edge {}-{} weight: {}".format(G.node[node]['name'],
                                                                           G.node[adj_node]['name'],
                                                                           attr['weight']))
                    G.remove_edge(node, adj_node)
            if degree <= 2:
                break

        # now, the G graph should contain only paths
        for path in Scaffolds._return_paths_from_graph(G):
            self.add_path(path)


    def add_path(self, path):
        """
        Adds all the edges to the internal PathGraphs based on the given path


        Parameters
        ----------
        path : list of pg_base nodes that want to be added

        Returns
        -------

        """

        # for each id in in path, find the 'initial_path' in the un-merged hic-matrix
        # that it points to. Because of internal splits a set is created to avoid duplicates.

        seen = set()
        bins_path = []
        for x in path:
            # the initial path from pg_base could be an split from a larger path.
            # Thus, to select the original, un split path, the first node of the
            # pg_base initial path is used to query the matrix_bins PathGraph to return
            # the full path
            init_path = self.matrix_bins[self.pg_base.node[x]['initial_path'][0]]
            if init_path[0] not in seen:
                bins_path.append(init_path)
                seen.update(init_path)

        # find best orientation
        if len(path) < 10:
            best_path = Scaffolds.find_best_permutation(self.hic.matrix, bins_path, only_expand_but_not_permute=True)
            if best_path != Scaffolds.find_best_direction(self.hic.matrix, bins_path):
                pass
        else:
            best_path = Scaffolds.find_best_direction(self.hic.matrix, bins_path)

        for path_u, path_v in zip(best_path[:-1], best_path[1:]):
            self.add_edge_matrix_bins(path_u[-1], path_v[0])

    def make_nx_graph(self):
        """
        makes a networkx graph based on the current paths and matrix. For the links between nodes
        in the same path, the weight is set to the matrix maximum + 1.

        Returns
        -------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 20, 30, 1), ('c-3', 30, 40, 1)]
        >>> A = csr_matrix(np.array([[2,2,1,1,1,1],[2,2,1,1,1,1],
        ... [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]]))

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)
        >>> S = Scaffolds(hic)
        >>> S.matrix.todense()
        matrix([[4, 4, 2, 2, 2, 2],
                [4, 4, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2]])

        >>> G = S.make_nx_graph()

        the edges for adjacent nodes in the same path, should had as weight
        the max value of the matrix +1.
        For this example, the paths are [0, 1, 2] and [3, 4, 5]. Thus, the contacts
        for (0,1) and (1,2) should have a weight of 5.
        >>> G.adj[0]
        AtlasView({1: {'weight': 5.0}, 2: {'weight': 2.0}, 3: {'weight': 2.0}, 4: {'weight': 2.0}, 5: {'weight': 2.0}})
        >>> G.node[0]
        {'start': 0, 'length': 10, 'end': 10, 'name': 'c-0', 'coverage': 1}


        The following matrix is used:

                      ______
                     |      |
                 a---b---c--d
                      \    /
                       --e--f
                                0  1  2  3  4  5
                                a  b  c  d  e  f
        >>> matrix = np.array([[0, 3, 0, 0, 0, 0], # a 0
        ...                    [0, 0, 3, 2, 2.5, 0], # b 1
        ...                    [0, 0, 0, 3, 0, 0], # c 2
        ...                    [0, 0, 0, 0, 1, 0], # d 3
        ...                    [0, 0, 0, 0, 0, 3], # e 4
        ...                    [0, 0, 0, 0, 0, 0]])# f 5

        >>> cut_intervals = [('a', 0, 1, 1), ('b', 1, 2, 1), ('c', 2, 3, 1),
        ... ('d', 0, 1, 1), ('e', 1, 2, 1), ('f', 0, 1, 1)]


        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)

        >>> G = S.make_nx_graph()
        >>> list(G.edges(data=True))
        [(0, 1, {'weight': 3.0}), (1, 2, {'weight': 3.0}), \
(1, 3, {'weight': 2.0}), (1, 4, {'weight': 2.5}), \
(2, 3, {'weight': 3.0}), (3, 4, {'weight': 1.0}), \
(4, 5, {'weight': 3.0})]
        """

        nxG = nx.Graph()
        if self.pg_base is not None:
            path_graph_to_use = self.pg_base
        else:
            path_graph_to_use = self.matrix_bins

        for node_id, node in path_graph_to_use.node.iteritems():
            # when saving a networkx object, numpy number types or lists, are not accepted
            nn = node.copy()
            for attr, value in nn.iteritems():
                if isinstance(value, np.int64):
                    nn[attr] = int(value)
                elif isinstance(value, np.float64):
                    nn[attr] = float(value)
                elif isinstance(value, list):
                    nn[attr] = ", ".join([str(x) for x in value])

            nxG.add_node(node_id, **nn)

        matrix = self.matrix.tocoo()
        max_weight = matrix.data.max() + 1

        for u, v, weight in zip(matrix.row, matrix.col, matrix.data):
            if u == v:
                continue

            if u in path_graph_to_use.adj[v]:
                # u and v are neighbors
                nxG.add_edge(u, v, weight=float(max_weight))
            else:
                nxG.add_edge(u, v, weight=float(weight))

        return nxG

    def add_scaffold_edge(self, _bin_u, _bin_v, _weight, direction):
        """
        To add an edge on the scaffold pathgraph the direction of the
        individual scaffolds has to be updated accordingly.

        Thus, apart from adding the scaffold edge, the orientation of
        the scaffolds is updated.

        Parameters
        ----------
        _bin_u
        _bin_v
        _weight
        direction

        Returns
        -------


        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-1', 10, 20, 1),
        ... ('c-1', 20, 30, 1), ('c-2', 0, 10, 1), ('c-2', 10, 20, 1)]
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix(np.array(
        ... [[80, 15,  5,  1,  2,  3],
        ...  [15, 80, 15,  2,  3,  5],
        ...  [ 5, 15, 80,  3,  15, 10],
        ...  [ 1,  2,  3, 80, 15,  5],
        ...  [ 2,  3,  15, 15, 80, 15],
        ...  [ 3,  5, 10,  5, 15, 80]]))

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)

        >>> S = Scaffolds(hic)

        Add and edge between c-0 and c-1, that requires a change in the direction
        of c-0. The numbers in the add_edge function are
        the bin ids, 1 is the second bin of 'c-0' and 2, is the first bin of
        'c-1'
        >>> S.add_edge(0, 2)

        >>> S.scaffold.path
        {'c-0, c-1': ['c-0', 'c-1']}
        >>> S.scaffold.node['c-0']
        {'direction': '-', 'end': 20, 'name': 'c-0', 'start': 0, 'length': 20, 'path': [1, 0]}
        >>> (S.scaffold.node['c-0']['direction'], S.scaffold.node['c-1']['direction'])
        ('-', '+')

        Now, an edge is added that causes the path ['c-0', 'c-1'] to be inverted and the c-2
        scaffold to be inverted as well. Fot the resulting path, the directions should be
        ('-', '-', '+') for c-2, c-0, and c-1 respectively. Notice that c-0, c-1 should change
        from ('-', '+') to now c-1, c-0 ('-', '+')
        >>> S.add_edge(4, 3)
        >>> S.scaffold.path
        {'c-2, c-0, c-1': ['c-2', 'c-1', 'c-0']}
        >>> S.scaffold.node['c-1']
        {'direction': '-', 'end': 30, 'name': 'c-1', 'start': 10, 'length': 20, 'path': [3, 2]}

        Node c-0 should be back to orientation + and the path should be [0, 1]
        >>> S.scaffold.node['c-0']
        {'direction': '+', 'end': 20, 'name': 'c-0', 'start': 0, 'length': 20, 'path': [0, 1]}

        >>> [S.scaffold.node[x]['direction'] for x in S.scaffold.path.values()[0]]
        ['-', '-', '+']
        """
        scaffold_u = self.bin_id_to_scaff[_bin_u]
        scaffold_v = self.bin_id_to_scaff[_bin_v]
        if direction[0] == '-':
            # change the direction of the path containing the scaffold_u
            for scaff_name in self.scaffold[scaffold_u]:
                if self.scaffold.node[scaff_name]['direction'] == "+":
                    self.scaffold.node[scaff_name]['direction'] = "-"
                elif self.scaffold.node[scaff_name]['direction'] == "-":
                    self.scaffold.node[scaff_name]['direction'] = "+"
                else: # when direction has not been set:
                    self.scaffold.node[scaff_name]['direction'] = "-"
                self.scaffold.node[scaff_name]['path'] = self.scaffold.node[scaff_name]['path'][::-1]

        if direction[1] == '-':
            # change the direction of the path containing the scaffold_u
            for scaff_name in self.scaffold[scaffold_v]:
                if self.scaffold.node[scaff_name]['direction'] == "+":
                    self.scaffold.node[scaff_name]['direction'] = "-"
                elif self.scaffold.node[scaff_name]['direction'] == "-":
                    self.scaffold.node[scaff_name]['direction'] = "+"
                else: # when direction has not been set:
                    self.scaffold.node[scaff_name]['direction'] = "-"
                self.scaffold.node[scaff_name]['path'] = self.scaffold.node[scaff_name]['path'][::-1]

        # check that direction is properly set
        for scaff in [scaffold_u, scaffold_v]:
            scaff_direction = '+' if self.scaffold.node[scaff]['path'][0] < self.scaffold.node[scaff]['path'][-1] else "-"
            # this check is only for scaffolds with only more than one bin
            if len(self.scaffold.node[scaff]['path']) > 1:
                assert self.scaffold.node[scaff]['direction'] == scaff_direction, "mismatch with scaffold direction"


        self.scaffold.add_edge(scaffold_u, scaffold_v, weight=_weight)

    def add_edge_matrix_bins(self, bin_u, bin_v, weight=None):
        """
        Adds an edge using the matrix_bins pathgraph. An edge to the
        scaffolds pathgraph is also added.

        Parameters
        ----------
        bin_u
        bin_v
        weight

        Returns
        -------

        """
        try:
            direction = self.matrix_bins.add_edge(bin_u, bin_v, return_direction=True, weight=weight)
            self.add_scaffold_edge(bin_u, bin_v, weight, direction)
        except PathGraphEdgeNotPossible:
            log.debug("*WARN* Skipping add edge between {} and {}".format(bin_u, bin_v))

    # def add_edge_scaffold(self, scaff_a, scaff_v, weight=None):

    def add_edge(self, u, v, weight=None):
        """
        Adds and edge both in the reduced PathGraph (pg_base), in the
        matrix_bins PathGraph and in the scaffolds PathGraph
        Parameters
        ----------
        u node index
        v node index
        weight weight

        Returns
        -------

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-1', 10, 20, 1),
        ... ('c-1', 20, 30, 1), ('c-2', 0, 10, 1), ('c-2', 10, 20, 1)]
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix(np.array(
        ... [[80, 15,  5,  1,  2,  3],
        ...  [15, 80, 15,  2,  3,  5],
        ...  [ 5, 15, 80,  3,  15, 10],
        ...  [ 1,  2,  3, 80, 15,  5],
        ...  [ 2,  3,  15, 15, 80, 15],
        ...  [ 3,  5, 10,  5, 15, 80]]))

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=A)

        >>> S = Scaffolds(hic)

        before adding an edge, 'c-0' and 'c-1' are not joined
        and the direction of 'c-1' is '+'

        >>> S.matrix_bins.path
        {'c-2': [4, 5], 'c-1': [2, 3], 'c-0': [0, 1]}
        >>> S.scaffold.node['c-1']
        {'direction': '+', 'end': 30, 'name': 'c-1', 'start': 10, 'length': 20, 'path': [2, 3]}

        >>> S.add_edge(1, 3)

        c-0 and c-1 are joined and c-1 is inverted
        >>> S.scaffold.path
        {'c-0, c-1': ['c-0', 'c-1']}

        >>> S.matrix_bins.path
        {'c-2': [4, 5], 'c-0, c-1': [0, 1, 3, 2]}

        >>> S.scaffold.node['c-1']
        {'direction': '-', 'end': 30, 'name': 'c-1', 'start': 10, 'length': 20, 'path': [3, 2]}

        >>> S.add_edge(2, 4)
        >>> S.scaffold.path
        {'c-0, c-1, c-2': ['c-0', 'c-1', 'c-2']}

        >>> S.scaffold.node['c-1']
        {'direction': '-', 'end': 30, 'name': 'c-1', 'start': 10, 'length': 20, 'path': [3, 2]}

        New test based on split_merge
        >>> S = Scaffolds(hic)
        >>> S.split_and_merge_contigs(num_splits=1, normalize_method='none')

        >>> S.matrix_bins.path
        {'c-2': [4, 5], 'c-1': [2, 3], 'c-0': [0, 1]}
        >>> S.scaffold.node['c-1']
        {'direction': '+', 'end': 30, 'name': 'c-1', 'merged_path_id': 1, 'start': 10, 'length': 20, 'path': [2, 3]}

        after split and merge, we have matrix_bins and pg_base (pg_base being the merge of contigs)
        Now, the ids refer to the 'contig' id and not to the bin ids.
        Thus, 'c-0' id is 0, 'c-1' is 1 and 'c-2' is 2
        >>> S.add_edge(0, 2)
        >>> S.matrix_bins.path
        {'c-0, c-2': [0, 1, 5, 4], 'c-1': [2, 3]}

        >>> S.scaffold.path
        {'c-0, c-2': ['c-0', 'c-2']}

        # check that the direction attribute has been set to the nodes
        >>> S.scaffold.node['c-0']
        {'direction': '+', 'end': 20, 'name': 'c-0', 'merged_path_id': 0, 'start': 0, 'length': 20, 'path': [0, 1]}
        >>> S.add_edge(2,1)
        >>> S.matrix_bins.path
        {'c-0, c-2, c-1': [0, 1, 5, 4, 2, 3]}

        >>> S.scaffold.node['c-1']
        {'direction': '+', 'end': 30, 'name': 'c-1', 'merged_path_id': 1, 'start': 10, 'length': 20, 'path': [2, 3]}
        >>> S.scaffold.node['c-2']
        {'direction': '-', 'end': 20, 'name': 'c-2', 'merged_path_id': 2, 'start': 0, 'length': 20, 'path': [5, 4]}
        """

        # get the initial nodes that should be merged
        try:
            initial_path_u = self.pg_base.node[u]['initial_path']
            initial_path_v = self.pg_base.node[v]['initial_path']
        except:
            initial_path_u = [u]
            initial_path_v = [v]

        best_paths = Scaffolds.find_best_permutation(self.hic.matrix, [initial_path_u, initial_path_v],
                                                     return_all_sorted_best_paths=True)
        path_added = False
        for iter_ in range(len(best_paths)):
            best_path = best_paths[iter_][0]
            try:
                bin_u = best_path[0][-1]
                bin_v = best_path[1][0]
                direction = self.matrix_bins.add_edge(bin_u, bin_v, return_direction=True, weight=weight)
                self.pg_base.add_edge(u, v, weight=weight)
                self.add_scaffold_edge(bin_u, bin_v, weight, direction)
                path_added = True
            except PathGraphEdgeNotPossible:
                raise ScaffoldException("Edge not possible. Edge between {} and {} corresponding "
                                        "to {} and {} not possible.".format(u, v, self.pg_base.get_path_name_of_node(u),
                                                                            self.pg_base.get_path_name_of_node(v)))
            if path_added:
                break
        if path_added is False:
            raise ScaffoldException ("Can't add edge between {} and {} corresponding to ({}) and ({})"
                                     "".format(u, v, self.pg_base.get_path_name_of_node(u),
                                               self.pg_base.get_path_name_of_node(v)))

    def delete_edge_from_matrix_bins(self, u, v):
        """
        deletes edge u,v. The edge is assumed to be
        from the matrix_bins PathGraph. The edge is also
        deleted in the scaffolds PathGraph

        THE EDGE IS NOT DELETED FROM pg_base

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.delete_edge_from_matrix_bins(1,2)
        Traceback (most recent call last):
        ...
        ScaffoldException: *ERROR* Edge can not be deleted inside an scaffold.
        Scaffold name: c-0, edge: 1-2
        >>> S.add_edge(2, 3)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3, 4], [5]]
        >>> list(S.scaffold.get_all_paths())
        [['c-3'], ['c-0', 'c-2']]
        >>> S.delete_edge_from_matrix_bins(2, 3)
        >>> list(S.get_all_paths())
        [[0, 1, 2], [3, 4], [5]]
        >>> list(S.scaffold.get_all_paths())
        [['c-3'], ['c-2'], ['c-0']]
        """
        # delete edge in self.scaffold
        scaff_u = self.matrix_bins.node[u]['name']
        scaff_v = self.matrix_bins.node[v]['name']
        if u not in self.matrix_bins[v]:
            raise ScaffoldException("*ERROR* Edge can not be deleted because "
                                    "the bins ({}, {}) are not connected.".format(u, v))
        if scaff_u == scaff_v:
            raise ScaffoldException("*ERROR* Edge can not be deleted inside an scaffold.\n"
                                    "Scaffold name: {}, edge: {}-{}".format(scaff_u, u, v))
        self.matrix_bins.delete_edge(u, v)
        self.scaffold.delete_edge(scaff_u, scaff_v)

        # self.pg_base.delete_edge(u, v)
        # for node in [u, v]:
        #     # get new name of path (assigned_automatically by PathGraph)
        #     path_name = self.pg_base.get_path_name_of_node(node)
        #     # get the ids that belong to that path
        #     for node_id in self.pg_base.path[path_name]:
        #         name, start, end, extra = self.hic.cut_intervals[node_id]
        #         self.hic.cut_intervals[node_id] = (path_name, start, end, extra)

    def delete_edge_from_scaffolds(self, u, v):
        """
        deletes edge u,v. The edge is assumed to be
        from the scaffold PathGraph. The edge is also deleted
        in the matrix_bins

        THE EDGE IS NOT DELETED FROM pg_base

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.delete_edge_from_scaffolds('c-0', 'c-2')
        Traceback (most recent call last):
        ...
        ScaffoldException: *ERROR* Edge can not be deleted because the scaffolds (c-0, c-2) are not connected.
        >>> S.add_edge(2, 3)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3, 4], [5]]
        >>> list(S.scaffold.get_all_paths())
        [['c-3'], ['c-0', 'c-2']]
        >>> S.delete_edge_from_scaffolds('c-0', 'c-2')
        >>> list(S.get_all_paths())
        [[0, 1, 2], [3, 4], [5]]
        >>> list(S.scaffold.get_all_paths())
        [['c-3'], ['c-2'], ['c-0']]
        """
        # delete edge in self.scaffold
        if u not in self.scaffold.adj[v]:
            raise ScaffoldException("*ERROR* Edge can not be deleted because "
                                    "the scaffolds ({}, {}) are not connected.".format(u, v))

        index_u = self.scaffold[u].index(u)
        index_v = self.scaffold[v].index(v)

        if index_u > index_v:
            u, v = v, u
        path_u = self.scaffold.node[u]['path']
        path_v = self.scaffold.node[v]['path']

        # identify the bins that form the edge
        self.matrix_bins.delete_edge(path_u[-1], path_v[0])
        self.scaffold.delete_edge(u, v)

        # self.pg_base.delete_edge(u, v)
        # for node in [u, v]:
        #     # get new name of path (assigned_automatically by PathGraph)
        #     path_name = self.pg_base.get_path_name_of_node(node)
        #     # get the ids that belong to that path
        #     for node_id in self.pg_base.path[path_name]:
        #         name, start, end, extra = self.hic.cut_intervals[node_id]
        #         self.hic.cut_intervals[node_id] = (path_name, start, end, extra)

    @logit
    def get_nearest_neighbors(self, confidence_score):
        """
        The algorithm works in two stages

        1. paths are connected for clear cases
        2. the bandwidth method is used for complicated cases

        Parameters
        ----------
        confidence_score : threshdold to remove values from the hic matrix

        Returns
        -------

        """
        def _join_degree_one_nodes():
            # iterate by edge, joining those paths
            # that are unambiguous
            _degree = G.degree()
            for u, v, weight in G.get_edges():
                if _degree[u] == 1 and _degree[v] == 1:
                    try:
                        self.pg_base.add_edge(u, v, weight=weight)
                    except PathGraphEdgeNotPossible as e:
                        log.info(e)
                    # unset edge in graph
                    G[u, v] = 0
                # TODO: some singletons with degree two could be  also joined

        matrix = self.matrix.copy()

        matrix.data[matrix.data <= confidence_score] = 0
        matrix.eliminate_zeros()

        # get nodes that can be joined
        flanks = []  # a node that is either the first or last in a path
        singletons = []  # a node that has not been joined any other
        for path in self.get_all_paths():
            if len(path) == 1:
                singletons.extend(path)
            else:
                flanks.extend([path[0], path[-1]])

        edge_nodes = flanks + singletons
        # remove from matrix all nodes that are not `edge_nodes`
        # turn matrix in to a simple graph
        ma = matrix[edge_nodes, :][:, edge_nodes]
        # remove diagonal
        ma.setdiag([0]*len(edge_nodes))
        G = SimpleGraph(ma, edge_nodes)

        ## temp
        import networkx as nx
        nxG = nx.Graph()
        for u,v,weight in G.get_edges():
            nxG.add_edge(u,v, weight=weight)

        nlist = [node for node in nxG]
        for node in nlist:
            path = self.pg_base[node]
            if node == 25:
                print path
            nxG.add_edge(path[0], path[-1], weight=1e4)
        nx.write_gml(nxG, "data/G_flanks.gml")
        exit()
        ## end temp
        # remove edge if edge_nodes are connected
        for path in self.get_all_paths():
            if len(path) == 2:
                G[path[0], path[1]] = 0
        # add path edges
        _join_degree_one_nodes()

        # try to solve hubs by using the bandwidth
        node_degree = G.degree()
        seen = set()
        solved_paths = []
        hub_below_5 = set()
        hub_over_5 = set()
        for node, degree in node_degree.iteritems():
            if degree <= 4:
                hub_below_5.add(node)
            else:
                hub_over_5.add(node)

        # sort edges by decreasing weight
        for u, v, weight in sorted(G.get_edges(), key=lambda(u, v, w): w, reverse=True):
            fixed_paths = []
            for node in G.adj(u) + G.adj(v):
                if node in seen:
                    continue
                path = self.pg_base[node]
                seen.update(path)

                fixed_paths.append(self.pg_base[node])
            if len(fixed_paths) > 1:
                check = True
                for _path in fixed_paths:
                    for node in _path:
                        if node in hub_over_5:
                            log.debug("{} in path {} is hub. Discarding {}".format(node, _path, fixed_paths))
                            check = False
                            break
                if check is True:
                    solved_paths.append(Scaffolds.find_best_permutation(self.matrix, fixed_paths))

        for s_path in solved_paths:
            for index, path in enumerate(s_path[:-1]):
                # s_path has the form: [1, 2, 3], [4, 5, 6]
                u = path[-1]
                v = s_path[index + 1][0]
                self.pg_base.add_edge(u, v, weight=G[u, v])
                for node in [u, v]:
                    for adj in G.adj(node):
                        G[node, adj] = 0

        _join_degree_one_nodes()

    @logit
    def get_nearest_neighbors_2(self, confidence_score):
        """

        Parameters
        ----------
        confidence_score : threshold to prune the matrix. Any value
        below this score is removed.

        Returns
        -------

        """
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
            neighbors[index] = (0, 0)

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
            flanks = set(Scaffolds.flatten_list(
                    [[x[0],x[-1]] for x in self.merged_paths]))
            for edge in G.edges():
                if len(flanks.intersection(edge)) == 0:
                    G.remove_edge(*edge)
        """
        return G

    #########

    def remove_paths(self, ids_to_remove):
        """
        Removes a path from the self.path list
        using the given ids_to_remove
        Parameters
        ----------
        ids_to_remove : List of ids to be removed. Eg. [1, 5, 20]

        Returns
        -------
        None
        """
        paths = self.get_all_paths()
        # translate path indices in mask_list back to contig ids
        # and merge into one list using sublist trick
        paths_to_remove = [paths[x] for x in ids_to_remove]
        contig_list = [item for sublist in paths_to_remove for item in sublist]

        self.pg_base.remove_nodes_from(contig_list)
        # reset the paths
        self.paths = None

    def has_edge(self, u, v):
        return self.pg_base.has_edge(u, v) or self.pg_base.has_edge(v, u)

    def check_edge(self, u, v):
        # check if the edge already exists
        if self.has_edge(u, v):
            message = "Edge between {} and {} already exists".format(u, v)
            raise ScaffoldException (message)

        # check if the node has less than 2 edges
        for node in [u, v]:
            if self.pg_base.degree(node) == 2:
                message = "Edge between {} and {} not possible,  contig {} " \
                          "is not a flaking node ({}, {}). ".format(u, v, node,
                                                                    self.pg_base.predecessors(node),
                                                                    self.pg_base.successors(node))
                raise ScaffoldException (message)

        # check if u an v are the two extremes of a path,
        # joining them will create a loop
        if self.pg_base.degree(u) == 1 and self.pg_base.degree(v) == 1:
            if v in self.pg_base[u]:
                message = "The edges {}, {} form a closed loop.".format(u, v)
                raise ScaffoldException (message)

    def get_neighbors(self, u):
        """
        Give a node u, it returns the
        successor and predecessor nodes

        Parameters
        ----------
        u : Node

        Returns
        -------
        predecessors and sucessors

        """
        return self.pg_base.predecessors(u) + self.pg_base.successors(u)

    def save_network(self, file_name):
        nx.write_gml(self.pg_base, file_name)


class ScaffoldException(Exception):
        """Base class for exceptions in Scaffold."""


def get_test_matrix(cut_intervals=None, matrix=None):
    hic = HiCMatrix.hiCMatrix()
    hic.nan_bins = []
    if matrix is None:
        matrix = np.array([
                          [1,  8,  5, 3, 0, 8],
                          [0,  4, 15, 5, 1, 7],
                          [0,  0,  0, 7, 2, 8],
                          [0,  0,  0, 0, 1, 5],
                          [0,  0,  0, 0, 0, 6],
                          [0,  0,  0, 0, 0, 0]])

    # make matrix symmetric
    matrix = csr_matrix(matrix + matrix.T)

    if not cut_intervals:
        cut_intervals = [('c-0', 0, 1, 1), ('c-1', 0, 1, 1), ('c-2', 0, 1, 1), ('c-4', 0, 1, 1), ('c-4', 0, 1, 1)]
    hic.matrix = csr_matrix(matrix[0:len(cut_intervals), 0:len(cut_intervals)])
    hic.setMatrix(hic.matrix, cut_intervals)
    return hic


class SimpleGraph(object):
    """
    Object to use a symmetric matrix as a graph
    """
    def __init__(self, matrix, row_labels):
        assert matrix.shape[0] == matrix.shape[1], "matrix no squared"
        assert len(row_labels) == matrix.shape[0], "row len != matrix shape"
        self.labels2index = dict([[row_labels[index], index] for index in range(len(row_labels))])
        self.index2label = row_labels
        self.matrix = matrix.tolil()

    def __getitem__(self, index):
        return self.matrix[self.labels2index[index[0]], self.labels2index[index[1]]]

    def __setitem__(self, index, x):
        u, v = self.labels2index[index[0]], self.labels2index[index[1]]
        self.matrix[u, v] = x
        self.matrix[v, u] = x

    def adj(self, u):
        """

        Parameters
        ----------
        u

        Returns
        -------

        >>> M = lil_matrix((4, 4))
        >>> nodes = ['a', 'b', 'c', 'd']
        >>> G = SimpleGraph(M, nodes)
        >>> G['a', 'b'] = 1
        >>> G['a', 'c'] = 1
        >>> G.adj('a')
        ['b', 'c']
        >>> G.adj('d')
        []

        """
        u = self.labels2index[u]

        return [self.index2label[x] for x in np.flatnonzero(self.matrix[u, :].A[0])]

    def get_edges(self):
        """

        Returns
        -------

        >>> M = lil_matrix((4,4))
        >>> nodes = ['a', 'b', 'c', 'd']
        >>> G = SimpleGraph(M, nodes)
        >>> G['a', 'b'] = 1
        >>> G['a', 'c'] = 1
        >>> list(G.get_edges())
        [('a', 'b', 1.0), ('a', 'c', 1.0)]

        >>> G['a', 'c'] = 0
        >>> list(G.get_edges())
        [('a', 'b', 1.0)]

        """
        cx = triu(self.matrix, k=1, format='coo')
        for i, j, v in zip(cx.row, cx.col, cx.data):
            yield (self.index2label[i], self.index2label[j], v)

    def degree(self):
        """

        Returns
        -------
        >>> M = lil_matrix((4,4))
        >>> nodes = ['a', 'b', 'c', 'd']
        >>> G = SimpleGraph(M, nodes)
        >>> G['a', 'b'] = 1
        >>> G['a', 'c'] = 1
        >>> G.degree()
        {'a': 2, 'c': 1, 'b': 1, 'd': 0}
        """

        # get degree of nodes
        node_degree = self.matrix.astype(bool).sum(0).A[0]

        # convert degree to dictionary
        node_degree = dict([(self.index2label[x], node_degree[x]) for x in range(len(node_degree))])
        return node_degree
