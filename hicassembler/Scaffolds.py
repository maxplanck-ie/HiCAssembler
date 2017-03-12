import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, triu
import logging
import time
import hicexplorer.HiCMatrix as HiCMatrix
from hicassembler.PathGraph import PathGraph, PathGraphEdgeNotPossible, PathGraphException

from hicassembler.HiCAssembler import HiCAssemblerException
from hicexplorer.reduceMatrix import reduce_matrix
from hicexplorer.iterativeCorrection import iterativeCorrection
from functools import wraps
import itertools

logging.basicConfig()
log = logging.getLogger("Scaffolds")
log.setLevel(logging.DEBUG)


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
    def __init__(self, hic_matrix):
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
        self.pg_base = PathGraph()
        self.pg_initial = None

        # initialize the contigs directed graph
        self._init_path_graph()

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

        """

        contig_path = []
        prev_label = None
        length_array = []
        self.matrix = self.hic.matrix.copy()
        self.pg_base = PathGraph()
        self.pg_initial = None

        for idx, interval in enumerate(self.hic.cut_intervals):
            label, start, end, coverage = interval
            length = end - start

            attr = {'name': label,
                    'start': start,
                    'end': end,
                    'coverage': coverage,
                    'length': length}
            length_array.append(length)

            self.pg_base.add_node(idx, **attr)
            if prev_label is not None and prev_label != label:
                self.pg_base.add_path(contig_path, name=prev_label)
                contig_path = []
            contig_path.append(idx)
            prev_label = label

        if len(contig_path) > 1:
            self.pg_base.add_path(contig_path, name=label)

    def get_all_paths(self):
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
        for v in self.pg_base:
            if v not in seen:
                yield self.pg_base[v]
            seen.update(self.pg_base[v])

    def remove_small_paths(self, min_length):
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
        >>> S.remove_small_paths(20)

        The paths that are smaller or equal to 20 are the one corresponding to c-2 and c-3.
        thus, only the path of 'c-0' is kept
        >>> [x for x in S.get_all_paths()]
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
        for path in self.get_all_paths():
            paths_total += 1
            length = (sum([self.pg_base.node[x]['length'] for x in path]))

            if length <= min_length:
                to_remove.extend(path)
                to_remove_paths.append(path)

        if len(to_remove) and len(to_remove) < self.matrix.shape[0]:
            log.debug("Removing {} scaffolds/contigs, containing {} bins, because they "
                      "are shorter than {} ".format(len(to_remove_paths),
                                                    len(to_remove),
                                                    min_length))

            self.hic.removeBins(to_remove)
            self._init_path_graph()

    def get_paths_length(self):
        for path in self.get_all_paths():
            yield (sum([self.pg_base.node[x]['length'] for x in path]))

    def get_paths_stats(self):
        import matplotlib.pyplot as plt
        paths_length = np.fromiter(self.get_paths_length(), int)
        plt.hist(paths_length, 100)
        file_name = "/tmp/stats_len_{}.pdf".format(len(paths_length))
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
        ... ('c-3', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)

        The lengths for the paths in this matrix are:
        [20 30 10 10 10]

        The sorted cumulative sum is:
        [10, 20, 30, 50, 80]
        >>> S.compute_N50(min_length=2)
        20

        """
        length = np.sort(np.fromiter(self.get_paths_length(), int))
        if len(length) == 0:
            raise HiCAssemblerException("No paths. Can't compute N50")
        length = length[length > min_length]
        if len(length) == 0:
            raise HiCAssemblerException("No paths with length > {}. Can't compute N50".format(min_length))
        cumsum = np.cumsum(length)

        # find the index at which the cumsum length is half the total length
        half_length = float(cumsum[-1]) / 2
        for i in range(len(length)):
            if cumsum[i] >= half_length:
                break

        return length[i]

    @logit
    def compute_mean_contact_matrix(self):
        """
        Builds a matrix that contains the mean contact between scaffolds

        Returns
        -------
        mean values matrix for merged paths.

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-1', 0, 10, 1),
        ... ('c-1', 10, 20, 1), ('c-2', 0, 10, 1), ('c-2', 10, 20, 1)]
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
        >>> S.pg_base.path # == {'c-0': [0, 1, 2, 3, 4, 5]}
        {'c-2': [4, 5], 'c-1': [2, 3], 'c-0': [0, 1]}
        >>> S.mean_contact_matrix()
        >>> S.matrix.todense()
        matrix([[ 4.,  2.,  2.],
                [ 2.,  2.,  2.],
                [ 2.,  2.,  2.]])

        Now there are not paths.
        >>> S.pg_base.path
        {}
        >>> S.pg_base.node
        {0: {'start': 0, 'length': 20, 'initial_path': [0, 1], \
'end': 20, 'name': 'c-0'}, 1: {'start': 0, 'length': 20, \
'initial_path': [2, 3], 'end': 20, 'name': 'c-1'}, \
2: {'start': 0, 'length': 20, 'initial_path': [4, 5], 'end': 20, 'name': 'c-2'}}

        """

        pg_merge = PathGraph()
        paths_list = []
        paths_len = []
        for index, path in enumerate(self.get_all_paths()):
            path_name = self.pg_base.get_path_name_of_node(path[0])
            # prepare new PathGraph nodes
            attr = {'length': sum([self.pg_base.node[x]['length'] for x in path]),
                    'start': self.pg_base.node[path[0]]['start'],
                    'end': self.pg_base.node[path[-1]]['end'],
                    'name': "{}".format(path_name),
                    'initial_path': path}

            pg_merge.add_node(index, attr_dict=attr)
            paths_list.append(path)
            paths_len.append(len(path))

        if len(paths_list) < 2:
            log.info("To few contigs/scaffold {}. Returning".format(len(paths_list)))
            return None

        # the reduced matrix contains the sum of the counts
        # between each contig/scaffold
        reduced_matrix = reduce_matrix(self.matrix, paths_list, diagonal=True).tocoo()
        # compute mean values for reduce matrix
        new_data = np.zeros(len(reduced_matrix.data))
        for index, value in enumerate(reduced_matrix.data):
            row_len = paths_len[reduced_matrix.row[index]]
            col_len = paths_len[reduced_matrix.col[index]]
            mean = float(value) / (row_len * col_len)
            new_data[index] = mean

        reduced_matrix.data = new_data

        # update matrix
        self.matrix = reduced_matrix.tocsr()

        if self.pg_initial is None:
            self.pg_initial = self.pg_base
        self.pg_base = pg_merge

        assert len(self.pg_base.node.keys()) == self.matrix.shape[0], "inconsistency error"

    @logit
    def merge_to_size(self, target_length=20000, reset_base_paths=True):
        """
        finds groups of bins/node that have a sum length of about the `target_length` size.
        The algorithm proceeds from the flanks of a path to the inside. If a bin/node
        is too small it is skipped.


        Parameters
        ----------
        target_length : in bp
        reset_base_paths: boolean   Reset means that the merge information is kept
                                    as the primary data, otherwise the original paths
                                    and node data is kept and the merge refers to this data. Why is this done?
                                    For contigs with dpnII based bins, the original paths can be hundreds of
                                    bins long but this detailed information is not needed. Thus, merging by
                                    size only the id of the contig and the shorter (merged) path is kept.
                                    However, subsequent merges by size of scaffolds (union of contigs) need
                                    to refer to the original contigs and all information should be kept. This
                                    is achieved by using two PathGraph objects, one that holds the current
                                    paths based on the current matrix bin ids and other PathGraph object
                                    that holds the original contig bins.

        Returns
        -------

        Examples
        --------
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> S.pg_base.path == {'c-0': [0, 1, 2, 3, 4, 5]}
        True
        >>> S.merge_to_size(target_length=20, reset_base_paths=True)
        >>> S.matrix.todense()
        matrix([[ 17.50086799,  22.87911701,  16.89293224],
                [ 22.87911701,  13.88686619,  20.50690064],
                [ 16.89293224,  20.50690064,  19.87307721]])

        Now the 'c-0' contig path is shorter
        >>> S.pg_base.path == {'c-0': [0, 1, 2]}
        True
        >>> len(S.pg_base.node)
        3
        >>> S.pg_base.node
        {0: {'start': 0, 'length': 20, 'end': 20, 'name': 'c-0', 'coverage': 1.5}, \
1: {'start': 20, 'length': 20, 'end': 40, 'name': 'c-0', 'coverage': 1.0}, \
2: {'start': 40, 'length': 20, 'end': 60, 'name': 'c-0', 'coverage': 1.0}}


        """
        log.info("merge_to_size. flank_length: {:,}".format(target_length))

        # flattened list of merged_paths  e.g [[1,2],[3,4],[5,6],[7,8]].
        # This is in contrast to a list containing split_path that may
        # look like [ [[1,2],[3,4]], [[5,6]] ]
        paths_flatten = []
        log.debug('value of reset paths is {}'.format(reset_base_paths))
        i = 0

        pg_merge = PathGraph()

        for path in self.get_all_paths():
            # split_path has the form [[0, 1], [2, 3], [4, 5]]
            # where each sub-list (e.g. [0,1]) has more or less the same size in bp
            split_path = self.split_path(path, target_length, 100)

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
            path_name = self.pg_base.get_path_name_of_node(path[0])
            for index, sub_path in enumerate(split_path):
                # prepare new PathGraph nodes
                attr = {'length': sum([self.pg_base.node[x]['length'] for x in sub_path]),
                        'name': "{}/{}".format(path_name, index),
                        'initial_path': sub_path}

                pg_merge.add_node(i, attr_dict=attr)

                merged_path.append(i)

                i += 1

            pg_merge.add_path(merged_path, name=path_name)
            paths_flatten.extend(split_path)

        if len(paths_flatten) == 0:
            log.warn("Nothing to reduce.")
            return None

        reduce_paths = paths_flatten[:]

        if len(reduce_paths) < 2:
            log.info("Reduce paths to small {}. Returning".format(len(reduce_paths)))
            return None

        reduced_matrix = reduce_matrix(self.matrix, reduce_paths, diagonal=True)

        # correct reduced_matrix
        start_time = time.time()
        corrected_matrix = iterativeCorrection(reduced_matrix, M=1000, verbose=False)[0]
        elapsed_time = time.time() - start_time
        log.debug("time iterative_correction: {:.5f}".format(elapsed_time))

        # update matrix
        self.matrix = corrected_matrix

        if self.pg_initial is None:
            self.pg_initial = self.pg_base
        self.pg_base = pg_merge

        if reset_base_paths is True:
            self.reset_pg_initial()

        assert len(self.pg_base.node.keys()) == self.matrix.shape[0], "inconsistency error"

    def reset_pg_initial(self):
        """
        This is a function called to reduce the information stored from original paths
        after merge_to_size is called for the first time.

        When contigs contains a lot of bins (e.g. when DpnII restriction enzyme was used)
        after the initial merge_to_size, there is no need to keep the detailed
        information of all the bins.

        Returns
        -------

        """

        if self.pg_initial is None:
            log.debug("can't reset. pg_initial is not set")
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
                attr = {'name': self.pg_initial.node[first_node]['name'],
                        'start': self.pg_initial.node[first_node]['start'],
                        'end': self.pg_initial.node[last_node]['end'],
                        'coverage': float(self.pg_initial.node[first_node]['coverage'] +
                                          self.pg_initial.node[last_node]['coverage']) / 2}
                assert attr['start'] < attr['end']
                attr['length'] = attr['end'] - attr['start']

                base_attr['initial_path'] = [node]
                self.pg_base.add_node(node, attr_dict=base_attr)
                pg_base.add_node(node, attr_dict=attr)
                cut_intervals.append((attr['name'], attr['start'], attr['end'], attr['coverage']))
            pg_base.add_path(path, name=path_name)

        self.pg_base = pg_base
        self.pg_initial = None

        # reset the hic matrix object
        self.hic.matrix = self.matrix
        self.hic.cut_intervals = cut_intervals
        self.hic.interval_trees, self.hic.chrBinBoundaries = \
            self.hic.intervalListToIntervalTree(self.hic.cut_intervals)

    def split_path(self, path, flank_length, recursive_repetitions, counter=0):
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
        >>> S.split_path(S.pg_base[0], flank_length, 30)
        [[0, 1], [2, 3], [4, 5]]

        # 5 bins, one internal is smaller than 20*75 (flank_length * tolerance) is skipped
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> flank_length = 20
        >>> S.split_path(S.pg_base[0], flank_length, 30)
        [[0, 1], [3, 4]]

        Get the flanks, and do not recursively iterate
        # 5 bins, one internal is smaller than 20*75 (flank_length * tolerance) is skipped
        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-0', 40, 50, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)
        >>> flank_length = 10
        >>> S.split_path(S.pg_base[0], flank_length, 1)
        [[0], [4]]

        """
        counter += 1
        if counter > recursive_repetitions:
            return []

        tolerance_max = flank_length * 1.25
        tolerance_min = flank_length * 0.75
        path_length = dict([(x, self.pg_base.node[x]['length']) for x in path])
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
                    interior = self.split_path(interior, flank_length, recursive_repetitions, counter=counter)
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
            raise HiCAssemblerException("Print no paths found\n")

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
            if len(v) < 10:
                log.warn('stats for distance {} contain only {} samples'.format(k, len(v)))
        return mean_path_length, sd_path_length, consolidated_dist_value

    @staticmethod
    @logit
    def find_best_permutation(ma, paths):
        """
        Computes de bandwidth(bw) for all permutations of rows (and, because
        the matrix is symmetric of cols as well).
        Returns the permutation having the minumum bw.

        The fixed pairs are not separated when considering
        the permutations to test

        Parameters
        ----------
        ma: HiCMatrix object
        paths: list of paths, containing paths that should not be reorder

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
#        for perm in itertools.permutations(enc_indices):
        for perm in itertools.permutations(paths):
            #log.debug("testing permutation {}".format(perm))
            for expnd in Scaffolds.permute_paths(perm):
                if expnd[::-1] in perm_list:
                    continue
                expand_indices = sum(expnd, [])
                mapped_perm = [mapping[x] for x in expand_indices]
                bw_value.append(Scaffolds.bw(ma[mapped_perm, :][:, mapped_perm]))
                perm_list.append(expnd)

        min_val = min(bw_value)
        min_indx = bw_value.index(min_val)

        return perm_list[min_indx]

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


        >>> cut_intervals = [('a', 0, 1, 1), ('b', 1, 2, 1), ('c', 2, 3, 1),
        ... ('d', 0, 1, 1), ('e', 1, 2, 1), ('f', 0, 1, 1)]

                                a  b  c  d  e  f
        >>> matrix = np.array([[0, 3, 0, 0, 0, 0], # a
        ...                    [0, 0, 3, 2, 2.5, 0], # b
        ...                    [0, 0, 0, 3, 0, 0], # c
        ...                    [0, 0, 0, 0, 1, 0], # d
        ...                    [0, 0, 0, 0, 0, 3], # e
        ...                    [0, 0, 0, 0, 0, 0]])# f

        >>> hic = get_test_matrix(cut_intervals=cut_intervals, matrix=matrix)
        >>> S = Scaffolds(hic)
        >>> list(S.get_all_paths())
        [[0], [1], [2], [3], [4], [5]]

        The weakest link for the hub 'b' (b-e) is removed
        >>> S.join_paths_max_span_tree(0, hub_solving_method='remove weakest', node_degree_threshold=5)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3], [4, 5]]

        >>> S = Scaffolds(hic)

        By setting the node_degree threshold to 3 the
        'b' node (id 1) is skipped.
        Thus, the path [0, 1, 2, 3] can not be formed.
        but the path [2, 3, 4, 5] is formed
        >>> S.join_paths_max_span_tree(0, hub_solving_method='remove weakest',
        ...                            node_degree_threshold=4)
        >>> list(S.get_all_paths())
        [[0], [1], [2, 3, 4, 5]]


        Test bandwidth permutation
        >>> S = Scaffolds(hic)
        >>> S.join_paths_max_span_tree(0, hub_solving_method='bandwidth permutation',
        ... node_degree_threshold=5)
        >>> list(S.get_all_paths())
        [[0, 1, 2, 3, 4, 5]]

        >>> S = Scaffolds(hic)
        >>> S.join_paths_max_span_tree(0, hub_solving_method='bandwidth permutation',
        ... node_degree_threshold=4)
        >>> list(S.get_all_paths())
        [[0], [1], [2, 3, 4, 5]]

        """
        matrix = self.matrix.copy()

        if confidence_score is not None:
            matrix.data[matrix.data <= confidence_score] = 0
        matrix.setdiag(0)
        matrix.eliminate_zeros()

        # get the node degree from the source  matrix.
        node_degree = dict([(x, matrix[x, :].nnz) for x in range(matrix.shape[0])])

        # define node degree threshold as the degree for the 90th percentile
        #node_degree_threshold = 1e10
        if node_degree_threshold is None:
            node_degree_threshold = np.percentile(node_degree.values(), 99)
        len_nodes_to_remove = len([x for x in node_degree.values() if x >= node_degree_threshold])
        log.info(" {} ({:.2f}%) nodes will be removed because they are below the degree "
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
        self.matrix[to_remove, :] = 0
        self.matrix[:, to_remove] = 0
        matrix[to_remove, :] = 0
        matrix[:, to_remove] = 0

        matrix.eliminate_zeros()
        self.matrix.eliminate_zeros()

        # get nodes that can be joined
        flanks = []  # a node that is either the first or last in a path
        singletons = []  # a node that has not been joined any other
        for path in self.get_all_paths():
            if len(path) == 1:
                singletons.extend(path)
            else:
                flanks.extend([path[0], path[-1]])

        edge_nodes = flanks + singletons
        import networkx as nx
        nxG = nx.Graph()
        for node in edge_nodes:
            nxG.add_node(str(node), name=self.pg_base.node[node]['name'])
        max_weight = matrix.data.max() + 1
        matrix = matrix.tocoo()
        for u, v, weight in zip(matrix.row, matrix.col, matrix.data):
            if u == v:
                continue
            if u in edge_nodes and v in edge_nodes:
                if u in self.pg_base[v]:
                    # u and v are in same path
                    nxG.add_edge(str(u), str(v), weight=max_weight)
                else:
                    nxG.add_edge(str(u), str(v), weight=weight)

        # add edges between same path nodes
        nlist = [node for node in nxG]
        for node in nlist:
            path = self.pg_base[int(node)]
            nxG.add_edge(str(path[0]), str(path[-1]), weight=max_weight)

        # compute maximum spanning tree
        nx.write_gml(nxG, "/tmp/mean_network.gml")
        nxG = nx.maximum_spanning_tree(nxG, weight='weight')
        nx.write_gml(nxG, "/tmp/mean_network_mst.gml")
        degree = np.array(dict(nxG.degree()).values())
        if len(degree[degree>2]):
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
                if int(u) in self.pg_base[int(v)]:
                    # skip same path nodes
                    continue
                else:
                    self.add_edge(int(u), int(v), weight=data['weight'])

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
                paths_to_check = [self.pg_base[int(node)]]
                seen.update(self.pg_base[int(node)])
                for v in G.adj[node]:
                    # only add paths that are not the same path
                    # already added.
                    if int(v) not in self.pg_base[int(node)]:
                        paths_to_check.append(self.pg_base[int(v)])
                        seen.update(self.pg_base[int(v)])

                # check, that the paths_tho_check do not contain hubs
                if len(paths_to_check) > 1:
                    check = True
                    """
                    for _path in paths_to_check:
                        for _node in _path:
                            if node_degree[_node] >= node_degree_threshold:
                                log.debug("degree {}".format(node_degree[_node]))
                                log.debug("{} in path {} is hub. Discarding {}".format(_node, _path, paths_to_check))
                                check = False
                                break
                    if check is True:
                        pass
                    else:
                        import ipdb;ipdb.set_trace()
                    """

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
                adj = sorted(G.adj[node].iteritems(), key=lambda (k, v): v['weight'])
                # remove the weakest edges but only if either of the nodes is not a hub
                for adj_node, attr in adj[:-2]:
                    log.info("Removing weak edge {}-{} weight: {}".format(node, adj_node, attr['weight']))
                    G.remove_edge(node, adj_node)
            if degree <= 2:
                break

        # based on the resulting maximum spanning tree, add the new edges to
        # the paths graph.
        for u, v, data in G.edges(data=True):
            if int(u) in self.pg_base[int(v)]:
                # skip same path nodes
                continue
            else:
                self.add_edge(int(u), int(v), weight=data['weight'])

    def add_edge(self, u, v, weight=None):
        """
        Adds and edge both in the reduced PathGraph (pg_base) and in the
        initial (non reduced) PathGraph.
        Parameters
        ----------
        u node index
        v node index
        weight weight

        Returns
        -------

        Examples
        --------

        >>> cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
        ... ('c-0', 30, 40, 1), ('c-1', 40, 50, 1), ('c-1', 50, 60, 1)]
        >>> hic = get_test_matrix(cut_intervals=cut_intervals)
        >>> S = Scaffolds(hic)

        >>> S.pg_base.path == {'c-0': [0, 1, 2, 3],
        ...                    'c-1': [4, 5]}
        True

        >>> S.add_edge(3, 4, weight=10)
        >>> S.pg_base.path == {'c-0, c-1': [0, 1, 2, 3, 4, 5]}
        True
        >>> S.pg_initial is None
        True

        Test case with merged bins
        >>> S = Scaffolds(hic)
        >>> S.merge_to_size(target_length=20, reset_base_paths=False)

        >>> S.pg_base.path == {'c-0': [0, 1], 'c-1': [2, 3]}
        True
        >>> S.pg_initial.path == {'c-0': [0, 1, 2, 3], 'c-1': [4, 5]}
        True

        >>> S.add_edge(1, 2, weight=10)
        >>> S.pg_base.path == {'c-0, c-1': [0, 1, 2, 3]}
        True

        >>> S.pg_initial.path == {'c-0, c-1': [0, 1, 2, 3, 4, 5]}
        True
        """
        path_added = False
        if self.pg_initial is not None:
            # get the initial nodes that should be merged
            initial_path_u = self.pg_base.node[u]['initial_path']
            initial_path_v = self.pg_base.node[v]['initial_path']

            for index_u, index_v in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
                # try all four combinations to join the `initial_path_u` and `initial_path_v`
                # For example, for [c, d, e], [x, y, z], the attempt is to
                # try to join (c, x), (c, z), (e, x), (e, z)
                # Because c, can be part of the larger path [a, b, c, d, e], any edge
                # containing 'e' can not be made and  the exception is raised.
                try:
                   self.pg_initial.add_edge(initial_path_u[index_u], initial_path_v[index_v], weight=weight)
                   self.pg_base.add_edge(u, v, weight=weight)
                   path_added = True
                   break
                except PathGraphEdgeNotPossible:
                    pass
            if path_added is False:
                raise HiCAssemblerException("Can't add edge between {}, {}".format(u, v))

        else:
           self.pg_base.add_edge(u, v, weight=weight)


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
                try:
                    self.pg_base.add_edge(u, v, weight=G[u, v])
                except:
                    import ipdb;ipdb.set_trace()
                for node in [u, v]:
                    for adj in G.adj(node):
                        G[node, adj] = 0

        _join_degree_one_nodes()

        import ipdb;ipdb.set_trace()


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
            raise HiCAssemblerException(message)

        # check if the node has less than 2 edges
        for node in [u, v]:
            if self.pg_base.degree(node) == 2:
                message = "Edge between {} and {} not possible,  contig {} " \
                          "is not a flaking node ({}, {}). ".format(u, v, node,
                                                                    self.pg_base.predecessors(node),
                                                                    self.pg_base.successors(node))
                raise HiCAssemblerException(message)

        # check if u an v are the two extremes of a path,
        # joining them will create a loop
        if self.pg_base.degree(u) == 1 and self.pg_base.degree(v) == 1:
            if v in self.pg_base[u]:
                message = "The edges {}, {} form a closed loop.".format(u, v)
                raise HiCAssemblerException(message)

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
