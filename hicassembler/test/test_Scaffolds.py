from scipy.sparse import csr_matrix
import numpy as np
import hicexplorer.HiCMatrix as HiCMatrix
from hicassembler.Scaffolds import Scaffolds


class TestClass:

    def __init__(self):
        self.hic = None
        self.S = None

    def setUp(self):
        cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
                         ('c-2', 0, 10, 1), ('c-2', 20, 30, 1), ('c-3', 0, 10, 1)]
        self.hic = get_test_matrix(cut_intervals=cut_intervals)
        self.S = Scaffolds(self.hic)

    def tearDown(self):
        pass

    def test_remove_small_paths(self):
        """
        Test the removal of paths.

        This causes a reduction of the matrix that requires a relabeling
        of the bins

        """

        # the original paths are c-0: [0, 1, 2], len 30,  c-2: [3, 4] len 20 and
        # c-3: [5] len(10)
        self.S.remove_small_paths(15)
        assert self.S.pg_base.path == {'c-0': [0, 1, 2], 'c-2': [3, 4]}

    def test_merge_to_size(self):
        """
        Two conditions are tested, one in which the paths are 'reset' and
        other in which the paths are not reset.

        Reset means that the merge information is kept as the primary data, otherwise
        the original paths and node data is kept and the merge refers to this data
        """

        # Test with two contigs and reset_base_paths
        cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
                         ('c-0', 30, 40, 1), ('c-1', 40, 50, 1), ('c-1', 50, 60, 1)]

        hic = get_test_matrix(cut_intervals=cut_intervals)
        S = Scaffolds(hic)

        # this is the initial list before merge
        assert list(S.get_all_paths()) == [[0, 1, 2, 3], [4, 5]]
        S.merge_to_size(target_length=20, reset_base_paths=False)
        assert list(S.get_all_paths()) == [[0, 1], [2, 3]]

        assert S.pg_base.node[0] == {'length': 20, 'initial_path': [0, 1], 'name': 'c-0/0'}
        assert S.pg_base.node[2] == {'length': 10, 'initial_path': [4], 'name': 'c-1/0'}
        assert S.matrix_bins.node[0] == {'start': 0, 'length': 10, 'end': 10, 'name': 'c-0', 'coverage': 1}

    def test_remove_small_paths_after_merge(self):
        cut_intervals = [('c-3', 0, 10, 1), ('c-0', 0, 10, 1), ('c-0', 10, 20, 1), ('c-0', 20, 30, 1),
                         ('c-2', 0, 10, 1), ('c-2', 20, 30, 1)]
        hic = get_test_matrix(cut_intervals=cut_intervals)
        S = Scaffolds(hic)
        S.merge_to_size(target_length=20, reset_base_paths=False)
        assert S.pg_base.path == {'c-3': [0], 'c-0': [1, 2], 'c-2': [3, 4]}
        # the path[0] should be removed and other paths
        # should be relabeled. Eg. path [1, 2], becomes [0, 1]
        S.remove_small_paths(15)
        assert S.pg_base.path == {'c-0': [0, 1], 'c-2': [2, 3]}
        assert S.pg_base.node[0] == {'initial_path': [1], 'length': 10, 'name': 'c-0/0'}
        # paths in the initial graph should not be changed
        print(S.matrix_bins.path)
        assert S.matrix_bins.path == {'c-3': [0], 'c-0': [1, 2, 3], 'c-2': [4, 5]}

    def test_reset_pg_initial(self):

        cut_intervals = [('c-0', 0, 10, 1), ('c-0', 10, 20, 2), ('c-0', 20, 30, 1),
                         ('c-0', 30, 40, 1), ('c-0', 40, 50, 1), ('c-0', 50, 60, 1)]
        hic = get_test_matrix(cut_intervals=cut_intervals)
        S = Scaffolds(hic)
        assert S.pg_base.path == {'c-0': [0, 1, 2, 3, 4, 5]}

        S.merge_to_size(target_length=20, reset_base_paths=True)
        # the path name is the same but contains fewer nodes
        assert S.pg_base.path == {'c-0': [0, 1, 2]}
        # the node names are the merge of the original start and end positions
        assert S.pg_base.node[2] == {'start': 40, 'length': 20, 'end': 60, 'name': 'c-0', 'coverage': 1.0}

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
