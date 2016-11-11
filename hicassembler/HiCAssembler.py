import numpy as np
from scipy.sparse import triu, lil_matrix
import networkx as nx
import gzip
import inspect
import itertools
import time
from hicexplorer.iterativeCorrection import iterativeCorrection
from hicexplorer.reduceMatrix import reduce_matrix
from functools import wraps

import logging
log = logging.getLogger("HiCAssembler")
log.setLevel(logging.DEBUG)

#log.basicConfig(format='%(levelname)s[%(funcName)s]:%(message)s', level=logging.DEBUG)

POWER_LAW_DECAY = 2**(-1.08)  # expected exponential decay at 2*distance

MERGE_LENGTH = 2000  # after the first iteration, contigs are merged as multiples of this length
MIN_LENGTH = 5000  # minimum contig or PE_scaffold length to consider
MIN_MAD = -0.5  # minimum zscore row contacts to filter low scoring bins
MAX_MAD = 50  # maximum zscore row contacts
MAX_INT_PER_LENGTH = 100  # maximum number of HiC pairs per length of contig
MIN_COVERAGE = 0.7
TEMP_FOLDER = '/tmp/'
ITER = 40
MIN_MATRIX_VALUE = 5
MERGE_LENGTH_GROW = 1.6

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
    def __init__(self, hic, min_mad=MIN_MAD, max_mad=MAX_MAD):
        """
        Prepares a hic matrix for assembly.
        It is expected that initial contigs or scaffolds contain bins
        of restriction fragment size.

        Parameters
        ----------
        hic : HiCMatrix object
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

        self.hic = hic

        # remove empty bins
        self.hic.maskBins(self.hic.nan_bins)
        del self.hic.orig_bin_ids
        del self.hic.orig_cut_intervals

        self.min_mad = min_mad
        self.max_mad = max_mad

        self.merged_paths = None
        self.iteration = 0

        self.remove_noise_from_matrix()

        # self.matrix is the working copy
        # of the matrix that is iteratively reduced
        self.matrix = self.hic.matrix.copy()

        # build scaffolds graph. Bins on the same contig are
        # put together.
        from hicassembler.Scaffolds import Scaffolds
        self.scaffolds_graph = Scaffolds(hic)

        # remove contigs that are too small
        self.scaffolds_graph.remove_small_paths(MIN_LENGTH)

        log.debug("Size of matrix is {}".format(self.hic.matrix.shape[0]))

        # compute initial N50
        self.N50 = []

        #self.matrix = hic.matrix.copy()
        #self.matrix.eliminate_zeros()

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

    def get_contig_order(self):
        super_scaffolds = []
        for path in self.scaffolds_graph.pg_initial.path.values():
            prev_contig_name = None
            prev_node_id = None
            scaffold = []
            for node in path:
                contig_name = self.scaffolds_graph.pg_initial.node[node]['name']
                if prev_contig_name is not None and contig_name != prev_contig_name:
                    scaffold.append((prev_contig_name, direction))
                else:
                    if node > prev_node_id:
                        direction = '+'
                    else:
                        direction = '-'
                prev_contig_name = contig_name
                prev_node_id = node
            scaffold.append((contig_name, direction))
            super_scaffolds.append(scaffold)
        return super_scaffolds, self.scaffolds_graph.get_all_paths()

    def assemble_contigs(self):
        """

        Returns
        -------

        """
        log.debug("Size of matrix is {}".format(self.matrix.shape[0]))
        for iteration in range(2):
            n50 = self.scaffolds_graph.compute_N50()
            self.N50.append(n50)

            # remove bins that are 1/3 the N50
            self.scaffolds_graph.remove_small_paths(n50 * 0.05)
            # the matrix may be of higher resolution than needed.
            # For example, if dpnII RE was used the bins could be
            # merged.
            reset_base_paths = True if iteration == 0 else False
            self.scaffolds_graph.merge_to_size(target_length=float(n50)/3, reset_base_paths=reset_base_paths)

            mean_len, std, stats = self.scaffolds_graph.get_stats_per_distance()
            self.scaffolds_graph.join_paths_max_span_tree(stats[2]['median'])


        return self.get_contig_order()

        # turn sparse matrix into graph
        self.scaffolds_graph.hic.diagflat(0)
        mat = self.scaffolds_graph.hic.matrix.copy()

        #mat.data[mat.data < stats[1]['median'] * 0.75] = 0
        #mat.eliminate_zeros()
        #mat.data = 1.0 / mat.data
        G = nx.from_scipy_sparse_matrix(mat)
        import ipdb;ipdb.set_trace()
        nx.write_gml(G, 'data/G.gml')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.title("test")
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True )
        plt.savefig("/tmp/G.png")

        mst = nx.minimum_spanning_tree(G, weight='weight')
        plt.title("test")
        nx.draw(mst, pos=nx.spring_layout(mst), with_labels=True)
        plt.savefig("/tmp/G_mst.png")

        exit()
        self.G = self.get_nearest_neighbors_2(self.paths, min_neigh=2, trans=False,
                                              max_int=max_int,
                                              threshold=confident_threshold)

        log.debug(((np.arange(1, ITER)**MERGE_LENGTH_GROW)*MERGE_LENGTH).astype(int))
        # for iter_num in range(1,ITER+1):
        for merge_length in ((np.arange(1, ITER)**MERGE_LENGTH_GROW)*MERGE_LENGTH).astype(int):
            self.iteration += 1
            max_int = self.reduce_to_flanks_and_center(flank_length=merge_length)
            #print_prof_data()
            if self.matrix.shape[0] < 2:
                log.info("No more to merge")
                self.compute_N50(merge_length)
                return
            # create a graph containing the nearest neighbors
#            import pdb;pdb.set_trace()
#            self.G = self.get_nearest_neighbors(self.paths, min_neigh=2, trans=False,
#                                                max_int = max_int,
#                                                threshold=confident_threshold)
            import ipdb;ipdb.set_trace()
            confident_threshold = stats[2][1]['median']
            if self.iteration == 1:
                confident_threshold *= 1.5
            if 1 < self.iteration <= 4:
                confident_threshold *= 1.1
            if self.iteration > 4:
                confident_threshold *= 0.7
            self.G = self.get_nearest_neighbors_2(self.paths, min_neigh=2, trans=False,
                                                  max_int=max_int,
                                                  threshold=confident_threshold)

            self.prev_paths = self.scaffolds_graph.get_all_paths()
            self.assemble_super_contigs(self.G, self.paths, max_int)
            orphans = [x for x in self.scaffolds_graph.get_all_paths() if len(x) == 1]
            log.info("in total there are {} orphans".format(len(orphans)))
            small = [x for x in self.scaffolds_graph.get_paths_length() if x <= merge_length]
            if len(small):
                log.info("in total there are {} scaffolds smaller than {}. "
                         "Median {}".format(len(small), merge_length, np.mean(small)))

            self.compute_N50(merge_length)
            if self.prev_paths == self.scaffolds_graph.get_all_paths():
                print "paths not changing. Returning after {} iterations, "\
                    "merge_length: {}".format(self.iteration, merge_length)
#                break
            # get the average length of all scaffolds that are larger
            # than the current merge_length
            """
            try:
                merge_length = int(np.median([x for x in
                                            self.scaffolds_graph.get_paths_length()
                                            if x >= merge_length]))/8
            except:
                merge_length *= 1.3

            if merge_length < prev_merge_length:
                merge_length = prev_merge_length * 2
            if merge_length == prev_merge_length:
                merge_length *= 2
            if merge_length > 5e6:
                log.info("Large merging value reached {}".format(merge_length))
                return
            prev_merge_length = merge_length
            """
        return

    def reduce_to_flanks_and_center(self, flank_length=20000):
        """
        removes the contigs that are inside of a path keeping only the
        flanking contigs up to the specified `flank_length`. The length
        of the contigs left at the flanks tries to be close to the
        flank_length argument. The purpose of this is to identify the orientation
        of a scaffold/contig by focusing on the sides.

        Parameters
        ----------
        flank_length : in bp

        Returns
        -------
        """
        log.info("reduce to flanks and center. flank_length: {}".format(flank_length))

        # flattened list of merged_paths  e.g [[1,2],[3,4],[5,6],[7,8]].
        # This is in contrast to a list containing flanks_of_path that may
        # look like [ [[1,2],[3,4]], [[5,6]] ]
        paths_flatten = []

        # list to keep the id of the new flanks_of_path after they
        # are merged. For example, for a flanks_of_path list e.g. [[0,1], [2,3]]]
        # after merging (that is matrix merging of the respective bins e.g 0 and 1)
        # the [0,1] becomes bin [0] and [2,3] becomes bin 1. Thus, merged_paths_id_map
        # has the value [[0,1]]. Further merged paths are appended as new lists
        # eg [[0,1], [2,3] .. etc. ]
        merged_paths_id_map = []
        i = 0
        contig_len = self.scaffolds_graph.get_contigs_length()
        for path in self.scaffolds_graph.get_all_paths():
            flanks_of_path = HiCAssembler.get_flanks(path, flank_length, contig_len, 6)
            if self.iteration > 1:
                # skip short paths after iteration 1
                if sum(contig_len[HiCAssembler.flatten_list(flanks_of_path)]) < flank_length*0.3:
                    continue
            merged_paths_id_map.append(range(i, len(flanks_of_path)+i))
            i += len(flanks_of_path)
            paths_flatten.extend(flanks_of_path)

#            print "in {} out {} ".format(path, flanks_of_path)
        if len(paths_flatten) == 0:
            print "[{}] Nothing to reduce.".format(inspect.stack()[0][3])
            return None, None

        reduce_paths = paths_flatten[:]
        # the matrix is to be reduced
        # but all original rows should be referenced
        # that is why i append the to remove to the
        # reduce_paths
        self.matrix = reduce_matrix(self.matrix, reduce_paths).tolil()
        if len(reduce_paths) < 2:
            log.info("Reduce paths to small {}. Returning".format(len(reduce_paths)))
            return None, None
        try:
            start_time = time.time()
            self.cmatrix = iterativeCorrection(self.matrix, M=30, verbose=True)[0]
            elapsed_time = time.time() - start_time
            log.debug("time iterative_correction: {:.5f}".format(elapsed_time))

            # put a high value to all edges belonging to an original path
            max_int = self.cmatrix.data.max()+1
        except:
            log.info("Reduce matrix is empty. Returning".format(len(reduce_paths)))
            return None, None

        self.cmatrix = self.cmatrix.tolil()
        for path in merged_paths_id_map:
            if len(path) > 1:
                # take pairs and replace the respective value
                # in the matrix by the masx int
#                import pdb;pdb.set_trace()
                for idx in range(len(path)-1):
                    # doing [(idx,idx+1),(idx+1,idx)]
                    # to keep the symmetry of the matrix.
                    # i.e. adding [1,2] and [2,1]
                    for c,d in [(idx,idx+1),(idx+1,idx)]:
#                        print "adding {},{}={}".format(path[c], path[d], max_int)
                        self.cmatrix[path[c], path[d]] = max_int
        self.cmatrix=self.cmatrix.tocsr()
        self.paths = paths_flatten
        self.merged_paths = merged_paths_id_map

        return max_int

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
