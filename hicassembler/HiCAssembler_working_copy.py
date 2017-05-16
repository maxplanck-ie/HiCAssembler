import numpy as np
from scipy.sparse import triu, lil_matrix
import networkx as nx
import gzip
import inspect
import itertools
import logging as log


from iterativeCorrection import iterativeCorrection
from reduceMatrix import reduce_matrix

debug = 1
save_time = 1

POWER_LAW_DECAY = 2**(-1.08) # expected exponential decay at 2*distance

MERGE_LENGTH = 2000 # after the first iteration, contigs are merged 
                  # as multiples of this length
MIN_LENGTH = 5000  # minimun contig or PE_scaffold length to consider
MIN_ZSCORE = -0.4 # minimun zscore row contacts to filter low scoring bins
MAX_ZSCORE = 5 # maximun zscore row contacts
#MAX_INT_PER_LENGTH = 'auto' # maximun number of HiC pairs per length of contig
# MAX_INT_PER_LENGTH is not used because it was replaced by the ASTAT
MAX_INT_PER_LENGTH = 100 # maximun number of HiC pairs per length of contig
MIN_COVERAGE = 0.7
TEMP_FOLDER = '/tmp/'
ITER = 40
MIN_MATRIX_VALUE = 5
MERGE_LENGTH_GROW = 1.6
MAX_ASTAT = 100

SIM = False
EXP = True
if SIM:
    MERGE_LENGTH = 2000 # after the first iteration, contigs are merged 
                      # as multiples of this length
    MIN_LENGTH = 1000  # minimun contig or PE_scaffold length to consider
    MAX_INT_PER_LENGH = 'auto' # maximun number of HiC pairs per length of contig
    #MAX_INT_PER_LENGTH = 1 # maximun number of HiC pairs per length of contig
    MIN_COVERAGE = 0.7
    TEMP_FOLDER = '/tmp/'
    ITER = 20
    MIN_MATRIX_VALUE = 20
    MERGE_LENGTH_GROW = 1.6

if EXP:
    MERGE_LENGTH = 2000 # after the first iteration, contigs are merged 
                      # as multiples of this length
    MIN_LENGTH = 5000  # minimun contig or PE_scaffold length to consider
    #MAX_INT_PER_LENGTH = 'auto' # maximun number of HiC pairs per length of contig
    MAX_INT_PER_LENGTH = 1 # maximun number of HiC pairs per length of contig
    MIN_COVERAGE = 0.8
    ITER = 50
    MIN_MATRIX_VALUE = 5
    MERGE_LENGTH_GROW = 1.3
    MAX_ASTAT = 50

import time
from functools import wraps

PROF_DATA = {}

def timeit(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        log.debug("{} took {}".format(fn.__name__, elapsed_time))
        return ret

    return with_profiling


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print "Function {} called {} ({}) times. ".format(fname, data[0], data[1])
        print 'Execution time max: %.3f, average: %.3f' % (max_time, avg_time)


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}


class HiCAssembler:
    def __init__(self, hic_matrix, PE_scaffolds_file):
        # the scaffolds list contains PE_scaffolds ids.
        # but for simplicity I refer to the initial PE_scaffolds
        # as contigs.

        # The list is modified with each iteration replacing its members
        # by lists. After two iterations a scaffold
        # list could look like: [[0],[1,2,3]]
        # which means that there are two scaffolds
        # one of those is composed of the contigs 1, 2 and 3

        # replace the diagonal from the matrix by zeros
        hic_matrix.diagflat(0)

        log.basicConfig(format='%(levelname)s[%(funcName)s]:%(message)s', level=log.DEBUG)
        self.PE_scaffolds_file = PE_scaffolds_file
        # the list if initialized with the ids as the matrix row numbers
        self.hic = hic_matrix
        self.matrix = hic_matrix.matrix.copy()
        self.matrix.eliminate_zeros()
        self.cmatrix = None
        self.scaffolds = Scaffolds(self.hic.cut_intervals)
        self.N50 = []
#        self.compute_N50()
        self.merged_paths = None
        # reads overlap file from SGA
        self.overlap = Overlap(None, self.scaffolds.id2contig_name)
#        self.overlap = Overlap(overlap_graph_file, self.scaffolds.id2contig_name)
#        self.overlap_graph = self.get_overlap_graph(overlap_graph_file)
        self.iteration = 0
        if debug:
            self.error = []
            self.bad_bw = 0
            self.good_bw = 0

    def assembleContigs_algorithm2(self):
        noise_level = np.percentile(self.hic.matrix.data, 70)
        log.debug("noise level set to {}".format(noise_level))
        self.hic.matrix.data= self.hic.matrix.data - noise_level
        self.hic.matrix.data[self.hic.matrix.data<0] = 0
        self.matrix.eliminate_zeros()
#        self.group = HiCAssembler.get_groups(self.hic.cut_intervals)
        # load PE data before filtering
        self.scaffolds_graph = Scaffolds(self.hic.cut_intervals)
        if self.PE_scaffolds_file:
            self.add_PE_scaffolds(self.PE_scaffolds_file, self.scaffolds_graph)
        # remove from the computation problematic contigs
        self.hic = self.mask_unreliable_rows()
#        self.group = HiCAssembler.get_groups(self.hic.cut_intervals)
        self.matrix = self.hic.matrix.copy()
        self.matrix.eliminate_zeros()
        start_time = time.time()
        log.info("starting iterative correction")
        self.cmatrix_orig = iterativeCorrection(self.matrix, M=30, verbose=True, tolerance=1e-4)[0]
        elapsed_time = time.time() - start_time
        log.debug("time iterative_correction: {:.5f}".format(elapsed_time))
        self.cmatrix = self.cmatrix_orig
        self.scaffolds = Scaffolds(self.hic.cut_intervals)
        self.compute_N50(0)
        # add PE scaffolds again, but this time with the new
        # id after filtering. The problem is that the scaffolds
        # have as id the row id from the hi_c matrix.
        # If the matrix changes the number of rows the id is affected
        # and the best way to keep the consistency is by resetting the 
        # scaffolds
        if self.PE_scaffolds_file:
            self.add_PE_scaffolds(self.PE_scaffolds_file, self.scaffolds)
        self.compute_N50(0)
        # needed for debugging, can be removed
        self.paths = [[x] for x in range(self.hic.matrix.shape[0])]
#        self.G3 = self.get_nearest_neighbors(self.paths, min_neigh=3, trans=True)

        merge_length = MERGE_LENGTH
        stats = self.compute_distances()
        print ((np.arange(1,ITER)**MERGE_LENGTH_GROW)*MERGE_LENGTH).astype(int)

        for merge_length in ((np.arange(1,ITER)**MERGE_LENGTH_GROW)*MERGE_LENGTH).astype(int):
#        for iter_num in range(1,ITER+1):
            self.iteration+=1
            max_int = self.reduce_to_flanks_and_center(
                flank_length=merge_length)
            print_prof_data()
            if self.matrix.shape[0] < 2:
                log.info("No more to merge")
                self.compute_N50(merge_length)
                return 
            # create a graph containing the nearest neighbors
#            import pdb;pdb.set_trace()
#            self.G = self.get_nearest_neighbors(self.paths, min_neigh=2, trans=False,
#                                                max_int = max_int,
#                                                threshold=confident_threshold)
            confident_threshold = stats[3][1]['median']
            if self.iteration==1:
                confident_threshold *= 1.5
            if 1 < self.iteration <= 4:
                confident_threshold *= 1.1
            if self.iteration>4:
                confident_threshold *= 0.7
            self.G = self.get_nearest_neighbors_2(self.paths, min_neigh=2, trans=False,
                                                  max_int=max_int,
                                                  threshold=confident_threshold)

            self.prev_paths = self.scaffolds.get_all_paths()
            self.assemble_super_contigs(self.G, self.paths, max_int)
            orphans = [x for x in self.scaffolds.get_all_paths() if len(x) == 1]
            log.info("in total there are {} orphans".format(len(orphans)))
            small = [x for x in self.scaffolds.get_paths_length() if x <= merge_length]
            if len(small):
                log.info("in total there are {} scaffolds smaller than {}. "
                             "Median {}".format(len(small), merge_length, np.mean(small)))
            
            self.compute_N50(merge_length)
            if self.prev_paths ==  self.scaffolds.get_all_paths():
                print "paths not changing. Returning after {} iterations, "\
                    "merge_length: {}".format(self.iteration, merge_length)
#                break
            # get the average length of all scaffolds that are larger
            # than the current merge_length
            """
            try:
                merge_length = int(np.median([x for x in
                                            self.scaffolds.get_paths_length()
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

    def mask_unreliable_rows(self, min_length=MIN_LENGTH,
                             min_zscore=MIN_ZSCORE,
                             max_zscore=MAX_ZSCORE,
                             min_coverage=MIN_COVERAGE,
                             max_int_per_length=MAX_INT_PER_LENGTH,
                             min_int_per_length=0.001):
        """
        identifies rows that are too small.  Those rows will be
        excluded from any computation.
        
        Params:
        ------
        hic: data structure returned by HiCMatrix
        min_length: in bp, minimum length of contig to be considered
        min_zscore: float, minimum number of contacts per row (as MAD zscore) to
            have in order to be considered a valid row
        max_zscore: float, maximum number of contacts per row (as MAD zscore) to
            have in order to be considered a valid row.

        Returns:
        -------
        None: The matrix object is edited in place

        """
        log.debug("filtering unreliable contigs")
        contig_id, c_start, c_end, coverage = zip(*self.hic.cut_intervals)

        # get scaffolds length
        length = np.array(c_end) - np.array(c_start) 

        # get the list of paths (usually just single contigs) that are too small
        small_list = np.flatnonzero(np.array(length) < min_length)
        # get the list of contigs that have too few interactions to other
        import ipdb;ipdb.set_trace()

        from hicCorrectMatrix import MAD
        row_sum = np.asarray(self.hic.matrix.sum(axis=1)).flatten()
        row_sum = row_sum - self.hic.matrix.diagonal()
        mad = MAD(row_sum)
        modified_z_score = mad.get_motified_zscores()

        few_inter = np.flatnonzero(modified_z_score < min_zscore)
        repetitive = np.flatnonzero(modified_z_score > max_zscore)

        # get list of contigs that have reduced coverage:
        low_cov_list = np.flatnonzero(np.array(coverage) < min_coverage)

        mask = np.unique(np.hstack([small_list, few_inter, low_cov_list]))

        mask_s = set(mask)
        log.debug("filtering paths with unreliable members")
        conn = [x for x in self.scaffolds_graph.get_all_paths() if len(x) > 1]
        for nodes in conn:
            overlap = list(mask_s.intersection(nodes))
            # only remove the whole scaffold if a significant part of
            # it does not pass filtering
            try:
                float(sum(length[overlap]))/(sum(length[nodes]))
            except Exception, e:
                log.error("Error message: {}".format(e))
                exit()
            if float(sum(length[overlap]))/(sum(length[nodes])) >= 0.75:
                # append to mask the whole path
                mask_s.update(nodes)
                try:
                    log.info("removing PE scaffold {} of size {} "
                             "containing {} parts".format([self.scaffolds_graph.id2contig_name[x] for x in nodes],
                                                          length[nodes],
                                                          len(nodes)))
                except IndexError:
                    # some contigs, that where
                    # previously filtered
                    # may not be found on the self.PE.id2contig_name dictionary
                    continue
            else:
                mask_s.difference_update(nodes)

        mask = list(mask_s)
        rows_to_keep = cols_to_keep = np.delete(range(self.hic.matrix.shape[1]), mask)
        log.info("Total: {}, small: {}, few inter: {}, low cover: {}, "
                 "repetitive: {}".format(len(contig_id), len(small_list),
                                         len(few_inter),
                                         len(low_cov_list),
                                         len(repetitive)))
        if len(mask) == len(ma_sum):
            print "Filtering to strong. All regions would be removed."
            exit(0)

        log.info("removing {} ({}%) low quality regions from hic matrix "
                 "whose length is smaller than {} and have in total less "
                 "than {} interactions to other contigs.\n\n"
                 "Keeping {} contigs.\n ".format(inspect.stack()[0][3], len(mask),
                                                 100 * float(len(mask))/self.matrix.shape[0],
                                                 min_length, min_int, len(rows_to_keep)))
        total_length = sum(length)
        removed_length = sum(length[mask])
        kept_length = sum(length[rows_to_keep])

        print "Total removed length:{} ({:.2f}%)\nTotal " \
            "kept length: {}({:.2f}%)".format(removed_length, 
                                              float(removed_length)/total_length,
                                              kept_length,
                                              float(kept_length)/total_length)
        # remove rows and cols from matrix
        hic.matrix = self.hic.matrix[rows_to_keep, :][:, cols_to_keep]
        hic.cut_intervals = [self.hic.cut_intervals[x] for x in rows_to_keep]

        # some rows may have now 0 read counts, remove them as well
        to_keep = np.flatnonzero(np.asarray(hic.matrix.sum(1)).flatten() > 0)
        if len(to_keep) != hic.matrix.shape[0]:
            print "removing {} extra rows that after filtering ended up "\
                "with no interactions".format(hic.matrix.shape[0] - len(to_keep))
            hic.matrix = self.hic.matrix[to_keep, :][:, to_keep]
            hic.cut_intervals = [self.hic.cut_intervals[x] for x in to_keep]

        return hic

    def compute_distances(self, merge_length=1):
        """
        takes the information from all contigs that are split
        or merged  and returns two values and two  vectors. The
        values are the average length used and the sd. 
        The vectors are: one containing the number of contacts found for such
        distance and the third one containing the normalized
        contact counts for different distances.
        The distances are 'contig' distance. Thus,
        if two contigs are next to each other, they are at distance 1

        """
        print "[{}] computing distances".format(inspect.stack()[0][3])

        dist_list = []
        contig_length = []
        # get all paths (connected componets) longer than 1
        conn = [x for x in self.scaffolds.get_all_paths() if len(x) > 1]
        label, start, end, coverage = zip(*self.hic.cut_intervals)
        # get the length of contigs in
        # conn
        if len(conn) == 0:
            message = "Print no joined contigs\n"
            message += "Can't determine distance estimations " \
                "for merge_length {} ".format(merge_length)
            raise HiCAssemblerException(message)

        if len(conn) > 200:
            # otherwise too many computations will be made
            # but 200 cases are enough to get 
            # an idea of the distributio of values
            conn = conn[0:200]
        if len(conn) < 20:
            message = "Contigs not long enough to compute a merge " \
                "of length {} ".format(merge_length)
            raise HiCAssemblerException(message)

        contigs = HiCAssembler.flatten_list(conn)
        contig_length = np.array(end)[contigs] - np.array(start)[contigs]
        mean_contig_length = np.mean(contig_length)
        sd_contig_length = np.std(contig_length)

        log.info("Mean contig length: {} sd{}".format(mean_contig_length,
                                                          sd_contig_length))

        # use all connected components to estimate distances
        dist_dict = dict()
        distc_dict = dict()
        for path in conn:
            # take upper trianbly of sumatrix containing selected path
            sub_m = triu(self.matrix[path,:][:,path], k=1, format='coo')
            sub_mc = triu(self.cmatrix[path,:][:,path], k=1, format='coo')
            # find counts that are one contig appart, two
            # contigs appart etc
            dist_list = sub_m.col - sub_m.row
            # tabulate all values that correspond
            # to distances
            for distance in np.unique(dist_list):
                if distance not in dist_dict:
                    dist_dict[distance] = sub_m.data[dist_list==distance]
                    distc_dict[distance] = sub_mc.data[dist_list==distance]
                else:
                    dist_dict[distance] = np.hstack([dist_dict[distance],
                                                     sub_m.data[dist_list==distance]])
                    distc_dict[distance] = np.hstack([distc_dict[distance],
                                                      sub_mc.data[dist_list==distance]])
                    
        # consolidate data:
        consolidated_dist_value =  dict()
        for k,v in dist_dict.iteritems():
            consolidated_dist_value[k] = {'mean': np.mean(v),
                                          'median': np.median(v),
                                          'max':np.max(v),
                                          'min':np.min(v),
                                          'len':len(v)}

        consolidated_distc_value =  dict()
        for k,v in distc_dict.iteritems():
            consolidated_distc_value[k] = {'mean': np.mean(v),
                                           'median': np.median(v),
                                           'max':np.max(v),
                                           'min':np.min(v),
                                           'len':len(v)}


        return (mean_contig_length, sd_contig_length,
               consolidated_dist_value, consolidated_distc_value)

    def compute_distances_bk(self, merge_length=1):
        """
        takes the information from all contigs that are split
        and returns three  vectors, one containing distances,
        other containing the number of contacts found for such
        distance and the third one containing the normalized
        contact counts.

        Also it returns tabulated values per distance.
        """
        print "[{}] computing distances".format(inspect.stack()[0][3])

        dist_list = []
        contact_list = []
        norm_contact_list = []
        contig_length = []
        if len(self.group) == 0:
            message =  "Print contigs not in groups"
            message = "Contigs not long enough to compute a merge " \
                "of length {} ".format(merge_length)
            raise HiCAssemblerException(message)
        contig_parts = self.group.values()
        contig_parts = [x for x in contig_parts if x[1]-x[0] > merge_length]
        if len(contig_parts) > 200:
            # otherwise too many computations will be made
            # but 200 cases are enough to get 
            # an idea of the distributio of values
            contig_parts = contig_parts[0:200]
        if len(contig_parts) < 20:
            message = "Contigs not long enough to compute a merge " \
                "of length {} ".format(merge_length)
            raise HiCAssemblerException(message)
        for g_start, g_end  in contig_parts:
            contig_id, c_start, c_end, extra = \
                zip(*self.hic.cut_intervals[g_start:g_end])

            if c_start[0] > c_start[-1]:
                # swap
                contig_id, c_start, c_end, extra = \
                    zip(*self.hic.cut_intervals[g_start:g_end][::-1])
                
                contig_range = range(g_start, g_end)[::-1]
            else:
                contig_range = range(g_start, g_end)

            sub_m = self.hic.matrix[contig_range,:][:,contig_range]
            sub_mc = self.cmatrix_orig[contig_range,:][:,contig_range]
            length = len(contig_id)
            # find the closest (smaller) number
            # that is a multiple of merge_length                
            length = length - length % merge_length
            merge_list = np.split(np.arange(length), length/merge_length)
            if merge_list > 1:
                sub_m = reduce_matrix(
                    sub_m[range(length),:][:,range(length)], merge_list)
                sub_mc = reduce_matrix(
                    sub_mc[range(length),:][:,range(length)], merge_list)
                # merge start and end data
                c_start = np.take(c_start, [x[0] for x in merge_list])
                c_end = np.take(c_end, [x[-1] for x in merge_list])
                
            # get the row and col indices of the upper triangle
            # of a square matrix of size M
            row, col = np.triu_indices(len(c_start), k=1)
            data = np.asarray(sub_m.todense()[(row, col)]).flatten()
            data_norm = np.asarray(sub_mc.todense()[(row, col)]).flatten()

            end_row = np.take(c_end, row)
            start_col = np.take(c_start, col) - 1
            dist_list.append(start_col - end_row)
            contact_list.append(data)
            norm_contact_list.append(data_norm)
            contig_length.append(np.array(c_end) - np.array(c_start))

        dist_list = np.hstack(dist_list)

        contact_list = np.hstack(contact_list)
        norm_contact_list = np.hstack(norm_contact_list)
        contig_length = np.hstack(contig_length)

        # get the average contig length
        # approximated to the closest multiple of 10.000
        contig_length_std = np.std(contig_length)
        contig_length_avg = np.mean(contig_length)
        contig_length_value = int(round(
                np.mean(contig_length)/contig_length_avg)*contig_length_avg)
        ## tabulate the distance information
        tab_dist = dict()
        tab_dist_norm = dict()
        dist_range = range(0, max(dist_list), contig_length_value)
        for index in range(len(dist_range)-1) :
            contacts = contact_list[(dist_list >= dist_range[index]) & 
                                    (dist_list < dist_range[index+1])]
            if len(contacts) > 0:
                tab_dist[dist_range[index]] = {'mean': round(np.mean(contacts),2),
                                  'median': np.median(contacts),
                                  'std': round(np.std(contacts),2),
                                  'len': len(contacts),
                                  'min': min(contacts),
                                  'max': max(contacts)}

                norm_contacts = norm_contact_list[(dist_list >= dist_range[index]) & 
                                    (dist_list < dist_range[index+1])]
                tab_dist_norm[dist_range[index]] = {'mean': round(np.mean(norm_contacts),2),
                                  'median': int(np.median(norm_contacts)),
                                  'std': round(np.std(norm_contacts),2),
                                  'len': len(norm_contacts),
                                  'min': int(np.min(norm_contacts)),
                                  'max': int(np.max(norm_contacts))}

        return(contig_length_value, dist_list, contact_list, tab_dist, tab_dist_norm,
               contig_length_std)

    def reduce_to_flanks_and_center(self, flank_length=20000):
        """
        removes the contigs that lie inside of a path
        keeping only the flanking contigs. The length
        of the contigs left at the flanks tries to be
        close to the tip_length argument
        """
        log.info("computing paths merge length {}".format(flank_length))

        to_keep = []
        path_list = self.scaffolds.get_all_paths()
        paths_flatten = [] # flattened list of merged_paths
                           # e.g [[1,2],[3,4],[5,6],[7,8]].
                           # This is in contrast to a list
                           # containing split paths that
                           # may look like [ [[1,2],[3,4]], [[5,6]] ]
        merged_paths = [] # list to keep the id of the new split paths 
                          # e.g. [[0,1], [2,3]]]
                          # which means path [1,2] in paths flatten has id 0
                          # and belongs to same path as [3,4] which has id 1
        i = 0
        
        contig_len = self.scaffolds.get_contigs_length()
        for path in path_list:
            
            split_path = HiCAssembler.get_flanks(path, flank_length, 
                                                 contig_len, 6)
            if self.iteration > 1:
                # skip short paths after iteration 1
                if sum(contig_len[HiCAssembler.flatten_list(split_path)]) < flank_length*0.3:
                    continue
            merged_paths.append(range(i, len(split_path)+i))
            i += len(split_path)
            paths_flatten.extend(split_path)

#            print "in {} out {} ".format(path, split_path_envenly)
        if len(paths_flatten) == 0:
            print "[{}] Nothing to reduce.".format(inspect.stack()[0][3])
            return None, None

        reduce_paths = paths_flatten[:]
        # the matrix is to be reduced
        # but all original rows should be referenced
        # that is why i append the to remove to the
        # reduce_paths
        self.matrix = reduce_matrix(self.hic.matrix, reduce_paths).tolil()
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
        for path in merged_paths:
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
        self.merged_paths = merged_paths
            
        return max_int


    def add_edge(self, u,v, **attr):
        """
        Does several checks before commiting to add an edge.
        
        First it checks whether the contigs to be joined share
        a significant number of contacts. Spurious
        interactions are discarted.

        Then tries to search in the overlap graph
        for evidence that may indicate 
        if two contigs may be together or
        one distance appart
        """
        if u == v:
            import pdb;pdb.set_trace()
        try:
            self.scaffolds.check_edge(u,v)
        except HiCAssemblerException, e:
#            print "edge exception {}".format(e)
            return

        """
        if u not in self.G3.neighbors(v):
            print "{} is not close neighbor to {} ({})".format(u,v,
                                                             self.G3.neighbors(v))
            return
        if v not in self.G3.neighbors(u):
            print "{} is not close neighbor to {} ({})".format(v,u,
                                                             self.G3.neighbors(u))
            return
        """
        """
        if self.hic.matrix[u,v] <= 5:
            message = "number of shared pairs to join {} and {} "\
                  "too low {}, "\
                  "skipping. ".format(u, v, self.hic.matrix[u,v])
            raise HiCAssemblerException(message)
        """
        ##### DEBUG LINES
        if debug and abs(u - v) != 1:
            self.error.append([u, v])
            log.debug("[{}] an error joining {} and {} has been made\n {} \n{}"
                          "".format(inspect.stack()[0][3], u, v, attr, inspect.stack()[1][3]))

#            print "[{}] an error joining {} and {} has been made\n {} \n{}"\
#                "".format(inspect.stack()[0][3], u, v, attr, inspect.stack()[1][3])
        if debug and abs(u - v) > 20:
            print "BAD error made {},{}".format(u,v)
        """
            print "neighbors u {}, neighbors v {}".format(self.G3[u], self.G3[v])
            def print_part_pat(v):
                path_v = self.scaffolds.get_path_containing_source(v)
                v_index = path_v.index(v)
                if len(path_v)>2:
                    if v_index == 0:
                        v_nei = v_index + 2
                    else:
                        v_nei =  v_index
                        v_index -= 2
                    pp = path_v[v_index:v_nei]
                else:
                    pp = path_v
                try: 
                    print "path {}: {}".format(v, pp)
                except:
                    pass
            print_part_pat(u)
            print_part_pat(v)
#            import pdb;pdb.set_trace()
        """
        ##### END DEBUG LINES

        # checks if an edge may be solved/refined by looking at the 
        # overlap graph
        attr = self.overlap.test_overlap(u,v, **attr)
        self.scaffolds.add_edge(u, v, **attr)

        #if debug:
        #    print "[{}] adding edge {} {}".format(inspect.stack()[0][3], u, v)

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
            import pdb;pdb.set_trace()
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
        

    @staticmethod
    def argsort(seq):
        """
        returns the indeces of an ordered python list
        """
        return sorted(range(len(seq)), key=seq.__getitem__)


    @staticmethod
    def get_fixed_paths(G, neighbors, max_int):
        """
        returns a list of tuples
        containing pairs of contigs
        who already have been found
        as being next to each other
        When computing all permutations
        to obtain the smallest
        bw, such pairs should be kept
        together.
        """
 
        # find which edges have the max_int value
#        fixed_paths = []
#        seen = []
        new_G = nx.Graph()
        for edge in G.edges(neighbors, data=True):
            u, v = edge[0:2]
            if u not in neighbors or v not in neighbors:
                continue
#            if u in seen or v in seen:
#                continue
#            seen.extend([u, v])
            weight = edge[2]['weight']
            if weight == max_int:
                new_G.add_edge(u,v)
                
#               fixed_paths.append((u,v))
        return nx.connected_components(new_G)



    def assemble_super_contigs(self, G, paths, max_int, add_first_neighbors=True):
        """Mon, 14 Oct 2013 22:10:46 +0200
        Uses networkx to find shortest path
        joining connected - components

        Params:
        ------
        G: a networkx object

        Returns:
        -------

        super_contigs: list of lists. Each list member
        contains the sequence of sequential contigs
        """
        log.info("Assembling contigs")
        for node in G.nodes(data=True):
            G.add_node(node[0], label="{}({})".format(paths[node[0]],node[0]), 
                       id=node[0])
        nx.write_gml(G,
                     "./prepros_g_{}.gml".format(self.iteration))

        cen = nx.degree(G)
        high_deg, degree = zip(*cen.iteritems())
        # sort hubs by increassing degree and filter by 
        # degree > 2
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
                    if self.overlap.have_direct_overlap(u,v):
                        to_keep.append(n)
                if len(to_keep)>2:
                    # suspicios
                    to_keep = []
                elif len(to_keep):
                    import pdb;pdb.set_trace()
                    log_m = "\n".join(["{}".format(G[node][x]) for x in to_keep])
                    log.debug("Found direct overlap for a neighbor of {}. "
                                  "Keeping neighbor {}".format(node, log_m))

                log.debug("High degree node {} {} removed "
                              "from graph".format(node,
                                                  G[node]))
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
            if G.degree(node) <=2:
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
            if 2==2:
#            else:
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
                            print "WRONG bw permutation of node: {} is: "\
                                "{}\n{}\n{}\n{}".format(
                                node,
                                bw_order,
                                [self.paths[x] for x in bw_order],
                                [int(self.cmatrix[bw_order[x], bw_order[x+1]]) 
                                 for x in range(len(bw_order)-1)],
                                [int(self.matrix[bw_order[x], bw_order[x+1]]) 
                                 for x in range(len(bw_order)-1)])
                            self.bad_bw += 1
#                            import pdb;pdb.set_trace()
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
            u,v = [paths[x][0] for x in edge]

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


    def compute_N50(self, merge_length):

        length = np.sort(np.array(self.scaffolds.get_paths_length()))
        length = length[length>200]
        cumsum = np.cumsum(length)
        for i in range(len(length)):
            if cumsum[i] >= float(cumsum[-1]) / 2:
                break
        try:
            itera = self.iteration
        except:
            itera = 0
        log.info("iteration:{}\tN50: {}\tMax length: {} " \
              "No. scaffolds {}".format(itera, length[i],
            length[-1], len(self.scaffolds.get_all_paths())))

        if debug:
            error = [abs(x[1]-x[0]) for x in self.error]
            log.debug("bad bw {} ({:.2f} from {} good)errors {}".format(
                    self.bad_bw, float(self.bad_bw)/(self.good_bw+self.bad_bw+0.01),
                    self.good_bw, error))
            self.bad_bw = self.good_bw = 0
        self.N50.append((itera, length[i], merge_length,  self.scaffolds.get_all_paths()))
        return length[i]

    @staticmethod
    def flatten_list(alist):
        """
        given a list of list of list, returns a list
        For example: given [1,[[2]], [3,4]]
        returns [1, 2, 3 4]
        This is a recursive function.

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
    def get_groups(contig_list):
        """
        Takes a list of contigs in the format
        of the HiCMatrix object (HiCMatrix.cut_intervals) and
        returns a dictionary contaning the range of
        contigs that span in the list

        Params:
        ======
        contig_list: The format is [(contig_id, start, end, flag), (contig_id2 ....)]

        Return:
        ======

        group: dictionary containing as key the contig_id, and as value
               a duple with the start and end indices of the group.
        """
        contig_id, c_start, c_end, extra = zip(*contig_list)

        prev = None
        in_group = False
        group = dict()
        for index in range(len(contig_id)):
            if contig_id[index]==prev and in_group is False:
                in_group = True
                group_start=index - 1
            elif contig_id[index]!=prev and in_group is True:
                in_group = False
                group_end = index
                group[contig_id[group_start]]=(group_start, group_end)
            prev = contig_id[index]

        return group

    @staticmethod
    def rmsd(ma):
        """
        NOT COMPLETED
        Computes the root mean square of the given order
        with respect to the decreasing order of the values
        """

        #get for each node t
        ma = triu(ma, k=1, format='coo').astype(float)
        ma.data *= (ma.col - ma.row)**1.1
        return ma.sum()

    @staticmethod
    def bw(ma):
        """
        Computes my version of the bandwidth of the matrix
        which is defined as \sum_i\sum_{j=i} log(M(i,j)*(j-i))
        The matrix that minimizes this function should have
        higher values next to the main diagonal and 
        decreasing values far from the main diagonal
        """
        ma = triu(ma, k=1, format='coo').astype(float)
        ma.data *= (ma.col - ma.row)
        return ma.sum()

    @staticmethod
    def expand(alist):
        """
        To reduce the number of permutations tested
        some pairs are fixed, meaning that they
        should always be together. Thus, the pair
        is treated like one item and the number of 
        permutations is reduced. However, each pair
        can be as either (a,b) or (b,a). 
        This code expands the original list
        adding pairs in both directions

        >>> HiCAssembler.expand([',1,2'])
        [[1, 2], [2, 1]]
        >>> HiCAssembler.expand([',1,2', 3])
        [[1, 2, 3], [2, 1, 3]]
        >>> HiCAssembler.expand([1,2, ',3,4',5])
        [[1, 2, 3, 4, 5], [1, 2, 4, 3, 5]]
        >>> HiCAssembler.expand([1,',2,3', 4, ',5,6'])
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5], [1, 3, 2, 4, 5, 6], [1, 3, 2, 4, 6, 5]]
        >>> HiCAssembler.expand([',1,2', ',3,4', ',5,6'])[0:3]
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5], [1, 2, 4, 3, 5, 6]]
        >>> HiCAssembler.expand([1,2,3])
        [[1, 2, 3]]
        """
        ret = []
        expand_vals = []
        positions = [-1]
        # identify positions to expand
        for idx, val in enumerate(alist):
            new_list = []
            try:
                if val[0]==",":
                    path = [int(x) for x in val.split(",")[1:]]
                    expand_vals.append(path)
                    positions.append(idx)
            except (IndexError, TypeError):
                # the exception happens when val is an int
                continue
        if not len(expand_vals):
            return [alist]
        #expand the list
#        for value in range(2**len(expand_vals)):
        for prod in itertools.product(*[(0,1)]*len(expand_vals)):
            new_list = []
            for idx, pair in enumerate(expand_vals):
#                print "taking: {},{}:{}".format(positions[idx]+1,positions[idx+1],
#                                             alist[positions[idx]+1:positions[idx+1]])
                new_list += alist[positions[idx]+1:positions[idx+1]]
#                print "idx {}, value {}, pair {}".format(idx, prod,pair)
                # a bitwise operator is used to decide
                # if the pair should be flipped or not
                pair = pair if prod[idx] == 0 else pair[::-1]
                new_list += pair
#            print "{}:len{}, positions {} idx {}".format(alist,len(alist), positions, idx)
            if len(alist) > idx+1:
                new_list += alist[positions[idx+1]+1:]
            ret.append(new_list)
        return ret 

    @staticmethod
    def encode_fixed_paths(indices, fixed_paths):
        """
        >>> HiCAssembler.encode_fixed_paths([1,2,3,4], [(2,4)])
        [1, 3, ',2,4']
        """
        fixed_paths_list = [ item for sublist in fixed_paths 
                             for item in sublist]
        new_list = [x for x in indices if x not in fixed_paths_list]
        seen = []
        for pair in fixed_paths:
            if pair[0] in seen or pair[1] in seen:
                message = "Pair {} invalid. One of the members of the pair " \
                    "has already been used. Seen indices {}".format(pair, seen)
                raise HiCAssemblerException(message)
            seen += pair
            new_list += [','+ ','.join([str(x) for x in pair])]
        return new_list

    @staticmethod
    def permute(ma, indices, fixed_paths):
        """
        Computes de bw for all permutations of rows (and, because
        the matrix is symetric of cols as well).
        Returns the permutation having the minumun bw.

        The fixed pairs are not separated when considering
        the permutations to test

        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix(np.array([[12,5,3,2,0],[0,11,4,1,1],
        ... [0,0,9,6,0], [0,0,0,10,0], [0,0,0,0,0]]))
        >>> HiCAssembler.permute(A, [2,3,4], [(3,4)])
        [2, 3, 4]
        >>> HiCAssembler.permute(A, [2,3,4], [])
        (3, 2, 4)
        """
        ma = ma[indices,:][:,indices]
        enc_indices = HiCAssembler.encode_fixed_paths(indices, fixed_paths)
        # mapping from 'indices' to new matrix id
        mapping = dict([(val, idx) for idx, val in enumerate(indices)])
        bw_value = []
        perm_list = []
        for perm in itertools.permutations(enc_indices):
            for expnd in HiCAssembler.expand(perm):
                if expnd[::-1] in perm_list:
                    continue
                mapped_perm = [mapping[x] for x in expnd]
                bw_value.append(HiCAssembler.bw(ma[mapped_perm,:][:,mapped_perm]))
                perm_list.append(expnd)

        min_val = min(bw_value)
        min_indx = bw_value.index(min_val)
#        return (perm_list[min_indx],perm_list,  bw_value)
        return perm_list[min_indx]

    @staticmethod
    def get_flanks(path, flank_length, contig_len,
                   recursive_repetitions, counter=0):
        """
        Takes a path are returns the flanking regions
        plus the inside. This is a recursive function
        and will split the inside as many times 
        as possible, stoping when 'recursive_repetitions'
        have been reached

        The flank lengh is set to 2000, thus, groups of two should be
        selected
        >>> HiCAssembler.get_flanks([1,2,5,3,4], 2000,
        ... np.array([1000,1000,1000,1000,1000,2000]), 3)
        [[1, 2], [5], [4, 3]]

        Same as before, but now I set id 5 to be smaller. 
        Now 5 should be skipped as is not at least flank_length*0.75
        >>> HiCAssembler.get_flanks([1,2,5,3,4], 2000,
        ... np.array([1000,1000,1000,1000,1000,800]), 3)
        [[1, 2], [4, 3]]

        Get the flanks, and do not recursively iterate
        >>> HiCAssembler.get_flanks([1,2,5,3,4], 1000,
        ... np.array([1000,1000,1000,1000,1000,1000]), 1)
        [[1], [4]]

        Get the flanks, and iterate twice
        >>> HiCAssembler.get_flanks([1,2,5,3,4], 1000,
        ... np.array([1000,1000,1000,1000,1000,1000]), 2)
        [[1], [2], [3], [4]]
        """
        counter += 1
        if counter > recursive_repetitions:
            return []

        tolerance_max = flank_length * 1.25
        tolerance_min = flank_length * 0.75
        path_length_sum = sum(contig_len[path])
        interior = []
        flanks = []
        def flank(path):
            flank = []
            for x in path:
                flank_sum = sum(contig_len[x] for x in flank)
                if flank_sum > tolerance_max:
                    break
                elif tolerance_min <= flank_sum <= tolerance_max:
                    break
                flank.append(x)
            return flank

#        if path == [1,2]:
#            import pdb;pdb.set_trace()
        if len(path) == 1:
            if contig_len[path[0]]>tolerance_min or counter == 1:
                flanks = [path]
        elif path_length_sum < 2*flank_length*0.75:
            path_half = len(path)/2
            left_flank = path[0:path_half]
            right_flank = path[path_half:][::-1]
            flanks.extend([left_flank, right_flank])
        else:
            left_flank=flank(path)
            right_flank=flank(path[::-1])
            over = set(left_flank).intersection(right_flank)
            if len(over):
                # remove overlap
                left_flank = [x for x in left_flank if x not in over]

            if len(left_flank) == 0 or len(right_flank) == 0:
                path_half = len(path)/2
                left_flank = path[0:path_half]
                right_flank = path[path_half:][::-1]

            interior = [x for x in path 
                        if x not in HiCAssembler.flatten_list(left_flank+right_flank)]

            if len(interior):
                interior = HiCAssembler.get_flanks(interior, flank_length, contig_len, 
                                      recursive_repetitions, counter=counter)
            if len(left_flank): flanks.append(left_flank)
            if len(interior): flanks.extend(interior)
            if len(right_flank): flanks.append(right_flank)

        try:
            if len(left_flank) == 0 or len(right_flank) == 0:
                import pdb;pdb.set_trace()
        except:
            pass
        return flanks

    def add_PE_scaffolds(self, scaffolds_file, scaffolds, format='sga'):
        """
        this code parses the file that contains
        the information to construct PE_scaffolds
        from contigs.

        This is saved in a .scaf file
        The format looks like this
        but I don't have an explanation:
        contig-863220\tcontig-513939,-34,18.1,1,0,D\tcontig-674650,133,34.1,1,0,D
        
        I imagine the format means:

        contig-863220 -> contig-513939 [d=-34, e=18.1, join=end-start]

        The last part, which is 1,0,D in both cases I thing it
        explains how to join the contigs: 
        
        ------->    ------>
         source      target

         --> --> 0,0
         --> <-- 0,1
         <-- <-- 1,0
         <-- --> 1,1

        """
        print "[{}] reading scaffolding data based on PE {} ".format(
            inspect.stack()[0][3], scaffolds_file.name)

        contig_name2id = dict([(scaffolds.id2contig_name[x],x) 
                               for x in range(len(scaffolds.id2contig_name))])

        def get_contig_id(contig_name,  head, scaffolds):
            if contig_name in scaffolds.split_contigs:
                # need to decide if the link should
                # be made with the first or with the
                # last member of the group
                first_group = scaffolds.split_contigs[contig_name][0]
                last_group  = scaffolds.split_contigs[contig_name][-1]

                # This check is needed because the contig can be 
                # inverted
                if scaffolds.contig_G.node[first_group]['start'] > \
                        scaffolds.contig_G.node[last_group]['start']:
                    # invert the order if the group members
                    # are ordered from last to first
                    first_group, last_group = last_group, first_group

                if head is True:
                        contig_id = last_group
                else:
                    contig_id = first_group
            else:
                contig_id = contig_name2id[contig_name]
            return contig_id

        scaffolds_file.seek(0)
        for line in scaffolds_file:
            path_raw = line.strip().split('\t')
            if len(path_raw) == 1:
                # frequently, there is just a contig id.
                # E.g. the line may look like:
                # contig-679531
                continue
            prev_contig_name = path_raw.pop(0)
            if prev_contig_name not in scaffolds.id2contig_name:
                # exclude means not to be shown in the 
                # scaffold path
                contig_name2id[prev_contig_name] = \
                    scaffolds.add_contig(prev_contig_name,
                                              **{'exclude':True})
            for contig_info in path_raw:
                contig_name, distance, error, source, \
                    target, D = contig_info.split(',')
                distance = int(distance)
                error=float(error)
                source= int(source)
                target=int(target)

                if contig_name not in scaffolds.id2contig_name:
                    # exclude means not to be shown in the 
                    # scaffold path
                    contig_name2id[contig_name] = \
                        scaffolds.add_contig(contig_name,
                                                  **{'exclude':True})

                """                
                --> --> 0,0
                --> <-- 0,1
                <-- --> 1,1
                <-- <-- 1,0
                """
                if source == 0 and target == 0:
                    # join head with tail
                    u_head = True
                    v_head = False
                elif source == 0 and target == 1:
                    # join head with head
                    u_head = True
                    v_head = True
                elif source == 1 and target == 0:
                    # join tail with head
                    u_head = False
                    v_head = True
                elif source == 1 and target == 1:
                    # join tail with tail
                    u_head = False
                    v_head = False

                u = get_contig_id(prev_contig_name, u_head, scaffolds)
                v = get_contig_id(contig_name, v_head, scaffolds)
                """
                if debug:
                    u_group = self.group[prev_contig_name] \
                        if prev_contig_name in self.group else []
                    v_group = self.group[contig_name] \
                            if contig_name in self.group else []
                    if abs(u-v) > 1:
                        log.debug("WARNING: PE scaffolding wrong u,v "
                                      "{},{} in group {} {}, "
                                      "source {}, target {} \n{}".format(
                                u,v, 
                                u_group, v_group, source,
                                target, line))
                        
                """
                try:
                    scaffolds.add_edge(u, v, no_check=True,
                                       iteration=0,
                                       distance=distance, error=error,
                                       source_direction=source,
                                       target_direction=target, D=D,
                                       PE_scaffold=True)
                except Exception, e:
                    print e
                    import pdb;pdb.set_trace()

                prev_contig_name = contig_name

        # remove the nodes that were labeled as excluded
        excluded = [x[0] for x in scaffolds.contig_G.nodes(data=True) if 'exclude' in x[1]]
        for contig in excluded:
            pred = scaffolds.contig_G.predecessors(contig)
            suc = scaffolds.contig_G.successors(contig)
            if pred:
                scaffolds.contig_G.remove_edge(pred[0], contig)
            if suc:
                scaffolds.contig_G.remove_edge(contig, suc[0])
            if pred and suc:
                scaffolds.contig_G.add_edge(pred[0], suc[0])
            scaffolds.contig_G.remove_node(contig)


class Scaffolds:
    """
    This class is a place holder to keep track of the iterative scaffolding.
    The underlying data structure is a special directed graph that does 
    not allow more than two edges per node.

    The list of paths in the graph (in the same order) is paired with 
    the rows in the HiC matrix.

    """
    def __init__(self, cut_intervals):
        # initialize the list of contigs as a graph with no edges
        self.paths = None
        self.id2contig_name = []
        self.contig_G = nx.DiGraph()
        self.contig_length = self.get_contig_graph(cut_intervals)
        self.split_contigs = None

    def get_contig_graph(self, cut_intervals):
        """ uses the hic information for each row (cut_intervals)
        to initialize a graph in which each node corresponds to 
        a contig.
        """
        from collections import defaultdict
        label, start, end, coverage = zip(*cut_intervals)
        length = np.array(end) - np.array(start)
        prev_label = None
        i = 1
        self.split_contigs = defaultdict(list)
        for index in range(len(cut_intervals)):
            if length[index] < MIN_LENGTH:
                continue

            # if the id of the contig is identical
            # to the previous contig, add and edge.
            # Contigs with the same ID are those divided by restriction fragment bins
            # and should be joined with and edge
            if prev_label == label[index]:
                label_name = "{}-{}".format(label[index],i)
                prev_contig_id = contig_id
                if i == 1:
                    # insert first group member
                    self.split_contigs[label[index]].append(contig_id)
                i += 1
            else:
                i = 1
                label_name = label[index]
                prev_label = label_name
            # contig id is the row index in the hic matrix
            attr = {'name':label_name,
                    'start': start[index],
                    'end': end[index],
                    'coverage': coverage[index],
                    'length':length[index]}
            contig_id = self.add_contig(label[index], **attr)
            if i > 1:
                self.split_contigs[label[index]].append(contig_id)
                self.contig_G.add_edge(prev_contig_id, contig_id,
                                       iteration=0,
                                       contig_part=True,
                                       source_direction=0,
                                       target_direction=1)

        return length


    def add_contig(self, contig_name, **attr):
        self.id2contig_name.append(contig_name)
        contig_id = len(self.id2contig_name) - 1
        if 'name' not in attr.keys():
            attr['name'] = contig_name
        attr['id'] = contig_id
        attr['label'] = '{}'.format(contig_id)
        self.contig_G.add_node(contig_id, **attr)
        self.paths = None
        return contig_id

    def get_flanks(self, flatten=False):
        """
        >>> S = Scaffolds([('c-0', 0, 1, 1)])
        >>> S.contig_G.add_path([1, 2, 3, 4])
        >>> S.contig_G.add_path([5, 6])
        >>> S.get_flanks()
        [(0, 0), (1, 4), (5, 6)]
        >>> S.get_flanks(flatten=True)
        [0, 0, 1, 4, 5, 6]
        """
        flanks = [(x[0],x[-1]) for x in self.get_all_paths()]
        if flatten:
            flanks = [x for pair in flanks for x in pair]
        return flanks
            
    def get_all_paths(self):
        """Returns all paths in the graph.
        This is similar to get connected components
        but in this case, the order of the returned  paths 
        represents scaffolds of contigs
        
        """
        if self.paths:
            return self.paths

        seen = []
        paths = []
        # the sorted function is to 
        # get reproducible results because
        # networkx uses a dict to store each
        # node and the order in which they are 
        # returned could vary
        for v, data in sorted(self.contig_G.nodes(data=True)):
            if v not in seen:
                path = self.get_path_containing_source(v)
                paths.append(path)
                seen.extend(path)

        self.paths = paths

        return paths

    def get_path_containing_source(self, v):
        """
        For a node v in a directed graph, find its
        successor, then the successor of the successor
        until no more successors are found. Then a similar
        iteration is repeteat for the predecesors.
        
        Because each node has at most on successor and one
        predecessor, the resulting path is ordered.
        In other words, if nodes a, b, c, and d are 
        connected as a --> b --> c --> d, the result
        of this function is [a, b, c, d].

        Parameters
        ----------

        v : a node id from the graph

        Returns
        -------

        A list

        >>> S = Scaffolds([('c-0', 0, 1, 1)])
        >>> S.get_path_containing_source(0)
        [0]
        >>> S.contig_G.add_path([1, 2, 3, 4])
        >>> S.get_path_containing_source(4)
        [1, 2, 3, 4]
        """
        source = v
        path = [v]
        # append all successors from node v
        while True:
            s = self.contig_G.successors(v)
            if s and s[0] not in path:
                path.append(s[0])
                v = s[0]
            else:
                break

        # get all predecesors from node v
        v = source
        while True:
            p = self.contig_G.predecessors(v)
            if p  and p[0] not in path:
                path.insert(0, p[0])
                v = p[0]
            else:
                break
        return path

    def get_paths_length(self):
        # get PE_scaffolds length
        path_lengths = []
        for scaff in self.get_all_paths():
            path_lengths.append(sum([self.contig_G.node[x]['length'] 
                                     for x in scaff]))

        return path_lengths

    def get_contigs_length(self):
        return self.contig_length
            
    def remove_paths(self, mask_list):
        paths = self.get_all_paths()
        # translate path indices in mask_list back to contig ids
        # and merge into one list using sublist trick
        paths_to_remove = [paths[x] for x in mask_list]
        contig_list = [item for sublist in paths_to_remove for item in sublist]
 
        self.contig_G.remove_nodes_from(contig_list)
        # reset the paths
        self.paths = None


    def has_edge(self,u,v):
        return self.contig_G.has_edge(u,v) or self.contig_G.has_edge(v,u)

    def check_edge(self, u,v):
        # check if the edge already exists
        if self.has_edge(u,v):
            message = "Edge between {} and {} already " \
                      "exists".format(u, v)
            raise HiCAssemblerException(message)

            
        # check if the node has less than 2
        # edges
        for node in [u, v]:
            if self.contig_G.degree(node) == 2:
                message =  "Edge between {} and {} not possible,  contig {} " \
                          "is not a flaking node ({}, {}). ".format(u, v, node,
                                                      self.contig_G.predecessors(node),
                                                      self.contig_G.successors(node))
                raise HiCAssemblerException(message)

        # check if u an v are the two extremes of a path,
        # joining them will create a loop
        if self.contig_G.degree(u) == 1 and self.contig_G.degree(v) == 1:
            if v in  self.get_path_containing_source(u):
                message = "The edges {}, {} form a closed loop.".format(u, v)
                raise HiCAssemblerException(message)


    def add_edge(self, u, v, **attr):
        """
        >>> S = Scaffolds([('c-0', 0, 1, 1), ('c-1', 0, 1, 1), ('c-2', 0, 1, 1)])
        >>> S.contig_G.add_path([0, 1, 2])
        >>> S.paths = None
        >>> contig_3 = S.add_contig('contig-3')
        >>> S.add_edge(2, contig_3)
        >>> S.get_all_paths()
        [[0, 1, 2, 3]]
        """
        if 'no_check' not in attr:
            self.check_edge(u,v)

        # check if the node exists
        for node in [u, v]:
            if not self.contig_G.has_node(node):
                message = "Node {} does not exists ".format(node)
                raise HiCAssemblerException(message)

        # if u already has a successor 
        # invert the direction of the
        # path it belongs to
        if self.contig_G.out_degree(u) > 0:
            self.invert_path_containing_source(u)

        # if v already has a successor 
        # invert the direction of the
        # path it belongs to
        if self.contig_G.in_degree(v) > 0:
            self.invert_path_containing_source(v)

        self.contig_G.add_edge(u, v, **attr)
        # reset paths
        self.paths = None
 
    def get_neighbors(self, u):
        """
        give a node u, it returns the
        succesor and predecesor nodes
        """
        return self.contig_G.predecessors(u) + self.contig_G.successors(u)
        
    def invert_path_containing_source(self, v):
        """
        inverts a path

        >>> S = Scaffolds([('c-0', 0, 1, 1), ('c-1', 0, 1, 1), ('c-2', 0, 1, 1),
        ... ('c-4', 0, 1, 1), ('c-4', 0, 1, 1)])
        >>> S.contig_G.add_path([0, 1, 2, 3, 4], source_direction=1, \
        target_direction=0)
        >>> S.invert_path_containing_source(0)

        The source, directions flags should be flipped
        >>> S.contig_G.edges(4, data=True)
        [(4, 3, {'source_direction': 0, 'contig_part': True, 'iteration': 0, 'target_direction': 1})]
        """
        path = self.get_path_containing_source(v)
        for index in range(len(path) - 1):
            v = path[index]
            u = path[index + 1]
            e_data = self.contig_G.edge[v][u]
            # swap source target direction
            try:
                e_data['source_direction'],  e_data['target_direction'] = \
                    e_data['target_direction'],  e_data['source_direction']
            except KeyError:
                # this case happens for hic edges
                pass
            self.contig_G.remove_edge(v, u)
            self.contig_G.add_edge(u, v, e_data)

        self.paths = None

    def read_PE_scaffolds_abyss(self, scaffolds_file):
        # OBSOLETE Sun, 28 Jul 2013 01:29:07
        # kept only for documentation
        """
        this code parses the file that contains
        the information to construct PE_scaffolds
        from contigs.

        This is saved in a .de file
        The format is explained here:
        https://groups.google.com/d/msg/abyss-users/G2BmG4I3YPs/2coOJSo5mNEJ
        and looks like this:

        contig-39 contig-81+,1251,3,469.6 contig-261-,361,3,469.6 ; contig-307-,926,3,469.6
        
        which is translated to:
        contig-39+ -> contig-81+ [d=1251 e=469.6 n=3] 
        contig-39+ -> contig-261- [d=361 e=469.6 n=3] 
        contig-39- -> contig-307+ [d=926 e=469.6 n=3]

        where d is estimated distance, n is number
        of pairs supporting the join and e is the
        error of the distance estimation.

        The semicolon separates the contigs that are merged to the left
        or to the right
        """
        print "[{}] reading scaffolding data based on PE {} ".format(
            inspect.stack()[0][3], scaffolds_file)

        PE_scaffolds = []
        PE_length = []
        for line in scaffolds_file:
            PE_scaf = []
            _temp = []
            left, right = line.strip().split(' ;')
            scaf = left.split(' ')
            # first element of 'left' is 
            # the seed contig that is extended left 
            # and right.
            seed_contig = scaf.pop(0).strip()
            if seed_contig not in self.contig2id:
                continue

            # add left contigs
            for contig in scaf:
                _temp.append(contig.split(',')[0][:-1])

            _temp.append(seed_contig)
            scaf = right.split(' ')
            # add contigs to the right
            for contig in scaf:
                _temp.append(contig.split(',')[0][:-1])

            # consolidate contig list
            for contig_id in _temp:
                if contig_id in self.contig2id:
                    PE_scaf.append(self.contig2id[contig_id])
                    del self.contig2id[contig_id]
                elif contig_id.strip() != '':
                    """
                    print "PE scaffold considers contig {} "\
                        "but this contig is not in the hic "\
                        "matrix".format(contig_id)
                    """
            if len(PE_scaf):
                PE_scaffolds.append(PE_scaf)

        # append all ids that were
        # not considered in the scaffolds_file as singletons
        for scaf in self.contig2id.values():
            PE_scaffolds.append([scaf])

        for PE_scaf in PE_scaffolds:
            PE_length.append(sum([length[x] for x in PE_scaf]))
        
        if debug:
            # order the PE_scaffols by contig id
            order = np.argsort([min(x) for x in PE_scaffolds])
            PE_scaffolds = [PE_scaffolds[x] for x in order]

        self.PE_scaffolds = PE_scaffolds

        return self.PE_scaffolds

    def save_network(self, file_name):
        nx.write_gml(self.contig_G, file_name)


class Overlap:
    def __init__(self, overlap_file, id2contig_name):
        self.overlap_G = Overlap.get_overlap_graph(overlap_file)
        self.id2contig_name = id2contig_name
#        mapping = dict([(k,v) for v,k in enumerate(id2contig_name)])
    
    @staticmethod
    def get_overlap_graph(overlap_file):
        """
        This is based on the SGA format stored
        in files ending with asqg.gz
        It contains, both the sequences and the
        network at the end of the file.
        The network part is recognized
        by the ED prefix.

        The format is
        ED      contig-46260 contig-5624 11103 11200 11201 73 170 171 1 -1

        where:
        11103  = start contig_a
        11200  = end contig_a
        11201 = len contig_a

        73  = start contig_a
        170  = end contig_a
        171 = len contig_a

        1 = direction (0 -> same direction, 1 -> contig_b needs to be flipped)
        -1 = no idea, same for all overlaps
        """
        if overlap_file is None:
            return nx.Graph()
        if save_time:
            from os import path
            basename = path.basename(overlap_file)
            filename = '{}{}.gpickle'.format(TEMP_FOLDER, basename)
            try:
                print "loading pickled overlap {}".format(filename)
                G = nx.read_gpickle(filename)
                return G
            except:
                print "no saved pickled overlap found"
                pass

        
        print "[{}] reading overlap graph {} ".format(
            inspect.stack()[0][3], overlap_file)

        if overlap_file.endswith(".gz"):
            fh = gzip.open(overlap_file, 'rb')
        else:
            fh = open(overlap_file, 'r')
        G = nx.Graph()

        for line in fh:
            if line.startswith("ED"):
                node_a, node_b, start_a, end_a, len_a, \
                    start_b, end_b, len_b, direction, num = \
                    line.split('\t')[1].split(' ')
                for node in [node_a, node_b]:
                    if node == node_a:
                        length = int(len_a)
                    else:
                        length = int(len_b)
                    G.add_node(node, 
                               label=node,
                               length=length)

                if len_a > 1000 and len_b > 1000:
                    G.add_edge(
                        node_a, node_b,
                        start_a=int(start_a),
                        end_a=int(end_a),
                        length_a=int(len_a),
                        start_b=int(start_b),
                        end_b=int(end_b),
                        length_b=int(len_b),
                        direction=int(direction))
        if save_time:
            nx.write_gpickle(G, filename)
            nx.write_gml(G, TEMP_FOLDER+'{}.gml'.format(basename))

        return G

    def have_direct_overlap(self, u,v):
        if len(self.get_shortest_path(u,v)):
            return True

    def get_shortest_path(self, u,v):
        s_path = []
        if self.id2contig_name [u] != self.id2contig_name [v]:
            try:
                s_path = nx.shortest_path(self.overlap_graph,
                                          self.id2contig_name[u],
                                          self.id2contig_name[v])
            except nx.NetworkXNoPath:
                pass
            except nx.NetworkXError:
                pass
            except:
                pass
        return s_path

    def test_overlap(self, u,v,**attr):
        s_path = self.get_shortest_path(u,v)
        name_u = self.id2contig_name[u]
        name_v = self.id2contig_name[v]
        if len(s_path) == 2:
            # the two contigs touch is other
            # high evidence for merge
            attr['overlap'] = True
            overlap_data = \
                self.overlap_graph.edge[name_u][name_v]
            contig_b = self.hic.cut_intervals[v]
            # find out in the overlap which contig is a and which is b
            if overlap_data['length_a'] == contig_b[2]:
                # swap u, and v
                u,v = v,u
            attr['source_direction'] = 0 if overlap_data['start_a'] == 0 else 1
            attr['target_direction'] = 0 if overlap_data['start_b'] == 0 else 1
            print "direct overlap between {}, {} found.".format(u,v)
        if len(s_path) == 3:
            attr['shared_neighbor'] = True
            print "indirect overlap between {}, {} {} found.".format(u,v, s_path)
            # this case is complicated unless there is only 
            # one short path, but even though it is still
            # possible to assume that the shortest path
            # is the one joing the two contings.
            # Probably for some cases that assumption
            # holds.
            """
            paths = [p for p in nx.all_shortest_paths(self.overlap_graph,
                                                      id2name[u], id2name[v])]
            """

        return attr


class HiCAssemblerException(Exception):
        """Base class for exceptions in HiCAssembler."""

