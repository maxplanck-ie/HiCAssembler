#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse

import numpy as np

import hicassembler.parserCommon as parserCommon

import hicexplorer.HiCMatrix as HiCMatrix
import hicassembler.HiCAssembler as HiCAssembler
import logging as log

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
debug = 0

TEMP_FOLDER = '/tmp/'

log.basicConfig(level=log.DEBUG)

def parse_arguments(args=None):
    parent_parser = parserCommon.getParentArgParse()

    parser = argparse.ArgumentParser(
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='The idea is to test whether the contigs '
        'align next to each other. The format of '
        'the matrix row labels should include '
        'the position of the contig.')

    parser.add_argument('--outFile', '-o',
                        help='prefix for output files.',
                        required=True)


    return(parser.parse_args(args))


def main(args):
    # load matrix
    basename = args.outFile
    ma = HiCMatrix.hiCMatrix(args.matrix)
    names_list = []

    assembl = HiCAssembler.HiCAssembler(ma)
    super_contigs, paths = assembl.assemble_contigs()
    super_check_list = []
    for s_contig in super_contigs:
        check_list = []
        for contig in s_contig:
            # check if the join is correct
            name, direction = contig
            name = name.replace("(-)", ":minus").replace("(+)", ":plus").replace('-', ':')
            try:
                chrom, start, end, strand = name.split(':')
            except:
                import ipdb;ipdb.set_trace()
            start = int(start)
            end = int(end)
            if strand == 'minus':
                start, end = end, start

            if direction == '-':
                start, end = end, start
            check_list.append([start, end])

        if check_list[0][0] > check_list[-1][-1]:
            check_list = [x[::-1] for x in check_list][::-1]
        if sorted(sum(check_list,[])) != sum(check_list, []):
            formatted_paths = []
            for c_path in check_list:
                formatted_paths.append(["{:,}".format(x) for x in c_path])
            log.warn("Problem with {}".format(formatted_paths))
        super_check_list.append(check_list)
    import ipdb;ipdb.set_trace()

    data = {'scaffolds':assembl.N50,
            'id2contig_pos': assembl.hic.cut_intervals,
            'contig_name': assembl.scaffolds.id2contig_name,
            'id2pos': position}

    make_plots(data, args.outFile, args.genomeSizes)

#    scaffolds_to_bed(assembl, position)
    import pickle as pi
    with open(args.outFile+'_final_assembl.pickle', 'wb') as f:
        pi.dump(data,f)
    import ipdb;ipdb.set_trace()
    exit()
    print_N50(data['scaffolds'], args.outFile)
    test_super_contigs(assembl, args.outFile,position)
    exit()
    print np.array([(k, v) for k, v in id2pos.iteritems()])

#    mat = ma.matrix[new_order,:][:,new_order]
#    np.savez("/tmp/ordermat.npz", mat=mat)
#    ma.setMatrixValues(mat)
#    ma.save(args.outFile.name)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
