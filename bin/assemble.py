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

    assembl.assemble_contigs()
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
