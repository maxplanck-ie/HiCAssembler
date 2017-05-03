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

    parser.add_argument('--fasta', '-f',
                        help='fasta used for the hic',
                        required=True)

    return(parser.parse_args(args))


def dot_plot_super_contigs(super_contigs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    num_chroms = len(super_contigs.keys())
    fig = plt.figure(figsize=(8 * num_chroms, 8))
    for index, (chrom, paths) in enumerate(super_contigs.iteritems()):
        ax = fig.add_subplot(1, num_chroms, index+1)
        ax.set_title(chrom)
        start = 0
        vlines = []
        for path in paths:
            vlines.append(start)
            for contig in path:
                contig_len = abs(contig[1] - contig[0])
                end = start + contig_len
                ax.plot((start, end), (contig[0], contig[1]))
                ax.scatter((start, end), (contig[0], contig[1]), s=3)
                start = end + 1
        y_min, y_max = ax.get_ylim()
        ax.vlines(vlines, y_min, y_max, linestyles="dotted", colors='gray', alpha=0.5)

    plt.savefig("/tmp/test.pdf")


def evaluate_super_contigs(super_contigs):
    from collections import defaultdict
    super_check_list = defaultdict(list)
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
            # revert check list such that the start of the first tuple is the smaller value
            check_list = [x[::-1] for x in check_list][::-1]
        if sorted(sum(check_list,[])) != sum(check_list, []):
            formatted_paths = []
            for c_path in check_list:
                formatted_paths.append(["{:,}".format(x) for x in c_path])
            log.warn("Problem with {}".format(formatted_paths))
        super_check_list[chrom].append(check_list)
    dot_plot_super_contigs(super_check_list)


def save_fasta(input_fasta, output_fasta, super_scaffolds):
    """

    Parameters
    ----------
    input_fasta input fasta file
    output_fasta
    super_scaffolds in the form of a list of list.

    Returns
    -------

    >>> super_scaffolds = [[('scaffold_12958', '+')], [('scaffold_12942', '+')], [('scaffold_12728', '+')], [('scaffold_12723', '+'), ('scaffold_12963', '+'), ('scaffold_13246', '-'), ('scaffold_12822', '+'), ('scaffold_13047', '-'), ('scaffold_12855', '+')]]
    >>> save_fasta("../virilis_assembly/data/dvir1.3.fa", "/tmp/test.fa", super_scaffolds)
    """

    from Bio import SeqIO
    from Bio.Seq import Seq
    record_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    new_rec_list = []

    for super_c in super_scaffolds:
        sequence = Seq("")
        id = []
        for contig_id, strand in super_c:
            if strand == '-':
                sequence += record_dict[contig_id].reverse_complement()
            else:
                sequence += record_dict[contig_id]
            id.append("{}_{}".format(contig_id, strand))

        id = "_".join(id)
        sequence.id = id
        sequence.description = ""
        new_rec_list.append(sequence)

    with open(output_fasta, "w") as handle:
        SeqIO.write(new_rec_list, handle, "fasta")


def main(args):
    # load matrix
    basename = args.outFile
    ma = HiCMatrix.hiCMatrix(args.matrix)
    names_list = []
    assembl = HiCAssembler.HiCAssembler(ma, args.fasta)
    #import ipdb;ipdb.set_trace()
    super_contigs, paths = assembl.assemble_contigs()
    save_fasta(args.fasta, "super_scaffolds.fa", super_contigs)

    flat = []
    for contig in super_contigs:
        for part in contig:
            flat.append(part[0])

    print " ".join(flat)

    print super_contigs
    evaluate_super_contigs(super_contigs)
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
