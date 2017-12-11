#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse
import os
import errno

import numpy as np

import hicassembler.parserCommon as parserCommon

import hicassembler.HiCAssembler as HiCAssembler
import logging as log

import matplotlib
matplotlib.use('Agg')
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

    parser.add_argument('--outFolder', '-o',
                        help='folder were to save output files.',
                        required=True)

    parser.add_argument('--fasta', '-f',
                        help='fasta used for the hic',
                        required=True)

    parser.add_argument('--min_scaffold_length',
                        help='Minimum scaffold length for backbone to use. Using larger (>300kb) scaffolds '
                             'avoids the formation of incorrect super-scaffolds. At a later stage the smaller'
                             'scaffolds are put back into the backbone. If just few large scaffolds are available '
                             'this parameter should be decreased.',
                        required=False,
                        type=int,
                        default=300000)

    parser.add_argument('--bin_size',
                        help='bin size (in bp) to use. Usually a high resolution matrix is provided to the assembler. '
                             'A lower resolution matrix can be used if the depth of sequencing is low.',
                        required=False,
                        type=int,
                        default=25000)

    parser.add_argument('--num_processors',
                        help='Number of processors to use.',
                        required=False,
                        type=int,
                        default=1)

    parser.add_argument('--misassembly_zscore_threshold', '-mzt',
                        help='To detect misasemblies, a z-score is computed similar to the method used to identify '
                             'TAD boundaries (see HiCExploer `hicFindTADs`). Missassemblies appear as strong TAD '
                             'boundaries but it not straightforward to distinguish true misassemblies from TAD '
                             'boundaries. A relax threshold is used by default because a false positive (a TAD '
                             'boundary that is thought to be a misassembly) has a little impact on the assembly '
                             'compared to a false negative (a misassembly that was not detected). However, the '
                             'default parameter may fragment the contigs/scaffolds to frequently and a more '
                             'stringent value is preferable. A simple way to test if misasemblies are present is '
                             'by inspecting the resulting matrix by clear gaps. If this is the case, change the '
                             'zscore threshold to a larger value. ',
                        required=False,
                        type=float,
                        default=-1)

    parser.add_argument('--split_positions_file',
                        help='BED file. If the location of some mis-assemblies are known, they can be provided on a '
                             'bed file to be splitted.',
                        required=False)

    return parser.parse_args(args)


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


def save_fasta(input_fasta, output_fasta, super_scaffolds, print_stats=True, contig_separator='N'*5000):
    r"""
    Takes the hic scaffolds information and the original fasta file
    and merges the individual scaffolds sequences. All sequences that are
    not part of the hic scaffolds are returned as well

    Parameters
    ----------
    input_fasta input fasta file
    output_fasta
    super_scaffolds in the form of a list of list.
    print_stats boolean If true, then the total number of bases on the input fasta, hic scaffolds and missing
                        scaffolds is printed
    contig_separator Sequence to add between contig/scaffolds to separate them.

    Returns
    -------

    # make small test fasta file
    >>> test_fasta = ">one\nAAAGGG\n>two\nTTTAAA\n"
    >>> fh = open("/tmp/test.fasta",'w')
    >>> fh.write(test_fasta)
    >>> fh.close()

    Check that first three bases of sequence 'two' are added
    >>> scaff = [[('one', 0, 6, '+'), ('two', 0, 3, '+')]]
    >>> save_fasta('/tmp/test.fasta', '/tmp/out.fasta', scaff, print_stats=False, contig_separator='-')
    >>> open('/tmp/out.fasta', 'r').readlines()
    ['>hic_scaffold_1 one:0-6:+,two:0-3:+\n', 'AAAGGG-TTT\n']

    Check that last three bases of sequence 'two' are added and that
    sequence 'one' is backwards
    >>> scaff = [[('one', 0, 6, '-'), ('two', 3, 6, '+')]]
    >>> save_fasta('/tmp/test.fasta', '/tmp/out.fasta', scaff, print_stats=False, contig_separator='-')
    >>> open('/tmp/out.fasta', 'r').readlines()
    ['>hic_scaffold_1 one:0-6:-,two:3-6:+\n', 'GGGAAA-AAA\n']

    >>> super_scaffolds = [[('scaffold_12472', 170267, 763072, '-'), ('scaffold_12932', 1529201, 1711857, '-'),
    ... ('scaffold_12932', 1711857, 2102469, '-'), ('scaffold_12726', 1501564, 2840439, '-')],
    ... [('scaffold_13042', 0, 239762, '-'), ('scaffold_12928', 0, 1142515, '-')]]
    >>> save_fasta("../hicassembler/test/scaffolds_test.fa", "/tmp/test.fa", super_scaffolds)
    Total fasta length: 18,618,060
    Total missing contig/scaffolds length: 2,748 (0.01%)
    Total hic scaffolds length: 3,907,225 (20.99%)
    """

    from Bio import SeqIO
    from Bio.Seq import Seq
    record_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    new_rec_list = []
    nnn_seq = Seq(contig_separator)
    super_scaffolds_len = 0
    seen = set([])
    for idx, super_c in enumerate(super_scaffolds):
        sequence = Seq("")
        info = []
        for contig_idx, (contig_id, start, end, strand) in enumerate(super_c):
            if strand == '-':
                sequence += record_dict[contig_id][start:end][::-1]
            else:
                sequence += record_dict[contig_id][start:end]
            if contig_idx < len(super_c) - 1:
                # only add the separator sequence
                sequence += nnn_seq
            info.append("{contig}:{start}-{end}:{strand}".
                        format(contig=contig_id, start=start, end=end, strand=strand))
            seen.add(contig_id)

        id = "hic_scaffold_{} ".format(idx + 1) + ",".join(info)
        sequence.id = id
        sequence.description = ""
        new_rec_list.append(sequence)
        super_scaffolds_len += len(sequence)

    # # check contigs that are in the input fasta but are not in the super_scaffolds
    # missing_fasta_len = 0
    # missing_fasta_ids = set(record_dict.keys()) - seen
    #
    # for fasta_id in missing_fasta_ids:
    #     new_rec_list.append(record_dict[fasta_id])
    #     missing_fasta_len += len(record_dict[fasta_id])
    #
    # with open(output_fasta, "w") as handle:
    #     SeqIO.write(new_rec_list, handle, "fasta")

    if print_stats:
        total_in_fasta_sequence_length = 0
        for record_id, record in record_dict.items():
            total_in_fasta_sequence_length += len(record.seq)
        print("Total fasta length: {:,}".format(total_in_fasta_sequence_length))
        print("Total missing contig/scaffolds length: {:,} ({:.2%})".
                 format(missing_fasta_len, float(missing_fasta_len) / total_in_fasta_sequence_length))
        print("Total hic scaffolds length: {:,} ({:.2%})".
                 format(super_scaffolds_len, float(super_scaffolds_len) / total_in_fasta_sequence_length))


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def main(args):
    # load matrix
    make_sure_path_exists(args.outFolder)
    assembl = HiCAssembler.HiCAssembler(args.matrix, args.fasta, args.outFolder,
                                        min_scaffold_length=args.min_scaffold_length,
                                        matrix_bin_size=args.bin_size,
                                        num_processors=args.num_processors,
                                        misassembly_zscore_threshold=args.misassembly_zscore_threshold,
                                        split_positions_file=args.split_positions_file)

    super_contigs = assembl.assemble_contigs()
    save_fasta(args.fasta, args.outFolder + "/super_scaffolds.fa", super_contigs)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
