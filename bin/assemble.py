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
                        help='Number of processors to use. Multithreading is only used to identify mis-assemblies.',
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

    parser.add_argument('--num_iterations',
                        help='HiCAssembles aims to produce longer and longer hi-c scaffolds in each iteration. The '
                             'firsts iterations use stringent criteria to join contigs/scaffolds. The number of '
                             'iterations required depend from sample to sample. By observing the after_assembly_.pdf '
                             'images the desired number of iterations can be selected. Usually no more than 3 iterations '
                             'are required.',
                        type=int,
                        default=2,
                        required=False)

    parser.add_argument('--split_positions_file',
                        help='BED file. If the location of some mis-assemblies are known, they can be provided on a '
                             'bed file to be splitted. The Hi-C matrix bin that overlaps with a region in the bed '
                             'file becomes the last bin before the split.',
                        required=False)

    parser.add_argument('--scaffolds_to_ignore',
                        help='The assembly process is affected by scaffolds that appear close to several other '
                             'scaffolds. Normally, for each scaffold a pair of neighbors can be identified, however '
                             'for some scaffolds several neighbors are found. These problematic scaffolds could be '
                             'caused by misassemblies, duplications or repetitive regions. Usually, better assemblies '
                             'are produced by removing this scaffolds initially. Then, when most of the scaffolds are '
                             'as assembled, the algorithm will try to place those scaffolds back into the assembly. '
                             'The scaffolds should be separated by space: e.g. scaffold_10 scafold_32 scaffold_27',
                        nargs='+',
                        required=False)
    return parser.parse_args(args)


def save_fasta(input_fasta, output_fasta, super_scaffolds, print_stats=True, contig_separator='N'*2000,
               chain_file = None):
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
    >>> save_fasta('/tmp/test.fasta', '/tmp/out.fasta', scaff, print_stats=False, contig_separator='-', chain_file='/tmp/chain.txt')
    >>> open('/tmp/out.fasta', 'r').readlines()
    ['>hic_scaffold_1 one:0-6:-,two:3-6:+\n', 'CCCTTT-AAA\n']

    Check chain file
    >>> open('/tmp/chain.txt', 'r').readlines()[0]
    'chain\t100\tone\t6\t+\t0\t6\thic_scaffold_1\t10\t-\t4\t10\t1\n'

    Test the separator that is added.
    >>> test_fasta = ">one\nAAAGGGTTTAAA\n"
    >>> fh = open("/tmp/test.fasta",'w')
    >>> fh.write(test_fasta)
    >>> fh.close()

    Test for separator on + strand (3 NNNs should be added)
    >>> scaff = [[('one', 0, 3, '+'), ('one', 6, 9, '+')]]
    >>> save_fasta('/tmp/test.fasta', '/tmp/out.fasta', scaff, print_stats=False, contig_separator='-')
    >>> open('/tmp/out.fasta', 'r').readlines()
    ['>hic_scaffold_1 one:0-3:+,one:6-9:+\n', 'AAANNNTTT\n']

    Test for separator on - strand (3 NNNs should be added)
    >>> scaff = [[('one', 9, 12, '-'), ('one', 3, 6, '-')]]
    >>> save_fasta('/tmp/test.fasta', '/tmp/out.fasta', scaff, print_stats=False, contig_separator='-')
    >>> open('/tmp/out.fasta', 'r').readlines()
    ['>hic_scaffold_1 one:9-12:-,one:3-6:-\n', 'TTTNNNCCC\n']

    >>> super_scaffolds = [[('scaffold_12472', 170267, 763072, '-'), ('scaffold_12932', 1529201, 1711857, '+'),
    ... ('scaffold_12932', 1711857, 2102469, '+'), ('scaffold_12726', 1501564, 2840439, '-')],
    ... [('scaffold_13042', 0, 239762, '-'), ('scaffold_12928', 0, 1142515, '-')]]
    >>> save_fasta("../hicassembler/test/scaffolds_test.fa", "/tmp/test.fa", super_scaffolds,
    ... chain_file="/tmp/chain.txt")
    Total fasta length: 18,618,060
    Total missing contig/scaffolds length: 2,748 (0.01%)
    Total hic scaffolds length: 3,907,225 (20.99%)
    """

    chain_id = 0
    from Bio import SeqIO
    from Bio.Seq import Seq
    record_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    hic_records_list = []
    nnn_seq = Seq(contig_separator)
    super_scaffolds_len = 0
    seen = set([])
    if chain_file and os.path.isfile(chain_file):
        os.unlink(chain_file)

    for idx, super_c in enumerate(super_scaffolds):
        chain_data = []
        hic_scaffold_start = 0
        sequence = Seq("")
        info = []
        for contig_idx, (contig_id, start, end, strand) in enumerate(super_c):
            if strand == '-':
                # the reverse complement is the correct way to get
                # the inverted sequence.
                sequence += record_dict[contig_id][start:end].reverse_complement()
            else:
                sequence += record_dict[contig_id][start:end]

            hic_scaffold_end = len(sequence)
            if contig_idx == len(super_c) - 1:
                # only add the separator sequence if the sequence is not the last
                pass

            else:
                (n_contig_id, n_start, n_end, n_strand) = super_c[contig_idx + 1]
                next_contig = {'contig_id': n_contig_id, 'start': n_start, 'end': n_end, 'strand': n_strand}

                if next_contig['contig_id'] == contig_id and next_contig['strand'] == strand:
                    # this means that a contig was split by the misassembly correction but was later joined together.
                    # the next cases test if the contigs are contiguous (separated by less than 500 bp)
                    # A number of NNs is added between the two contigs equal to their distance.
                    # For example, for ('A', 0, 100, '+') and ('A', 110, 200, '+'), 10 'N's are added in the fasta
                    # file.
                    if strand == '+' and next_contig['start'] - end < 500:
                        assert(next_contig['start'] - end >= 0)
                        sequence += Seq('N' * (next_contig['start'] - end))
                    elif strand == '-' and start - next_contig['end'] < 500:
                        assert(start - next_contig['end'] >= 0)
                        sequence += Seq('N' * (start - next_contig['end']))
                else:
                    sequence += nnn_seq

            info.append("{contig}:{start}-{end}:{strand}".
                        format(contig=contig_id, start=start, end=end, strand=strand))
            chain_data.append((hic_scaffold_start, hic_scaffold_end, contig_id, start, end,
                               len(record_dict[contig_id]), strand))
            hic_scaffold_start = len(sequence)
            seen.add(contig_id)

        id = "hic_scaffold_{}".format(idx + 1)
        sequence.id = id + " " + ",".join(info)
        sequence.description = ""
        hic_records_list.append(sequence)
        seq_length = len(sequence)
        super_scaffolds_len += seq_length
        if chain_file:
            with open(chain_file, 'a') as fh:
                for hic_start, hic_end, contig_id, contig_start, contig_end, contig_len, strand in chain_data:
                    chain_id += 1
                    if strand == '-':
                        # for the chain file, if the strand is -, the coordinates need to be
                        # given with respect to the inverted sequence. Thus, if hic_start=0 and hic_end=10
                        # for a hic scaffold of length 20, the negative coordinates are hic_start_10, hic_end 20
                        new_hic_start = seq_length - hic_end
                        new_hic_end = seq_length - hic_start
                        hic_start = new_hic_start
                        hic_end = new_hic_end
                    fh.write("chain\t100\t{contig_id}\t{contig_length}\t+\t{contig_start}\t{contig_end}"
                             "\t{id}\t{length}\t{strand}\t{hic_start}\t{hic_end}\t{chain_id}\n".
                             format(contig_id=contig_id,
                                    contig_length=contig_len,
                                    strand=strand,
                                    contig_start=contig_start,
                                    contig_end=contig_end,
                                    id=id,
                                    hic_start=hic_start,
                                    length=seq_length,
                                    hic_end=hic_end,
                                    chain_id=chain_id))
                    fh.write("{}\n\n".format(contig_end-contig_start))
    # check contigs that are in the input fasta but are not in the super_scaffolds
    missing_fasta_len = 0
    missing_fasta_ids = set(record_dict.keys()) - seen

    for fasta_id in missing_fasta_ids:
        hic_records_list.append(record_dict[fasta_id])
        missing_fasta_len += len(record_dict[fasta_id])

    with open(output_fasta, "w") as handle:
        SeqIO.write(hic_records_list, handle, "fasta")

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
                                        split_positions_file=args.split_positions_file,
                                        num_iterations=args.num_iterations,
                                        scaffolds_to_ignore=args.scaffolds_to_ignore)

    super_contigs = assembl.assemble_contigs()
    save_fasta(args.fasta, args.outFolder + "/super_scaffolds.fa", super_contigs,
               chain_file=args.outFolder + "/liftover.chain")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
