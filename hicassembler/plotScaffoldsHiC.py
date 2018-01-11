from __future__ import division
import matplotlib
matplotlib.use("Agg")


from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import sys
import hicexplorer.HiCMatrix as HiCMatrix
from hicexplorer.utilities import writableFile
from hicexplorer.utilities import toString, toBytes

from hicexplorer._version import __version__
from hicexplorer.trackPlot import file_to_intervaltree
import numpy as np
import pyBigWig
from builtins import range
from past.builtins import zip
from future.utils import itervalues

import cooler
import argparse
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import OrderedDict

import logging
log = logging.getLogger(__name__)

import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='Creates a Heatmap of a HiC matrix')

    # define the arguments
    parser.add_argument('--matrix', '-m',
                        help='Path of the Hi-C matrix to plot',
                        required=True)

    parser.add_argument('--title', '-t',
                        help='Plot title')

    parser.add_argument('--scoreName', '-s',
                        help='Score name')

    parser.add_argument('--outFileName', '-out',
                        help='File name to save the image. ',
                        type=writableFile,
                        required=True)

    parser.add_argument('--perScaffold',
                        help='Instead of plotting the whole matrix, '
                        'each scaffold is plotted next to the other. Small scaffolds may appear as a white box '
                        'This parameter is not compatible with --region',
                        action='store_true')

    parser.add_argument('--clearMaskedBins',
                        help='if set, masked bins are removed from the matrix',
                        action='store_true')


    parser.add_argument('--chromosomeOrder',
                        help='Chromosomes and order in which the '
                        'chromosomes should be plotted. This option '
                        'overrides --region and --region2 ',
                        nargs='+')

    parser.add_argument('--region',
                        help='Plot only this region. The format is '
                        'chr:start-end The plotted region contains '
                        'the main diagonal and is symmetric unless '
                        ' --region2 is given'
                        )

    parser.add_argument('--region2',
                        help='If given, then only the region defined by '
                        '--region and --region2 is given. The format '
                        'is the same as --region1'
                        )

    parser.add_argument('--log1p',
                        help='Plot the log1p of the matrix values.',
                        action='store_true')

    parser.add_argument('--log',
                        help='Plot the *MINUS* log of the matrix values.',
                        action='store_true')

    parser.add_argument('--colorMap',
                        help='Color map to use for the heatmap. Available '
                        'values can be seen here: '
                        'http://matplotlib.org/examples/color/colormaps_reference.html',
                        default='RdYlBu_r')

    parser.add_argument('--vMin',
                        help='vMin',
                        type=float,
                        default=None)

    parser.add_argument('--vMax',
                        help='vMax',
                        type=float,
                        default=None)

    parser.add_argument('--dpi',
                        help='Resolution for the image in case the'
                             'ouput is a raster graphics image (e.g png, jpg)',
                        type=int,
                        default=72)


    return parser


def relabel_ticks(pXTicks):
    if pXTicks[-1] > 1.5e6:
        labels = ["{:.2f} ".format(x / 1e6)
                  for x in pXTicks]
        labels[-2] += " Mbp"
    elif pXTicks[-1] > 1500:
        labels = ["{:.0f}".format(x / 1e3)
                  for x in pXTicks]
        labels[-2] += " Kbp"
    else:
        labels = ["{:.2f} ".format((x))
                  for x in pXTicks]
        labels[-2] += " bp"
    return labels


def change_chrom_names(chrom):
    """
    Changes UCSC chromosome names to ensembl chromosome names
    and vice versa.
    """
    # TODO: mapping from chromosome names like mithocondria is missing
    chrom = toString(chrom)
    if chrom.startswith('chr'):
        # remove the chr part from chromosome name
        chrom = chrom[3:]
    else:
        # prefix with 'chr' the chromosome name
        chrom = 'chr' + chrom

    return chrom


def plotHeatmap(ma, chrBinBoundaries, fig, position, args, cmap, xlabel=None,
                ylabel=None, start_pos=None, start_pos2=None, pNorm=None, pAxis=None):
    log.debug("plotting heatmap")
    if ma.shape[0] < 5:
        log.info("Matrix for {} too small to plot. Matrix size: {}".format(chrBinBoundaries.keys()[0], ma.shape))
        return
    if pAxis is not None:
        axHeat2 = pAxis
    else:
        axHeat2 = fig.add_axes(position)

    if args.title:
        axHeat2.set_title(toString(args.title))

    if start_pos is None:
        start_pos = np.arange(ma.shape[0])
    if start_pos2 is None:
        start_pos2 = start_pos

    xmesh, ymesh = np.meshgrid(start_pos, start_pos2)

    img3 = axHeat2.pcolormesh(xmesh.T, ymesh.T, ma, vmin=args.vMin, vmax=args.vMax, cmap=cmap, norm=pNorm)
    axHeat2.invert_yaxis()
    img3.set_rasterized(True)
    if args.region:
        xtick_lables = relabel_ticks(axHeat2.get_xticks())
        axHeat2.get_xaxis().set_tick_params(which='both', bottom='on', direction='out')
        axHeat2.set_xticklabels(xtick_lables, size='small', rotation=45)

        ytick_lables = relabel_ticks(axHeat2.get_yticks())
        axHeat2.get_yaxis().set_tick_params(which='both', bottom='on', direction='out')
        axHeat2.set_yticklabels(ytick_lables, size='small')
        xticks = [xtick_lables]
    else:

        ticks = [int(pos[0] + (pos[1] - pos[0]) / 2) for pos in itervalues(chrBinBoundaries)]
        labels = list(chrBinBoundaries)
        axHeat2.set_xticks(ticks)
        axHeat2.set_yticks(ticks)
        labels = toString(labels)
        xticks = [labels, ticks]

        if len(labels) > 20:
            axHeat2.set_xticklabels(labels, size=4, rotation=90)
            axHeat2.set_yticklabels(labels, size=4)

        else:
            axHeat2.set_xticklabels(labels, size=8)
            axHeat2.set_yticklabels(labels, size=8)

    divider = make_axes_locatable(axHeat2)
    cax = divider.append_axes("right", size="2.5%", pad=0.09)
    if args.log1p:
        from matplotlib.ticker import LogFormatter
        formatter = LogFormatter(10, labelOnlyBase=False)
        # get a useful log scale
        # that looks like [1, 2, 5, 10, 20, 50, 100, ... etc]
        aa = np.array([1, 2, 5])
        tick_values = np.concatenate([aa * 10 ** x for x in range(10)])
        try:
            cbar = fig.colorbar(img3, ticks=tick_values, format=formatter, cax=cax)
        except:
            import ipdb;ipdb.set_trace()
    else:
        cbar = fig.colorbar(img3, cax=cax)

    cbar.solids.set_edgecolor("face")  # to avoid white lines in the color bar in pdf plots
    if args.scoreName:
        cbar.ax.set_ylabel(args.scoreName, rotation=270, size=8)

    if ylabel is not None:
        ylabel = toString(ylabel)
        axHeat2.set_ylabel(ylabel)

    if xlabel is not None:
        xlabel = toString(xlabel)
        axHeat2.set_xlabel(xlabel)



def translate_region(region_string):
    """
    Takes an string and returns a list
    of chrom, start, end.
    If the region string only contains
    the chrom, then start and end
    are set to a 0 and 1e15
    """

    if sys.version_info[0] == 2:
        region_string = region_string.translate(None, ",;!").replace("-", ":")
    if sys.version_info[0] == 3:
        # region_string = toBytes(region_string)
        region_string = region_string.replace(",", "")
        region_string = region_string.replace(";", "")
        region_string = region_string.replace("!", "")
        region_string = region_string.replace("-", ":")

    fields = region_string.split(":")
    chrom = fields[0]
    try:
        region_start = int(fields[1])
    except IndexError:
        region_start = 0
    try:
        region_end = int(fields[2])
    except IndexError:
        region_end = 1e15  # vert large number

    return chrom, region_start, region_end


def plotPerChr(hic_matrix, cmap, args):
    """
    plots each chromosome individually, one after the other
    in one row. scale bar is added at the end
    """
    from math import ceil
    chromosomes = hic_matrix.getChrNames()
    chrom_per_row = 15
    num_rows = int(ceil(float(len(chromosomes)) / chrom_per_row))
    num_cols = min(chrom_per_row, len(chromosomes))
    width_ratios = [1.0] * num_cols + [0.05]
    grids = gridspec.GridSpec(num_rows, num_cols + 1,
                              width_ratios=width_ratios,
                              height_ratios=[1] * num_rows)

    fig_height = 6 * num_rows
    fig_width = sum((np.array(width_ratios) + 0.05) * 6)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi)

    chrom, start, end, _ = zip(*hic_matrix.cut_intervals)
    for idx, chrname in enumerate(chromosomes):
        log.debug('chrom: {}'.format(chrname))

        row = idx // chrom_per_row
        col = idx % chrom_per_row
        axis = plt.subplot(grids[row, col])
        axis.set_title(toString(chrname))
        chrom_range = hic_matrix.getChrBinRange(chrname)
        matrix = np.asarray(hic_matrix.matrix[chrom_range[0]:chrom_range[1],
                                              chrom_range[0]:chrom_range[1]].todense().astype(float))

        norm = None
        if args.log or args.log1p:
            mask = matrix == 0
            mask_nan = np.isnan(matrix)
            mask_inf = np.isinf(matrix)
            log.debug("any nan {}".format(np.isnan(matrix).any()))
            log.debug("any inf {}".format(np.isinf(matrix).any()))

            try:
                matrix[mask] = np.nanmin(matrix[mask == False])
                matrix[mask_nan] = np.nanmin(matrix[mask_nan == False])
                matrix[mask_inf] = np.nanmin(matrix[mask_inf == False])

                if args.log:
                    matrix = np.log(matrix)
            except Exception:
                log.debug("Clearing of matrix failed.")
            log.debug("any nanafter remove of nan: {}".format(np.isnan(matrix).any()))
            log.debug("any inf after remove of inf: {}".format(np.isinf(matrix).any()))
        if args.log1p:
            matrix += 1
            norm = LogNorm()

        chr_bin_boundary = OrderedDict()
        chr_bin_boundary[chrname] = hic_matrix.chrBinBoundaries[chrname]

        args.region = toString(chrname)
        chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = getRegion(args, hic_matrix)
        plotHeatmap(matrix, chr_bin_boundary, fig, None,
                    args, cmap, xlabel=chrname, ylabel=chrname,
                    start_pos=start_pos1, start_pos2=start_pos2, pNorm=norm, pAxis=axis)
    return fig


def getRegion(args, ma):
    chrom = region_start = region_end = idx1 = start_pos1 = chrom2 = region_start2 = region_end2 = idx2 = start_pos2 = None
    chrom, region_start, region_end = translate_region(args.region)

    if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
        chrom = toBytes(chrom)

    if chrom not in list(ma.interval_trees):

        chrom = change_chrom_names(chrom)

        if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
            chrom = toBytes(chrom)

        if chrom not in list(ma.interval_trees):
            exit("Chromosome name {} in --region not in matrix".format(change_chrom_names(chrom)))

    args.region = [chrom, region_start, region_end]
    is_cooler = False
    if args.matrix.endswith('.cool') or cooler.io.is_cooler(args.matrix):
        is_cooler = True
    if is_cooler:
        idx1, start_pos1 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom and
                                 ((x[1] >= region_start and x[2] < region_end) or
                                  (x[1] < region_end and x[2] < region_end and x[2] > region_start) or
                                  (x[1] > region_start and x[1] < region_end))])
    else:
        idx1, start_pos1 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom and
                                 x[1] >= region_start and x[2] < region_end])
    if args.region2:
        chrom2, region_start2, region_end2 = translate_region(args.region2)
        if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
            chrom2 = toBytes(chrom)
        if chrom2 not in list(ma.interval_trees):
            chrom2 = change_chrom_names(chrom2)
            if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
                chrom2 = toBytes(chrom)
            if chrom2 not in list(ma.interval_trees):
                exit("Chromosome name {} in --region2 not in matrix".format(change_chrom_names(chrom2)))
        if is_cooler:
            idx2, start_pos2 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom2 and
                                     ((x[1] >= region_start2 and x[2] < region_end2) or
                                      (x[1] < region_end2 and x[2] < region_end2 and x[2] > region_start2) or
                                      (x[1] > region_start2 and x[1] < region_end2))])
        else:
            idx2, start_pos2 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom2 and
                                     x[1] >= region_start2 and x[2] < region_end2])
    else:
        idx2 = idx1
        chrom2 = chrom
        start_pos2 = start_pos1

    return chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2


def main(args=None):
    args = parse_arguments().parse_args(args)
    chrom = None
    start_pos1 = None
    chrom2 = None
    start_pos2 = None

    if args.perScaffold and args.region:
        log.error('ERROR, choose from the option '
                  '--perScaffold or --region, the two '
                  'options at the same time are not '
                  'compatible.')
        exit(1)

    is_cooler = False
    if args.matrix.endswith('.cool') or cooler.io.is_cooler(args.matrix):
        is_cooler = True
    if is_cooler and not args.region2:
        log.debug("Retrieve data from cooler format and use its benefits.")
        regionsToRetrieve = None
        if args.region:
            regionsToRetrieve = []
            regionsToRetrieve.append(args.region)
            if args.region2:
                chrom2, region_start2, region_end2 = translate_region(args.region2)
                regionsToRetrieve.append(args.region2)
        if args.chromosomeOrder:
            args.region = None
            args.region2 = None
            regionsToRetrieve = args.chromosomeOrder

        ma = HiCMatrix.hiCMatrix(args.matrix, chrnameList=regionsToRetrieve)

        if args.clearMaskedBins:
            ma.maskBins(ma.nan_bins)
        if args.region:
            chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = getRegion(args, ma)

        matrix = np.asarray(ma.matrix.todense().astype(float))

    else:
        ma = HiCMatrix.hiCMatrix(args.matrix)
        if args.clearMaskedBins:
            ma.maskBins(ma.nan_bins)
        if args.chromosomeOrder:
            args.region = None
            args.region2 = None

            valid_chromosomes = []
            invalid_chromosomes = []
            log.debug('args.chromosomeOrder: {}'.format(args.chromosomeOrder))
            log.debug("ma.chrBinBoundaries {}".format(ma.chrBinBoundaries))
            if sys.version_info[0] == 3:
                args.chromosomeOrder = toBytes(args.chromosomeOrder)
            for chrom in args.chromosomeOrder:
                if chrom in ma.chrBinBoundaries:
                    valid_chromosomes.append(chrom)
                else:
                    invalid_chromosomes.append(chrom)

            if len(invalid_chromosomes) > 0:
                log.warning("WARNING: The following chromosome/scaffold names were not found. Please check"
                            "the correct spelling of the chromosome names. \n")
                log.warning("\n".join(invalid_chromosomes))
            ma.reorderChromosomes(valid_chromosomes)

        log.info("min: {}, max: {}\n".format(ma.matrix.data.min(), ma.matrix.data.max()))

        if args.region:
            chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = getRegion(args, ma)

            matrix = np.asarray(ma.matrix[idx1, :][:, idx2].todense().astype(float))

        else:
            log.debug("Else branch")
            matrix = np.asarray(ma.getMatrix().astype(float))

    matrix_length = len(matrix[0])
    for matrix_ in matrix:
        if not matrix_length == len(matrix_):
            log.error("Matrices do not have the same length: {} , {}".format(matrix_length, len(matrix_)))

    cmap = cm.get_cmap(args.colorMap)
    log.debug("Nan values set to black\n")
    cmap.set_bad('black')

    if args.perScaffold:
        fig = plotPerChr(ma, cmap, args)

    else:
        norm = None

        if args.log or args.log1p:
            mask = matrix == 0
            matrix[mask] = np.nanmin(matrix[mask == False])

            if np.isnan(matrix).any() or np.isinf(matrix).any():
                log.debug("any nan {}".format(np.isnan(matrix).any()))
                log.debug("any inf {}".format(np.isinf(matrix).any()))
                mask_nan = np.isnan(matrix)
                mask_inf = np.isinf(matrix)
                matrix[mask_nan] = np.nanmin(matrix[mask_nan == False])
                matrix[mask_inf] = np.nanmin(matrix[mask_inf == False])

            if args.log:
                matrix = np.log(matrix)

        log.debug("any nan after remove of nan: {}".format(np.isnan(matrix).any()))
        log.debug("any inf after remove of inf: {}".format(np.isinf(matrix).any()))
        if args.log1p:
            matrix += 1
            norm = LogNorm()

        fig_height = 7
        height = 4.8 / fig_height

        fig_width = 8
        width = 5.0 / fig_width
        left_margin = (1.0 - width) * 0.5

#        fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi)
        fig = plt.figure(figsize=(5, 5), dpi=args.dpi)

        ax1 = None
        bottom = 1.3 / fig_height

        position = [left_margin, bottom, width, height]
        plotHeatmap(matrix, ma.chrBinBoundaries, fig, position,
                    args, cmap, xlabel=chrom, ylabel=chrom2,
                    start_pos=start_pos1, start_pos2=start_pos2, pNorm=norm, pAxis=ax1)

    if args.perScaffold:
        plt.tight_layout()

    plt.savefig(args.outFileName, dpi=args.dpi)
    plt.close(fig)

