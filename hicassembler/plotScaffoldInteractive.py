from __future__ import division

import matplotlib
matplotlib.use("TkAgg")

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


    parser.add_argument('--clearMaskedBins',
                        help='if set, masked bins are removed from the matrix',
                        action='store_true')


    parser.add_argument('--region',
                        help='Plot only this region. The format is '
                        'chr:start-end The plotted region contains '
                        'the main diagonal and is symmetric unless '
                        ' --region2 is given',
                        required=True)

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
        region_end = 1e15  # very large number

    return chrom, region_start, region_end


def getRegion(args, ma):
    chrom = region_start = region_end = idx1 = start_pos1 = chrom2 = region_start2 = region_end2 = idx2 = start_pos2 = None
    chrom, region_start, region_end = translate_region(args.region)

    if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
        chrom = toBytes(chrom)

    if chrom not in list(ma.interval_trees):

        if type(next(iter(ma.interval_trees))) in [np.bytes_, bytes]:
            chrom = toBytes(chrom)

        if chrom not in list(ma.interval_trees):
            exit("The contig/scaffold name '{}' given in --region is not part of the Hi-C matrix. "
                 "Check spelling".format(chrom))

    args.region = [chrom, region_start, region_end]

    idx1, start_pos1 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom and
                             x[1] >= region_start and x[2] < region_end])
    idx2 = idx1
    chrom2 = chrom
    start_pos2 = start_pos1

    return chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2


def main(args=None):
    args = parse_arguments().parse_args(args)

    ma = HiCMatrix.hiCMatrix(args.matrix)
    if args.clearMaskedBins:
        ma.maskBins(ma.nan_bins)

    chrom, region_start, region_end, idx1, start_pos1, chrom2, region_start2, region_end2, idx2, start_pos2 = getRegion(args, ma)
    matrix = np.asarray(ma.matrix[idx1, :][:, idx2].todense().astype(float))

    cmap = cm.get_cmap(args.colorMap)
    log.debug("Nan values set to black\n")
    cmap.set_bad('black')

    mask = matrix == 0
    matrix[mask] = np.nanmin(matrix[mask == False])

    if np.isnan(matrix).any() or np.isinf(matrix).any():
        log.debug("any nan {}".format(np.isnan(matrix).any()))
        log.debug("any inf {}".format(np.isinf(matrix).any()))
        mask_nan = np.isnan(matrix)
        mask_inf = np.isinf(matrix)
        matrix[mask_nan] = np.nanmin(matrix[mask_nan == False])
        matrix[mask_inf] = np.nanmin(matrix[mask_inf == False])

    log.debug("any nan after remove of nan: {}".format(np.isnan(matrix).any()))
    log.debug("any inf after remove of inf: {}".format(np.isinf(matrix).any()))

    matrix += 1
    norm = LogNorm()

    fig_height = 7
    height = 4.8 / fig_height

    fig_width = 8
    width = 5.0 / fig_width
    left_margin = (1.0 - width) * 0.5

    fig = plt.figure(figsize=(5, 5), dpi=args.dpi)

    bottom = 1.3 / fig_height

    position = [left_margin, bottom, width, height]

    log.debug("plotting heatmap")

    if matrix.shape[0] < 5:
        log.info("Matrix for {} too small to plot. Matrix size: {}".format(ma.chrBinBoundaries.keys()[0], matrix.shape))
        return

    ax = fig.add_axes(position)

    if start_pos1 is None:
        start_pos1 = np.arange(matrix.shape[0])
    if start_pos2 is None:
        start_pos2 = start_pos1

    xmesh, ymesh = np.meshgrid(start_pos1, start_pos2)
    ax.set_title(chrom)
    ax.pcolormesh(xmesh.T, ymesh.T, matrix, vmin=args.vMin, vmax=args.vMax, cmap=cmap, norm=norm)
    ax.invert_yaxis()

    ax.get_xaxis().set_tick_params(which='both', bottom='on', direction='out')
    ax.get_yaxis().set_tick_params(which='both', bottom='on', direction='out')

    plt.show()


