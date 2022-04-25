

import sys
sys.path.append('.')

from src.timer import Timer
import pandas as pd
import numpy as np
from src.utils import print_fl


# Cache repeated calls to MNase data
CACHED_MNASE_SPAN = None
CACHED_CHROM = None
CACHED_MNASE = None
CACHE_SPAN_LENGTH = 100000
CACHE_PADDING = 500


def filter_mnase(mnase, start=None, end=None, chrom=None, 
    sample=None, time=None, length_select=(0, 250), use_cache=False, 
    flip=False, translate_origin=None, sample_key='time'):
    """
    Filter MNase-seq data given the argument parameters, do not filter if
    not specified
    """

    global CACHED_MNASE_SPAN
    global CACHED_CHROM
    global CACHED_MNASE

    if time is not None:
        sample = time
        sample_key = 'time'

    # Use cache if parameters are valid
    if use_cache:

        if end is None or start is None:
            use_cache = False

        if end-start < CACHE_SPAN_LENGTH:
            use_cache = False

    if use_cache:

        reset_cache = False

        # If not caching yet
        if CACHED_MNASE_SPAN is None:
            reset_cache = True

        # Already caching
        else:

            # Check if search span is outside of cached span
            if (end > CACHED_MNASE_SPAN[1] or start < CACHED_MNASE_SPAN[0] 
                or CACHED_CHROM != chrom):
                reset_cache = True

        # If cache needs to be reset, get MNase and set cache parameters
        if reset_cache:

            CACHED_MNASE_SPAN = start, start+CACHE_SPAN_LENGTH
            cached_mnase_search_span = (CACHED_MNASE_SPAN[0]-CACHE_PADDING, 
                                        CACHED_MNASE_SPAN[1]+CACHE_PADDING)
            CACHED_CHROM = chrom

            # Set new cache of MNase
            CACHED_MNASE = filter_mnase(mnase, start=cached_mnase_search_span[0], 
                chrom=CACHED_CHROM, end=cached_mnase_search_span[1], use_cache=False)

        # MNase data we are going to search through will be the cached data
        mnase = CACHED_MNASE

    select = ((mnase.length >= length_select[0]) &
              (mnase.length < length_select[1]))

    if start is not None and end is not None:
        select = ((mnase.mid < end) & 
                  (mnase.mid >= start)) & select

    if chrom is not None:
        select = (mnase.chr == chrom) & select
 
    if sample is not None:
        select = select & (mnase[sample_key] == sample)

    ret_mnase = mnase[select].copy()

    # translate mid positions to translated point
    if translate_origin is not None:
        ret_mnase.mid = ret_mnase.mid - translate_origin
        ret_mnase.start = ret_mnase.start - translate_origin
        ret_mnase.stop = ret_mnase.stop - translate_origin

        # flip across origin if needed
        if flip: 
            ret_mnase.mid = -ret_mnase.mid
            old_start = ret_mnase.start.values.copy()
            ret_mnase.start = -ret_mnase.stop.copy()
            ret_mnase.stop = -old_start

    return ret_mnase
