# cython: language_level = 3
# distutils: language = c++

import numpy as np
from syri.scripts.func import *
from igraph import Graph
from collections import deque
from scipy.stats import *
import logging
import sys

from libcpp.vector cimport vector as cpp_vec

cimport numpy as np
cimport cython
np.random.seed(1)

cpdef inline getOverlapWithSynBlocks(np.ndarray[np.int64_t, ndim=1] start, np.ndarray[np.int64_t, ndim=1] end, np.ndarray chrom, np.ndarray[np.int64_t, ndim=1] in_start, np.ndarray[np.int64_t, ndim=1] in_end, np.ndarray in_chrom, np.int64_t threshold, np.int64_t count, np.int64_t tUC, np.float_t tUP):

    assert(len(start) == len(end) == len(chrom) ==count)
    assert(len(in_start) == len(in_end) == len(in_chrom))

    cdef Py_ssize_t i, k, n = len(in_start), right
    cdef np.int64_t blockuni, s, e
    cdef np.ndarray[np.npy_bool, ndim = 1, cast=True] uni = np.zeros(count, dtype="bool")
    cdef np.ndarray[np.int64_t, ndim=1] grp_starts, grp_ends

    # Pre-group in-blocks by chromosome: within each group, in_start order is preserved
    # (subset of a sorted sequence maintains sort order)
    chr_groups = {}
    for k in range(n):
        c_val = in_chrom[k]
        if c_val not in chr_groups:
            chr_groups[c_val] = ([], [])
        chr_groups[c_val][0].append(in_start[k])
        chr_groups[c_val][1].append(in_end[k])

    chr_arrays = {}
    for c_val in chr_groups:
        chr_arrays[c_val] = (np.array(chr_groups[c_val][0], dtype=np.int64),
                             np.array(chr_groups[c_val][1], dtype=np.int64))

    for i in range(count):
        blockuni = 0
        s = start[i]
        e = end[i]
        c = chrom[i]

        if c in chr_arrays:
            grp_starts = chr_arrays[c][0]
            grp_ends = chr_arrays[c][1]
            # Binary search: first index in group where grp_starts >= end[i]
            right = np.searchsorted(grp_starts, e, side='left')
            for k in range(right):
                if grp_ends[k] > start[i]:
                    if (grp_starts[k] - s < threshold) and (e - grp_ends[k] < threshold):
                        s = e
                        break
                    elif grp_starts[k] < s and grp_ends[k] < e:
                        s = grp_ends[k]
                    else:
                        blockuni += grp_starts[k] - s
                        if grp_ends[k] < e:
                            s = grp_ends[k]
                        else:
                            s = e
                            break

        blockuni += (e-s)

        if (blockuni > tUC) or (blockuni > tUP*(end[i]-start[i])):
            uni[i]=True
    return uni


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline getmeblocks(long[:] astart, long[:] aend, long[:] bstart, long[:] bend, int threshold, long[:] aUni, long[:] bUni, long[:] status, long[:] aIndex, long[:] bIndex, aGroups, bGroups, long[:] clstrsize, float tdolp):
    # Function take the coordinates and cluster information of all translocated blocks and identifies mutually exclusive
    #  blocks (candidates with which a given candidate cannot co-exist) by comparing the coordinates of each block to the coordinates of the member blocks in its cluster
    logger = logging.getLogger("getmeblocks")
    cdef np.ndarray[np.int64_t, ndim=1] members, temp
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] meb, meb_a, meb_b, rem = np.zeros(len(astart), dtype="bool")
    cdef int overlap
    cdef np.int64_t i, j, index
    meblock = {}            ## for blocks which are overlapping with inplace blocks
    melist = {}             ## for blocks which are not overlapping with inplace blocks

    for i in range(len(astart)):
        if i%50000 == 0:
            logger.debug("Number of mutually exclusive blocks identified " + str(i))
        if not aUni[i] and not bUni[i]:
            rem[i] = True
        elif status[i] == 1:
            continue
        elif clstrsize[i] >= 10000:
            continue
        elif not aUni[i]:
            members = bGroups[bIndex[i]]
            meb = np.zeros(len(members), dtype="bool")              ## vector of mutually exclusive block
            for index in range(len(members)):
                j = members[index]
                if bend[j] < bstart[i]: continue
                if bstart[j] > bend[i]: break
                if j==i: continue
                if bstart[j] - threshold < bstart[i] and bend[j] + threshold > bend[i]:
                    meb[index]=True
                    continue
                overlap = min(bend[i], bend[j]) - max(bstart[i], bstart[j])
                if overlap/(bend[i] - bstart[i]) > tdolp:
                    meb[index]=True
            meblock[i] = np.array(members[meb], dtype="uint32")
        elif not bUni[i]:
            members = aGroups[aIndex[i]]
            meb = np.zeros(len(members), dtype="bool")               ## vector of mutually exclusive block
            for index in range(len(members)):
                j = members[index]
                if aend[j] < astart[i]: continue
                if astart[j] > aend[i]: break
                if j==i: continue
                if astart[j] - threshold < astart[i] and aend[j]+threshold > aend[i]:
                    meb[index] = True
                    continue
                overlap = min(aend[i], aend[j]) - max(astart[i], astart[j])
                if overlap/(aend[i] - astart[i]) > tdolp:
                    meb[index] = True
            meblock[i] = np.array(members[meb], dtype="uint32")
        else:
            members = aGroups[aIndex[i]]
            meb_a = np.zeros(len(members), dtype="bool")             ## vector of mutually exclusive block on A genome
            for index in range(len(members)):
                j = members[index]
                if aend[j] < astart[i]: continue
                if astart[j] > aend[i]: break
                if j==i: continue
                if astart[j] - threshold < astart[i] and aend[j]+threshold > aend[i]:
                    meb_a[index] = True
                    continue
                overlap = min(aend[i], aend[j]) - max(astart[i], astart[j])
                if overlap/(aend[i] - astart[i]) > tdolp:
                    meb_a[index] = True
            temp = members[meb_a]

            members = bGroups[bIndex[i]]
            meb_b = np.zeros(len(members), dtype="bool")             ## vector of mutually exclusive block on B genome
            for index in range(len(members)):
                j = members[index]
                if bend[j] < bstart[i]: continue
                if bstart[j] > bend[i]: break
                if j==i: continue
                if bstart[j] - threshold < bstart[i] and bend[j] + threshold > bend[i]:
                    meb_b[index] = True
                    continue
                overlap = min(bend[i], bend[j]) - max(bstart[i], bstart[j])
                if overlap/(bend[i] - bstart[i]) > tdolp:
                    meb_b[index] = True
            melist[i] = (np.array(temp, dtype="uint32"), np.array(members[meb_b], dtype="uint32"))
    return rem, meblock, melist


cpdef inline getConnectivityGraph(blocksList):
    from igraph import Graph
    outOG = Graph().as_directed()
    outOG.add_vertices(len(blocksList))
    if len(blocksList) == 0:
        return outOG

    ## Add edges and edge weight
    edgeList = deque()
    esWeight = deque()
    sourceList = deque()
    targetList = deque()
    for i in blocksList:
        if len(i.children) > 0:
            edgeList.extend(list(zip([i.id]*len(i.children), i.children)))
            esWeight.extend([-i.score]*len(i.children))
            sourceList.extend([i.id]*len(i.children))
            targetList.extend(i.children)
    outOG.add_edges(list(edgeList))
    outOG.es["weight"] = list(esWeight)
    outOG.es["source"] = list(sourceList)
    outOG.es["target"] = list(targetList)
    return outOG
