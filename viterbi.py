#!/usr/bin/env python3
__author__ = 'dvorkjoker'
"""
Script to train weights for features for decomposition lattice.
"""

import fst
import ast

def dot(seq1, seq2):
    assert(len(seq1) == len(seq2))
    return sum(x[0]*x[1] for x in zip(seq1, seq2))

def acceptor_from_edges(edges, weights):
    result = fst.Acceptor()
    for key, value in edges.items():
        for entry in value:
            to, label, features = entry
            result.add_arc(key, to, label, dot(features, weights))
    result[1].final = True
    return result

def plf_to_lattice(lattice):
    result = fst.Acceptor()
    edges = list(ast.literal_eval(lattice)[1:])
    end = 0
    for i, val in enumerate(edges):
        for target in val:
            result.add_arc(i, i + target[2], target[0])
            end = i + target[2]
    result[end].final = True
    return result

def all_splits(lattice):
    result = []
    for path in lattice.paths():
        result.append([lattice.isyms.find(arc.ilabel) for arc in path])
    return result

def golden_splits(s):
    if s.startswith('#'):
        # no ambiguity, just one split
        return [s.split()[1:]]
    return all_splits(plf_to_lattice(s))

def split_to_footprint(compound, split):
    result = []
    current = compound.lower()
    current_offset = 0
    for span in split:
        pos = current.find(span.lower())
        assert(pos >= 0)
        current = current[pos + len(span):]
        result.append((current_offset + pos, current_offset + pos + len(span)))
        current_offset += pos + len(span)
    return result

class Lattice(object):
    def __init__(self, arg):
        # parse from string, if needed
        if isinstance(arg, str):
            arg = ast.literal_eval(arg)

        self._lattice = arg

    def get_full(self, weights):
        return acceptor_from_edges(self._lattice, weights)

    def get_viterbi(self, weights):
        lattice = acceptor_from_edges(self._lattice, weights)
        result = lattice.shortest_path()
        result.top_sort()
        return result
