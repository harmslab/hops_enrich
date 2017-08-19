#!/usr/bin/env python3
__description__ = \
"""
Utilities for dealing with and creating sequence clusters using dbscan.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-15"

import fast_dbscan

import numpy as np

import sys, argparse

def read_cluster_file(clusters):
    """
    Read a cluster output file.  This file should have the form: 

    cluster_integer SEQUENCE1
    cluster_integer SEQUENCE2
    ...

    Returns two dictionaries. One maps sequence to cluster, the other maps 
    cluster number to sequence.
    """

    # Read file
    f = open(clusters,'r')
    lines = f.readlines()
    f.close()

    # Parse file
    seq_to_cluster = {}
    cluster_to_seq = {}
    for l in lines:

        if l.strip() == "" or l[0] == "#": 
            continue
   
        col = l.split() 
        cluster = int(col[0])
        seq = col[1]

        try:
            cluster_to_seq[cluster].append(seq)
        except KeyError:
            cluster_to_seq[cluster] = [seq]

        seq_to_cluster[seq] = cluster

    return seq_to_cluster, cluster_to_seq
       
def cluster_seqs(sequence_list,epsilon=1,min_neighbors=2,dist_function="dl",out_file=None):
    """
    Cluster a collection of sequences using fast_dbscan.
    
    sequence_list: list of strings containing sequences
    epsilon: int. neighborhood cutoff for clustering.
    min_neighbors: int. minimum number of sequence neighbors to form a cluster.
    dist_function: "simple" or "dl" -- use hamming distance or Damerau-Levenshtein
    out_file: string or None.  if string, open file and write clusters to it. 
              if None, do not write file.

    Returns dictionaries keying seq_to_cluster and cluster_to_seq.  
    """

    # Do clustering
    d = fast_dbscan.DbscanWrapper(dist_function=dist_function)
    d.load_sequences(sequence_list)
    d.run(epsilon=epsilon,min_neighbors=min_neighbors)

    # Parse results
    cluster_to_seq = d.results
    cluster_ids = list(cluster_to_seq.keys())

    seq_to_cluster = {}
    for c in cluster_ids:
        seqs = cluster_to_seq[c]
        for s in seqs:
            seq_to_cluster[s] = c

    # If requested, write out the clusters           
    if out_file is not None:
        f = open(out_file,'w')

        f.write("# epsilon: {}\n".format(epsilon))
        f.write("# min_neighbors: {}\n".format(min_neighbors))
        f.write("# dist_function: {}\n".format(dist_function))
            
        for c in cluster_ids:
            seqs = cluster_to_seq[c]
            for s in seqs:
                f.write("{} {}\n".format(c,s))

        f.close()

    return seq_to_cluster, cluster_to_seq

def main(argv=None):
    """
    If invoked from the command line.
    """
 
    if argv is None:
        argv = sys.argv[1:]
 
    parser = argparse.ArgumentParser(description=__description__)
        
    # Positionals
    parser.add_argument("sequence_file",help="file containing sequences of same length, on per line",type=str,action="store")

    # Options 
    parser.add_argument("-e","--epsilon",help="epsilon to use for dbscan",action="store",type=int,default=1)
    parser.add_argument("-s","--size",help="minimum cluster for dbscan",action="store",type=int,default=2)
    parser.add_argument("-d","--distance",help="distance function. should be either 'simple' (Hamming) or 'dl' (Damerau-Levenshtein)",
                        action="store",type=str,default="dl")

    args = parser.parse_args(argv)

    # Parse distance function
    allowable_dist = ["simple","dl"]
    if args.distance not in ["simple","dl"]:
        err = "--distance DISTANCE must be one of:\n"
        for d in allowable_dist:
            err += "    {}\n".format(d)

        err += "\n"

        raise ValueError(err)

    # Read seq file
    seq_list = []
    with open(args.sequence_file,'r') as input_file:
        for l in input_file:
            if l.strip() == "" or l[0] == "#":
                continue
            seq_list.append(l.strip())

    seq_list = list(set(seq_list))

    # Do clustering
    seq_to_cluster, cluster_to_seq = cluster_seqs(seq_list,
                                                  epsilon=args.epsilon,
                                                  min_neighbors=args.size,
                                                  dist_function=args.distance)

    # Create pretty output and send to stdout
    clusters = list(cluster_to_seq.keys())
    clusters.sort()

    out = []
    for c in clusters:
        for s in cluster_to_seq[c]:
            out.append("{} {}".format(c,s)) 
        
    return "\n".join(out)

if __name__ == "__main__":
    print(main())
