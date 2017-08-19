#!/usr/bin/env python3
__description__ = \
"""
Calculate enrichment of peptides given their counts in an experiment with and 
without competitor added.  If added, a cluster file from counts and cluster files.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-09"
__usage__ = "enrichment.py alone_counts_file competitor_counts_file"

from . import gaussian, cluster, count
import numpy as np
import sys, argparse

def calc_enrichment(out_base,
                    alone_counts_file,
                    competitor_counts_file,
                    cluster_file=None,
                    min_counts=8,
                    noise=0.000001):
    """
    Calculate enrichments from counts files and clusters.

    out_base: string used to give output files unique names
    alone_counts_file: string. file containing counts from experiment without competitor.
    competitor_counts_file: string. file containing counts from experiment with competitor.
    cluster_file: string or None.  file containing sequence cluster.  if None, do not use
                  clusters.
    min_counts: exclude any sequence with counts less than min_counts
    noise: noise to inject into cluster enrichment (avoids numerical errors)
    """  
 
    # Read in alone and competitor freqs from count files
    alone_freq, alone_count = count.read_counts(alone_counts_file,min_counts)
    competitor_freq, competitor_count = count.read_counts(competitor_counts_file,min_counts)

    # Create a set of all unique sequences
    sequences = list(alone_freqs.keys())
    sequences.extend(competitor_freqs.keys())
    sequences = set(sequences)

    # Read in clusters if a file is specified
    if cluster_file is None:
        seq_to_cluster = {}
        cluster_to_seq = {}
    else: 
        seq_to_cluster, cluster_to_seq = cluster.read_cluster_file(cluster_file)
   
    # ---------------------- Process clusters ---------------------------------

    # Go through all clusters and record freqencies and counts for all sequences
    # in the cluster
    cluster_freq = {}
    cluster_stdev = []
    for c in cluster_to_seq.keys():

        # Skip cluster 0, which are the sequences that did not end up in a
        # cluster
        if c == 0:
            continue

        # Go through all sequences in this cluster
        cluster_freq[c] =  [0.0,0.0]
        cluster_stdev[c] = []
        for s in cluster_to_seq[c]:

            # Grab alone frequencies if sequence seen in that experiment    
            try:
                cluster_freq[c][0]  += alone_freq[s]
            except KeyError:
                pass

            # Grab competitor frequencies if sequence seen in that experiment
            try:
                cluster_freq[c][1]  += competitor_freq[s]
            except KeyError:
                pass 

            # Try to calculate individual enrichment for this sequence.  This
            # will be used to calculate the standard deviation of enrichment
            # within this cluster
            try:
                cluster_stdev[c].append(np.log(competitor_freq[s]/alone_freq[s]))
            except KeyError:
                pass

    # Record final cluster enrichment. This will only record if both the alone
    # and competitor have non-zero frequencies.
    cluster_enrichment = {}
    cluster_weight = {}
    for c in cluster_freq.keys():

        if cluster_freq[c][0] > 0.0 and cluster_freq[c][1] > 0.0:
            cluster_enrichment[c] = np.log(cluster_freq[c][1]/cluster_freq[c][0]) 
           
            # Record weight for this cluster 
            if len(cluster_stdev[c]) > 1:
                cluster_weight[c] = 1/(np.std(cluster_stdev[c])**2)
            else:
                cluster_weight[c] = np.nan

    # Find the lowest cluster weight, ignoring any nan.  If there are only nan,
    # assign everyone an even weight.
    weights = np.array(list(cluster_weight.values()))
    non_nan_weights = weights[~np.isnan(weights)]
    if len(non_nan_weights) > 0:
        lowest_weight = np.min(non_nan_weights)
    else:
        lowest_weight = 1.0

    # Assign nan weights to the lowest weight observed
    for c in cluster_weight.keys():
        if np.isnan(cluster_weight[c]):
            cluster_weight[c] = lowest_weight

    # --------------------- End processing of clusters ------------------------

    # Go through all sequences and record enrichment and weight.  First try to 
    # calculate enrichment directly from frequencies of individual sequences in 
    # the alone and competitor experiments. If this fails, grab the enrichment
    # and weight from the cluster associated with that sequence. If this fails, 
    # discard the sequence.
    seq_enrichment = {}
    seq_weight = {}
    for seq in sequences:
        
        # Try to calculate the enrichment straight up 
        try:
            enrichment = np.log(competitor_freq[s]/alone_freq[s])

            c = competitor_count[s]
            a = alone_count[s]

            c_err = np.sqrt(c)
            a_err = np.sqrt(a)

            sigma = (c/a)*np.sqrt((c_err/c)**2 + (a_err/a)**2)
            
            weight = 1/(sigma**2)

        except KeyError:

            # If this fails, try the cluster enrichment.  Inject a tiny bit of 
            # noise so a whole bunch of sequences do not have the same value.
            # This prevents a downstream numerical problem in regression.
            try:
                cluster = seq_to_cluster[seq]
                enrichment = cluster_enrichment[cluster] + np.random.normal(0,noise)
                weight = cluster_weight[cluster]
            except KeyError:
                continue
            
        seq_enrichment[seq] = enrichment
        seq_weight[seq] = weight

    # Final list of good sequences
    final_sequences = list(seq_enrichment.keys())

    # Find the probability that each sequence derives from a competitor-induced 
    # versus random binding process based on the distribution of enrichment
    # values.
    plot_name = "{}_process-model".format(out_base)
    seq_process = gaussian.find_process_probability(seq_enrichment,plot_name=plot_name)

    return seq_enrichment, seq_weight, seq_process
 
def main(argv=None):
    """
    Calculate the enrichment and write it to an output file.
    """

    if argv is None:
        argv = sys.argv[:]

    parser = argparse.ArgumentParser(description=__description__)
        
    # Positionals
    parser.add_argument("alone_counts_file",help="file with counts from experiment without competitor")
    parser.add_argument("competitor_counts_file",help="file with counts from experiment with competitor")
 
    # Optional 
    parser.add_argument("-b","--base",help="base name for output files",action="store",type=str,default="random_string")
    parser.add_argument("-f","--clusterfile",help="file containing sequence clusters",action="store",type=str,default="None")
    parser.add_argument("-n","--noise",help="noise to add to each cluster enrichment",action="store",type=float,default=0.0000001)
    parser.add_argument("-m","--mincounts",help="only include sequences with this number of counts or more",
                        action="store",type=int,default=8)

    parser.add_argument("-c","--cluster",help="use dbscan to create clusters (overrides --clusterfile)",action="store_true")
    parser.add_argument("-e","--clusterepsilon",help="epsilon to use for dbscan (requires --cluster)",action="store",type=int,default=1)
    parser.add_argument("-s","--clustersize",help="minimum cluster for dbscan (requires --cluster)",action="store",type=int,default=2)
    parser.add_argument("-d","--clusterdistance",help="cluster distance function ('dl' or 'simple') (requires --cluster)",
                        action="store",type=str,default='dl')

    parser.parse_args(argv)

    # Figure out the base string for the file
    if parser.out_base == "random_string":
        rand = str(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"),5))
        out_base = "enrich_{}".format(rand)
    else:
        out_base = parser.out_base  
 
    cluster_file = parser.clusterfile

    # If the user requests the clustering is done on-the-fly, do it here.  Use
    # all counts, regardless of min cutoff, to maximize the number of sequences
    # and more effectively cluster the space
    if parser.cluster:

        sequences = count.read_counts(parser.alone_counts_file,min_counts=0)
        sequences.extend(count.read_counts(parser.competitor_counts_file,min_counts=0))
        sequences = list(set(sequences))
   
        cluster_out_file = "{}.cluster".format(out_base) 
        cluster.cluster_seqs(sequences,
                             epsilon=parser.clusterepsilon,
                             min_neighbors=parser.clustersize
                             dist_function=parser.clusterdistance,
                             out_file=cluster_out_file)                                                            
        
        cluster_file = cluster_out_file
 
    # Calculate enrichments 
    enrich, weight, process = calc_enrichment(out_base=out_base,
                                              alone_counts_file=parser.alone_counts_file,
                                              competitor_counts_file=parser.competitor_counts_file,
                                              cluster_file=cluster_file,
                                              min_counts=parser.min_counts,
                                              noise=parser.noise)
                   
    out = []
    out.append("# alone counts file: {}".format(parser.alone_counts_file))
    out.append("# competitor counts file: {}".format(parser.competitor_counts_file))
    out.append("# cluster file: {}".format(cluster_file))
    out.append("# mininum counts: {}".format(parser.min_counts))
    out.append("# noise: {}".format(parser.noise))

    for seq in enrich.keys(): 
        out.append("{} {:20.10e} {:20.10e} {:20.10e}".format(seq,enrich[seq],weight[seq],process[seq]))
 
   
    f = open("{}.enrich".format(out_base),"w")
    f.write("\n".join(out))
    f.close()
    
 
if __name__ == "__main__":
    main()
