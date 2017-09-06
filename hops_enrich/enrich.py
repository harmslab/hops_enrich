#!/usr/bin/env python3
__description__ = \
"""
Calculate enrichment of peptides given their counts in an experiment with and 
without competitor added.  This can also coarse-grain this calculation and 
calculate enrichment of clusters and assign those enrichments to individual 
cluster members.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-09"

from . import count, cluster

import numpy as np
import scipy.optimize, scipy.stats
from matplotlib import pyplot as plt

import sys, argparse

def _multi_gaussian(x,means,stds,areas):
    """
    Function calculating multiple gaussians (built from
    values in means, stds, areas).  The number of gaussians
    is determined by the length of means, stds, and areas.
    The gaussian functions are calculated at values in
    array x. 
    """
    
    if len(means) != len(stds) or len(means) != len(areas):
        err = "means, standard deviations and areas should have the same length!\n"
        raise ValueError(err)
    
    # Decide if out should be a single value or array
    try:
        out = np.zeros(len(x),dtype=float)
    except TypeError:
        out = 0.0
    
    for i in range(len(means)):
        out += areas[i]*scipy.stats.norm(means[i],stds[i]).pdf(x)

    return out

def _multi_gaussian_r(params,x,y):
    """
    Residuals function for multi_guassian. 
    """
    
    params = np.array(params)
    if params.shape[0] % 3 != 0:
        err = "num parameters must be divisible by 3\n"
        raise ValueError(err)
    
    means = params[np.arange(0,len(params),3)]
    stds = params[np.arange(1,len(params),3)]
    areas = params[np.arange(2,len(params),3)]
    
    return _multi_gaussian(x,means,stds,areas) - y

def _fitter(x,y,means_guess,stds_guess,areas_guess):
    """
    Fit an arbitrary number of gaussian functions to x/y data.
    The number of gaussians that will be fit is determined by
    the length of means_guess.  
    
    x: measurement x-values (array)
    y: measurement y-values (array)
    means_guess: array of guesses for means for gaussians.  
                 length determines number of gaussians
    stds_guess: array of guesses of standard deviations for
                gaussians. length must match means_guess
    areas_guess: array of area guesses for gaussians.  
                 length must match means guess.
                 
    returns: means, stds, areas and fit sum-of-squared-residuals
    """
    
    # Sanity check
    if len(means_guess) != len(stds_guess) or len(means_guess) != len(areas_guess):
        err = "means, standard deviations and areas should have the same length!\n"
        raise ValueError(err)
    
    # Construct an array of parameter guesses by assembling
    # means, stds, and areas
    param_guesses = []
    for i in range(len(means_guess)):
        param_guesses.append(means_guess[i])
        param_guesses.append(stds_guess[i])
        param_guesses.append(areas_guess[i])
    param_guesses = np.array(param_guesses)
    
    # Fit the multigaussian function
    fit = scipy.optimize.least_squares(_multi_gaussian_r,param_guesses,
                                       args=(x,y))
    
    # Disassemble into means, stds, areas
    means = fit.x[np.arange(0,len(fit.x),3)]
    stds = fit.x[np.arange(1,len(fit.x),3)]
    areas = fit.x[np.arange(2,len(fit.x),3)]
    
    return means, stds, areas, fit.cost
    
def _plot_gaussians(x,means,stds,areas):
    """
    Plot a collection of gaussians.
    
    means: array of means for gaussians.  
           length determines number of gaussians
    stds: array of standard deviations for gaussians.
          length must match means_guess
    areas: array of areas for gaussians.  
           length must match means guess.
    """
    
    plt.plot(x,_multi_gaussian(x,means,stds,areas))
    

    for i in range(len(means)):
        plt.plot(x,_multi_gaussian(x,
                                  [means[i]],
                                  [stds[i]],
                                  [areas[i]]))
        
def _calc_aic(ssr_list,k_list,num_obs):
    """
    Calculate an AIC between models.  
    
    ssr_list: sum of square residuals for each model
    k_list: degrees of freedom for each model
    num_obs: number of observations made
    """

    aic_list = []
    for i in range(len(ssr_list)):
        aic_list.append(num_obs*np.log(ssr_list[i]) + 2*(k_list[i] + 1))
    
    aic_list = np.array(aic_list)
    delta_list = aic_list - np.min(aic_list)
    
    Q = np.exp(-delta_list/2)
    return Q/np.sum(Q)

def find_process_probability(enrich_dict,breaks=None,plot_name=None):
    """
    For each sequence in enrich dict, determine the relative probability
    that it arose from a gaussian process, relative to a different 
    process.
    
    enrich_dict: dictionary of enrichments keyed to sequences
    breaks: tuple or None.  If tuple, should be (min,max,num_steps).  If None,
            breaks are set to (min(value)-10%,max(value)+10%,50)
    plot_name: string or None.  If string, writes file with that name.  If None,
               no plot is written.
               
    Fits distribution of enrichments to multiple Gaussians.  Decides number of 
    Gaussians using an AIC test. Will throw error if it detects more than two
    Gaussians in the distribution or if the best fit includes a negative Gaussian.
    """

    values = np.array(list(enrich_dict.values()))
    
    # Construct histogram breaks if not specified
    if breaks is None:
        ten_pct = (np.max(values) - np.min(values))*0.1
        breaks = (np.min(values) - ten_pct,np.max(values) + ten_pct,40)
   
    # Construct histogram of frequencies
    counts, bins = np.histogram(values,bins=np.linspace(*breaks))
    mids = (bins[1:]+bins[:-1])/2
    prob = (counts/np.sum(counts))/(bins[1]-bins[0])

    ssr_list = []
    num_params = []
    model_results = []
    
    # Fit 1 and 2 gaussian models
    for i in range(1,3):

        # Initialize guesses
        means_guess = -5*np.arange(i) #guesses are 0; -5,0
        stds_guess = np.ones(i,dtype=float)
        areas_guess = np.ones(i,dtype=float)

        # Do fit
        fit_means, fit_stds, fit_areas, ssr =  _fitter(mids,prob,
                                                       means_guess,
                                                       stds_guess,
                                                       areas_guess)
    
        # Record fit results
        ssr_list.append(ssr)
        num_params.append((i+1)*3)
        model_results.append((fit_means,fit_stds,fit_areas))
           
    # Find best model by AIC
    aic_weights = _calc_aic(ssr_list,num_params,len(mids))
    
    model_to_choose = [(w,i) for i, w in enumerate(aic_weights)]
    model_to_choose.sort()
    model_to_choose = model_to_choose[-1]

    best_model = model_results[model_to_choose[1]]
    
    # Grab best model parameters
    means = best_model[0]
    stds = best_model[1]
    areas = best_model[2]

    # If the user specifies a plot_name, make one
    if plot_name is not None:
        plt.plot(mids,prob,"o")
        bin_range = np.linspace(min(mids),max(mids),200)
        _plot_gaussians(bin_range,means,stds,areas)
        plt.savefig("{}.pdf".format(plot_name))
        plt.show()
    
    # Make sure the best model has only negative peaks
    for a in areas:
        if a < 0:
            err = "Best fit has a negative peak.\n"
            raise ValueError(err)
    
    # Find peak with the lowest mean.  This is defined as the peak of
    # interest
    peak_of_interest = np.where(np.min(means) == means)[0][0]
    
    # Go through all sequences in the original dictionary and assign a
    # a weight to that sequence
    out_dict = {}
    for seq in enrich_dict.keys():    
        
        enrich = enrich_dict[seq] 
    
        if len(means) == 1:
            weight = 1.0
        else:
            m_weights = np.zeros(len(means),dtype=float)
            for i in range(len(means)):
                m_weights[i] = areas[i]*scipy.stats.norm(means[i],stds[i]).pdf(enrich)
            weight = m_weights[peak_of_interest]/np.sum(m_weights)
            
        out_dict[seq] = weight
            
    return out_dict

def _process_counts(count_dict,min_counts=1):
    """
    Process counts dictionary, removing low-count sequences and determining 
    frequencies.  Returns log frequency and estimate of the standard deviation
    of that frequency based on std ~ sqrt(N).

    count_dict: dictionary containing sequences keying counts as values
    min_counts: only include sequences with counts above min_counts
    """
    
    filtered_counts = {}
    for k in count_dict.keys():
        if count_dict[k] >= min_counts:
            filtered_counts[k] = count_dict[k]

    freq = {}
    std = {}
    total = 1.0*np.sum(list(filtered_counts.values()))
    for k in filtered_counts.keys():

        f = filtered_counts[k]/total

        s = np.sqrt(filtered_counts[k])/total
        s = np.abs(np.log(f + s) - np.log(f))

        freq[k] = np.log(f)
        std[k] = s

    return freq, std
    
def _process_clusters(seq_to_cluster,seq_freq,cluster_size_cutoff=0):
    """
    Determine the frequency of a given cluster in the experiment.  This is the
    sum of the frequency of each cluster member.  Return the log of the cluster
    frequency and standard deviation of the frequencies of cluster members.
    
    seq_to_cluster: dictionary mapping sequences to their cluster
    seq_freq: dictionary mapping sequence to its frequency in the experiment
    cluster_sie_cutoff: integer. if a cluster has fewer than this number of 
                        members, ignore it.
    """

    # Create a dictionary mapping clusters to sequences
    cluster_freq = {}
    for s in seq_to_cluster.keys():

        # Only consider sequences with measured frequencies
        try:
            freq = seq_freq[s]
        except KeyError:
            continue
           
        c = seq_to_cluster[s]
        try:
            cluster_freq[c].append(freq)
        except KeyError:
            cluster_freq[c] = [freq]

    # Find mean and standard deviation of frequencies of peptides in this 
    # cluster.  If there are  less than cluster_size_cutoff observations,
    # do not record it
    cluster_total_freq = {}
    cluster_std  = {}
    for c in cluster_freq.keys():

        # Ignore the 0 cluster --> cluster of non-clustered sequences
        if c == 0:
            continue

        obs = cluster_freq[c]

        if len(obs) < cluster_size_cutoff:
            continue

        cluster_total_freq[c] = np.log(np.sum(np.exp(obs)))
        cluster_std[c] =   np.std(obs)

    return cluster_total_freq, cluster_std



def calc_enrichment(alone_counts,
                    competitor_counts,
                    seq_to_cluster=None,
                    min_counts=6,
                    noise=0.000001,
                    cluster_size_cutoff=0,
                    out_file=None,
                    header=None):
    """
    Calculate enrichments from counts files and clusters.

    alone_counts: dictionary mapping sequence to counts for experiment without competitor.
    competitor_counts: dictionary mapping sequence to counts for experiment with competitor.
    seq_to_cluster: dictionary or None.  dictionary mapping sequence to cluster if None, do not use
                    clusters.
    min_counts: exclude any sequence with counts less than min_counts
    noise: noise to inject into cluster enrichment (avoids numerical errors)
    cluster_size_cutoff: minimum number of cluster members with observations before that 
                         cluster is used.
    out_file: string or None.  file to write out to.  if None, do not write file
    header: string or None.  header to place in file.  If None, do not write header
    """  

    if seq_to_cluster is None:
        seq_to_cluster = {}

    # Determine the frequencies of sequences in the alone and competitor data
    alone_freq, alone_std = _process_counts(alone_counts,min_counts)
    competitor_freq, competitor_std = _process_counts(competitor_counts,min_counts)

    # Grab **every** sequence, including those that do not pass our counting 
    # threshold.  Sequences in each cluster, even those not seen, will be 
    # assigned the cluster frequency
    all_alone_freq,      _ = _process_counts(alone_counts,0)
    all_competitor_freq, _ = _process_counts(competitor_counts,0)

    sequences = list(all_alone_freq.keys())
    sequences.extend(all_competitor_freq.keys())
    sequences = set(sequences)

    # Load in clusters as a function of epsilon
    cluster_alone_freq, cluster_alone_std = _process_clusters(seq_to_cluster,
                                                              alone_freq,
                                                              cluster_size_cutoff)
    cluster_competitor_freq, cluster_competitor_std = _process_clusters(seq_to_cluster,
                                                                        competitor_freq,
                                                                        cluster_size_cutoff)

    # Go through all sequences seen anywhere and get enrichment values.
    # First try to get their enrichment from their cluster.  If this fails, try
    # to get them individually (i.e. they are their own cluster).  
    seq_enrichment = {}
    seq_weight = {}
    seq_source = {}
    seq_seqlevel_e = {}
    for seq in sequences:
           
        # First try to grab cluster sequences 
        source = None    
        try:
            c = seq_to_cluster[seq]

            alone = cluster_alone_freq[c]
            alone_err = cluster_alone_std[c]  

            competitor = cluster_competitor_freq[c]
            competitor_err = cluster_competitor_std[c]  

            source = "c"
        
        except KeyError:
 
            # Try singles -- cluster with only one sequence
            try:

                alone = alone_freq[seq]
                alone_err = alone_std[seq]

                competitor = competitor_freq[seq]
                competitor_err = competitor_std[seq]

    
                source = "s"
            
            except KeyError:
                pass

        # Either alone or competitor could not be assigned
        if source is None:
            continue

        # Record individual freq, if calculable
        try:
            seq_seqlevel_e[seq] = competitor_freq[seq] - alone_freq[seq]
        except KeyError:
            pass

        # Determine enrichments
        enrichment = competitor - alone
        sigma = np.sqrt((alone*alone_err)**2 + (competitor*competitor_err)**2)
        weight = 1/sigma

        seq_enrichment[seq] = enrichment + np.random.normal(0,noise)
        seq_weight[seq] = weight
        seq_source[seq] = source

    # Final list of good sequences
    final_sequences = list(seq_enrichment.keys())

    # ----------------------- Noramlize weights ----------------------------- #

    # Find the lowest weight, ignoring any nan.  If there are only nan,
    # assign everyone an even weight.
    weights = np.array(list(seq_weight.values()))
    non_nan_weights = weights[~np.isnan(weights)]
    if len(non_nan_weights) > 0:
        lowest_weight = np.min(non_nan_weights)
    else:
        lowest_weight = 1.0

    # Assign nan weights to the lowest weight observed
    for c in seq_weight.keys():
        if np.isnan(seq_weight[c]):
            seq_weight[c] = lowest_weight
       
    # Normalize to one
    total = np.sum(list(seq_weight.values()))
    for c in seq_weight.keys():
        seq_weight[c] = seq_weight[c]/total
 
    # Find the probability that each sequence derives from a competitor-induced 
    # versus random binding process based on the distribution of enrichment
    # values.
    seq_process = find_process_probability(seq_enrichment,plot_name=out_file)

    # Write output file
    if out_file is not None:

        out = []

        if header is not None:
            out.append(header)

        seq_list = list(seq_enrichment.keys())
        seq_list.sort()

        local_header = ["#"]
        seq_fmt = "{{:>{:d}}}".format(len(seq_list[0])-1)
        local_header.append(seq_fmt.format("seq"))
        local_header.append(" {:>20s} {:>20s} {:>20s} {:>20s} {:5}".format("E","weight","process","E_from_seq","src"))
        out.append("".join(local_header))

        for seq in seq_list:
            try:
                seqlevel = seq_seqlevel_e[seq]
            except KeyError:
                seqlevel = np.nan
            out.append("{} {:20.10e} {:20.10e} {:20.10e} {:20.10e} {:5s}".format(seq,seq_enrichment[seq],
                                                                       seq_weight[seq],
                                                                       seq_process[seq],
                                                                       seqlevel,
                                                                       seq_source[seq]))
 
        f = open(out_file,"w")
        f.write("\n".join(out))
        f.close()


    return seq_enrichment, seq_weight, seq_process

def load_counts_file(filename,phred_cutoff=15,base=""):
    """
    Load in a counts file, deciding whether to read a fastq.gz file or text
    file.

    filename: fastq.gz file or text file with counts of the form:
    
            SEQUENCE1 counts1
            SEQUENCE2 counts2 
            ....

    phred_cutoff: only reads with a mean phred score greaater than phred_cutoff
                  included
    
    base: string or None. If string, use as base for output file. If None, do not
          write output file.
    """

    # First detremine the file type
    with open(filename,'rb') as f:
        file_start = f.read(3)

    # gz file --> treat as fastq
    if file_start.startswith(b"\x1f\x8b\x08"):
        out_file = "_".join([base,"{}.counts".format(filename)])
        count_dict = count.fastq_to_count(filename,phred=phred_cutoff,out_file=out_file)

    # text file --> treat as counts file
    else:
        count_dict = count.read_counts(filename)
     
    return count_dict
 
def main(argv=None):
    """
    Calculate the enrichment and write it to an output file.
    """
    
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
        
    # Positionals
    parser.add_argument("alone_counts_file",help="file with counts from experiment without competitor (fastq.gz or text)")
    parser.add_argument("competitor_counts_file",help="file with counts from experiment with competitor (fastq.gz or text)")
 
    # Optional 
    parser.add_argument("-b","--base",help="base name for output files",action="store",type=str,default="random_string")
    parser.add_argument("-f","--clusterfile",help="file containing sequence clusters",action="store",type=str,default=None)
    parser.add_argument("-n","--noise",help="noise to add to each cluster enrichment",action="store",type=float,default=0.0000001)
    parser.add_argument("-m","--mincounts",help="only include sequences with this number of counts or more",
                        action="store",type=int,default=8)
    parser.add_argument("-a","--clustcut",help="only use frequencies with CLUSTCUT or more members to calculate enrichment",
                        action="store",type=int,default=0)

    parser.add_argument("-c","--cluster",help="use dbscan to create clusters (overrides --clusterfile)",action="store_true")
    parser.add_argument("-e","--cluster_epsilon",help="epsilon to use for dbscan (requires --cluster)",action="store",type=int,default=1)
    parser.add_argument("-s","--cluster_size",help="minimum cluster for dbscan (requires --cluster)",action="store",type=int,default=2)
    parser.add_argument("-d","--cluster_distance",help="cluster distance function ('dl' or 'simple') (requires --cluster)",
                        action="store",type=str,default='dl')

    parser.add_argument("-p","--fastq_phred",help="phred cutoff to use for processing fastq files",action="store",type=int,default=15)

    args = parser.parse_args(argv)

    # Figure out the base string for the file
    if args.base == "random_string":
        rand = list(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"),5))
        out_base = "enrich_{}".format("".join(rand))
    else:
        out_base = args.base  

    # Read in counts
    alone_counts = load_counts_file(args.alone_counts_file,args.fastq_phred,args.base)
    competitor_counts = load_counts_file(args.competitor_counts_file,args.fastq_phred,args.base)

    # If the user requests the clustering is done on-the-fly, do it here.  Use
    # all counts, regardless of min cutoff, to maximize the number of sequences
    # and more effectively cluster the space
    if args.cluster:

        sequences = list(alone_counts.keys())
        sequences.extend(list(alone_counts.keys()))
        sequences = list(set(sequences))
   
        cluster_out_file = "{}.cluster".format(out_base) 
        seq_to_cluster, cluster_to_seq = cluster.cluster_seqs(sequences,
                                                              epsilon=args.cluster_epsilon,
                                                              min_neighbors=args.cluster_size,
                                                              dist_function=args.cluster_distance,
                                                              out_file=cluster_out_file) 
    else:
        seq_to_cluster = None
        if args.clusterfile is not None:
            seq_to_cluster, cluster_to_seq = cluster.read_cluster_file(args.clusterfile)
        
    header = []
    header.append("# alone counts file: {}".format(args.alone_counts_file))
    header.append("# competitor counts file: {}".format(args.competitor_counts_file))
    if args.alone_counts_file[-3:] == ".gz" or args.competitor_counts_file[-3:] == ".gz":
        header.append("# phred cutoff: {}".format(args.fastq_phred))

    header.append("# mininum counts: {}".format(args.mincounts))
    header.append("# noise: {}".format(args.noise))
    
    if args.cluster:
        header.append("# cluster?: yes")
        header.append("#    cluster epsilon: {}".format(args.cluster_epsilon))
        header.append("#    cluster size: {}".format(args.cluster_size))
        header.append("#    cluster distance function: {}".format(args.cluster_distance))
    else:
        if args.clusterfile is not None:
            header.append("# cluster file: {}".format(args.clusterfile))

    header = "\n".join(header)

    # Calculate enrichments 
    enrich, weight, process = calc_enrichment(alone_counts,
                                              competitor_counts,
                                              seq_to_cluster=seq_to_cluster,
                                              min_counts=args.mincounts,
                                              noise=args.noise,
                                              cluster_size_cutoff=args.clustcut,
                                              out_file="{}.enrich".format(out_base),
                                              header=header)
    

if __name__ == "__main__":
    main()
