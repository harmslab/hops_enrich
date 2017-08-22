#!/usr/bin/env python3
__description__ = \
"""
Calculate enrichment of peptides given their counts in an experiment with and 
without competitor added. 
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-09"

from . import count, cluster

import numpy as np
import scipy.optimize, scipy.stats
from matplotlib import pyplot as plt

import sys, argparse

from multiprocessing import Process, Queue

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
    frequencies.
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
    
def _process_clusters(seq_to_cluster,seq_freq,cluster_size_cutoff=3):
    """
    Calculate mean and standard deviation of values in a ditionary.  Take log,
    as this normalizes distribution.
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
    cluster_mean = {}
    cluster_std  = {}
    for c in cluster_freq.keys():

        obs = cluster_freq[c]

        if len(obs) < cluster_size_cutoff:
            continue

        cluster_mean[c] = np.log(np.sum(np.exp(obs)))
        cluster_std[c] =   np.std(obs)

    return cluster_mean, cluster_std



def calc_enrichment(alone_counts,
                    competitor_counts,
                    seq_to_cluster=None,
                    min_counts=6,
                    noise=0.000001,
                    cluster_size_cutoff=3,
                    out_file=None,
                    header=None):
    """
    Calculate enrichments from counts files and clusters.

    out_base: string used to give output files unique names
    alone_counts: dictionary mapping sequence to counts for experiment without competitor.
    competitor_counts: dictionary mapping sequence to counts for experiment with competitor.
    seq_to_cluster: dictionary or None.  dictionary mapping sequence to cluster if None, do not use
                    clusters.
    min_counts: exclude any sequence with counts less than min_counts
    noise: noise to inject into cluster enrichment (avoids numerical errors)
    cluster_size_cutoff: minimum number of cluster members with observations before that 
                         cluster can be used.
    out_file: string or None.  file to write out to.  if None, do not write file
    header: string or None.  header to place in file.  If None, do not write header
    """  

    # Determine the frequencies of sequences in the alone and competitor data
    alone_freq, alone_std = _process_counts(alone_counts,min_counts)
    competitor_freq, competitor_std = _process_counts(competitor_counts,min_counts)

    # Grab **every** sequence, including those that do not pass our counting 
    # threshold.
    all_alone_freq, all_alone_std = _process_counts(alone_counts,0)
    all_competitor_freq, all_competitor_std = _process_counts(competitor_counts,0)

    sequences = list(all_alone_freq.keys())
    sequences.extend(all_competitor_freq.keys())
    sequences = set(sequences)

    # Load in clusters as a function of epsilon
    seq_to_cluster = []
    cluster_alone_mean = []
    cluster_alone_std = []
    cluster_competitor_mean = []
    cluster_competitor_std = []
    for i in range(1,13):

        filename = "/home/harmsm/s100-specificity-pipeline/00_phage-display/02_clusters/hA5_all_{}.txt".format(i)
        s2c, c2s = cluster.read_cluster_file(filename)
        seq_to_cluster.append(s2c)

        c_a_m, c_a_s = _process_clusters(s2c,alone_freq,cluster_size_cutoff)
        cluster_alone_mean.append(c_a_m)
        cluster_alone_std.append(c_a_s)

        c_c_m, c_c_s = _process_clusters(s2c,competitor_freq,cluster_size_cutoff)
        cluster_competitor_mean.append(c_c_m)
        cluster_competitor_std.append(c_c_s)

    # Go through all sequences and record enrichment and weight.  First try to 
    # get sequence frequency directly from sequence.  If this fails, try to get
    # frequency from increasingly coarse-grained clusters (increasing epsilon)
    seq_enrichment = {}
    seq_weight = {}
    seq_source = {} 
    for seq in sequences:

        # Get alone frequency and uncertainty
        alone_source = None
        try:
            raise KeyError() # hack to force coarse-graining
            alone = alone_freq[seq]
            alone_err = alone_std[seq]
            alone_source = 0
        except KeyError:

            epsilon = 1
            while alone_source is None:

                try:
                    c = seq_to_cluster[epsilon-1][seq]
                    alone = cluster_alone_mean[epsilon-1][c]
                    alone_err = cluster_alone_std[epsilon-1][c]  
                    alone_source = epsilon
                except KeyError:
                    epsilon += 1

        # Get competitor frequency and uncertainty
        competitor_source = None
        try:
            raise KeyError() # hack to force coarse-graining
            competitor = competitor_freq[seq]
            competitor_err = competitor_std[seq]
            competitor_source = 0
        except KeyError:

            epsilon = 1
            while competitor_source is None:

                try:
                    c = seq_to_cluster[epsilon][seq]
                    competitor = cluster_competitor_mean[epsilon-1][c]
                    competitor_err = cluster_competitor_std[epsilon-1][c]  
                    competitor_source = epsilon
                except KeyError:
                    epsilon += 1

        enrichment = competitor - alone

        sigma = np.sqrt((alone*alone_err)**2 + (competitor*competitor_err)**2)
        weight = 1/sigma

        seq_enrichment[seq] = enrichment + np.random.normal(0,noise)
        seq_weight[seq] = weight
        seq_source[seq] = "{}-{}".format(alone_source,competitor_source)

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

    if out_file is not None:

        out = []

        if header is not None:
            out.append(header)

        seq_list = list(seq_enrichment.keys())
        seq_list.sort()
        for seq in seq_list:
            out.append("{} {:20.10e} {:20.10e} {:20.10e} {:5s}".format(seq,seq_enrichment[seq],
                                                                       seq_weight[seq],
                                                                       seq_process[seq],
                                                                       seq_source[seq]))
 
        f = open(out_file,"w")
        f.write("\n".join(out))
        f.close()


    return seq_enrichment, seq_weight, seq_process

def load_counts_file(filename,phred_cutoff=15,base=""):
    """
    Load in a counts file, deciding whether to read a fastq.gz file or text
    file.
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
                                              out_file="{}.enrich".format(out_base),
                                              header=header)
    

    


 
if __name__ == "__main__":
    main()
