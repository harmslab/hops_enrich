#!/usr/bin/env python3
__description__ = \
"""
Tools for discovering how many random processes underlie enrichment values and 
determining the relative contribution of each process at each enrichment value.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-08-15"

import numpy as np
import scipy.optimize, scipy.stats
from matplotlib import pyplot as plt

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
        fit_means, fit_stds, fit_areas, ssr =  fitter(mids,prob,
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
        
