import numpy as np
import pandas as pd

def generate_bids(A, B, bid_upper_bound):
    bids = np.random.rand(A,B)
    bids *= bid_upper_bound
    bids = np.round(bids,0)
    return(bids)

def generate_gamma(bids):
    Gamma = []
    A,B = bids.shape
    for i in range(A):
        for j in range(B):
            for k in range(B):
                gamma = bids[i,j]/bids[i,k]
                Gamma.append(gamma)
    Gamma_un = sorted(list(set(Gamma)))
    Gamma_un = np.round(np.array(Gamma_un),2)
    Gamma_un = Gamma_un[Gamma_un>=1]
    return(Gamma_un)

def calc_revenue(bids, boosts, reserves):
    '''
    Bids, boosts, and reserves are all numpy arrays.
    '''
    A,B = bids.shape
    
    boosted_bids = np.multiply(bids, boosts)
    sorted_boosted_bids_id = np.argsort(boosted_bids, axis=1)
    sorted_boosted_bids = np.sort(boosted_bids, axis=1)

    winner_reserves = np.array([reserves[i] for i in sorted_boosted_bids_id[:,-1]])
    boosted_revs = np.array(sorted_boosted_bids[:,-2]/[boosts[i] for i in sorted_boosted_bids_id[:,-1]])
    
    revs = np.maximum(winner_reserves, boosted_revs)
    return(np.sum(revs))

def greedy_one_bidder(bids, reserves, initial_boosts, bidder_id, lamb, verbose=False):
    '''
    We have to consider a large numbers of Lambda values
    '''
    Gamma = generate_gamma(bids)

    boosts = initial_boosts.copy()
    boost = initial_boosts[bidder_id]
    
    rev = calc_revenue(bids, initial_boosts, reserves)
    obj = calc_revenue(bids, initial_boosts, reserves)

    if verbose:
        print(f'Initial revenue is {obj}')
    for gamma in Gamma:
        boosts[bidder_id] = gamma
        cur_rev = calc_revenue(bids, boosts, reserves)
        cur_obj = cur_rev - lamb*(gamma-initial_boosts[bidder_id])**2
        if cur_obj > obj:
            boost = gamma
            rev = cur_rev
            obj = cur_obj
    
    if verbose:
        print(f'Final revenue is {obj}')
    return(boost, rev)

def BSPAM(bids, reserves, initial_boosts, lamb, epsilon):
    A,B = bids.shape
    prev_boosts = np.array(initial_boosts).astype('float64') 
    current_boosts = np.array(initial_boosts).astype('float64') 
        
    #First Iteration
    for i in range(B):
        boost, rev = greedy_one_bidder(bids, reserves, current_boosts, i, lamb)
        current_boosts[i] = boost
    
    while (calc_revenue(bids, current_boosts, reserves) - calc_revenue(bids, prev_boosts, reserves))**2 > epsilon:
        prev_boosts = current_boosts.copy()
        for i in range(B):
            boost, rev = greedy_one_bidder(bids, reserves, prev_boosts, i, lamb, False)
            current_boosts[i] = boost
    
    return(current_boosts, rev)

