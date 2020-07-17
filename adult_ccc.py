import tensorflow as tf
import numpy as np
import os
from adult import preprocess_adult_data, get_sensitive_directions_and_projection_matrix, get_consistency, get_metrics
from train_clp_adult import train_fair_nn
from optparse import OptionParser
from sklearn.utils.random import sample_without_replacement
from mle_sgd import proximal_gd_sigmoid

def generate_pairs(len1, len2, n_pairs=100):
    """
    vanilla sampler of random pairs (might sample same pair up to permutation)
    n_pairs > len1*len2 should be satisfied
    """
    idx = sample_without_replacement(len1*len2, n_pairs)
    return np.vstack(np.unravel_index(idx, (len1, len2)))

def parse_args():
    
    parser = OptionParser()

    # SenSR parameters
    parser.add_option("--eps", type="float", dest="eps")
    parser.add_option("--fe", type="int", dest="full_epoch")
    parser.add_option("--flr", type="float", dest="full_step")
    parser.add_option("--se", type="int", dest="subspace_epoch")
    parser.add_option("--slr", type="float", dest="subspace_step")
    parser.add_option("--lr", type="float", dest="lr")
    parser.add_option("--idx", type="int", dest="idx")
    
    parser.add_option("--seed", type="int", dest="seed")
    
    (options, args) = parser.parse_args()
 
    return options

def main(out_folder = './metrics/'):
    
    try:
        os.makedirs(out_folder)
    except:
        pass
    
    options = parse_args()
    print(options)

    ro = options.eps
    adv_epoch_full = options.full_epoch
    l2_attack = options.full_step
    adv_epoch = options.subspace_epoch
    adv_step = options.subspace_step
    lr = options.lr
    idx = options.idx
    
    seed = options.seed

    
    X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test, names_income, names_gender = preprocess_adult_data(seed = seed)
    
    sensitive_directions, _ = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)
    
    ## EXPLORE metric
    K = y_train.shape[1]
    comparable_pairs = None
    for i in range(K):
        c0_idx = np.where((y_gender_train[:,0] + y_train[:,i])==2)[0]
        c1_idx = np.where((y_gender_train[:,1] + y_train[:,i])==2)[0]
        pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=10000)
        comparable_pairs_now = X_train[c0_idx[pairs_idx[0]]]- X_train[c1_idx[pairs_idx[1]]]
        if comparable_pairs is None:
            comparable_pairs = comparable_pairs_now
        else:
            comparable_pairs = np.vstack((comparable_pairs, comparable_pairs_now))
    
    
    # incomparable pairs are simply sampled from different classes
    c0_idx = np.where(y_train[:,0])[0]
    c1_idx = np.where(y_train[:,1])[0]
    pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=5000)
    incomp_pairs_new = [X_train[c0_idx[pairs_idx[0]]], X_train[c1_idx[pairs_idx[1]]]]
    Xnew_incomp = incomp_pairs_new[0] - incomp_pairs_new[1]
           
    X_pairs = np.vstack((comparable_pairs, Xnew_incomp))
    Y_pairs = np.zeros(X_pairs.shape[0])
    Y_pairs[:comparable_pairs.shape[0]] = 1
    
    Sigma_fair_mle = proximal_gd_sigmoid(X_pairs,Y_pairs,mbs=1000,maxiter=10000)

    tf.reset_default_graph()
    fair_info = [y_gender_train, y_gender_test, names_income[0], names_gender[0], sensitive_directions, Sigma_fair_mle]
    weights, train_logits, test_logits, _  = train_fair_nn(X_train, y_train, tf_prefix='explore', adv_epoch_full=adv_epoch_full, l2_attack=l2_attack,
                                              adv_epoch=adv_epoch, ro=ro, adv_step=adv_step, plot=True, fair_info=fair_info, balance_batch=True, 
                                              X_test = X_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                              n_units=[100], l2_reg=0, epoch=30000, batch_size=1000, lr=lr, lambda_clp=0,
                                              fair_start=0., counter_init=False, seed=seed)
    
    ## Metrics
    preds = np.argmax(test_logits, axis = 1)    
    gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights)
    print('gender/race combined consistency', gender_race_consistency)
    print('spouse consistency', spouse_consistency)
    acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)
       
    params = [adv_epoch, adv_epoch_full, adv_step, lr, ro]
    relevant_val = [gender_race_consistency, spouse_consistency, acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp]
    np.save(out_folder + 'relevant_val_%a' % idx, np.array(params + relevant_val))
    
    return

if __name__ == '__main__':
    main()