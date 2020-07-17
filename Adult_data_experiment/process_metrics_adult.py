import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.core.arrayprint._line_width = 10
import os
from itertools import product
import glob
from adult import preprocess_adult_data, get_metrics
from train_clp_adult import get_consistency, predict_from_checkpoint
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.FATAL)

seeds = list(range(10))

base_dir = './tensorboard_adult_seeds/'
save_dir = './seeds_summary/'

try:
    os.makedirs(save_dir)
except:
    pass

# Selection
eps_grid = [0.01]
fe_grid = [50]
slr_grid = [5., 10., 20.]
se_grid = [50]
lr_grid = [1e-5]
lambda_grid = [5., 20., 40.]

hypers = [eps_grid, fe_grid, slr_grid, se_grid, lr_grid, lambda_grid]
# hypers = [se_grid, slr_grid, fe_grid, eps_grid, lr_grid, lambda_grid]
# names = ['eps', 'fe', 'slr', 'se', 'lr', 'lamb']

model = 'clp_fair-dim:4_adv-epoch:%d_batch_size:1000_adv-step:%.1f_l2_attack:%s_adv_epoch_full:%d_ro:%s_balanced:True_lr:%s_clp:%.1f_start:0.0_c_init:False_arch:100_%d'
 
for pack in product(*hypers):
    
    (
        eps,
        full_epoch,
        subspace_step,
        subspace_epoch,
        lr,
        lamb
    ) = pack
    
    full_step = eps/10
    names = ['se', 'slr', 'flr', 'fe', 'eps', 'lr', 'lamb']
    values = [subspace_epoch, subspace_step, str(full_step), full_epoch, str(eps), str(lr), lamb]
    
    exp_descriptor = []
    for n, v in zip(names, values):
        exp_descriptor.append(':'.join([n,str(v)]))
    
    exp_descriptor = '_'.join(exp_descriptor)
      
    n_metrics = 8
    
    result_exp = np.zeros((len(seeds), n_metrics))
    
    for seed_idx, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test, names_income, names_gender = preprocess_adult_data(seed = seed)
            
        ## Metrics
        model_dir = base_dir + model % tuple(values + [seed])

        tf.reset_default_graph()
        meta_file = glob.glob(model_dir + '/*.meta')[0]
        saver = tf.train.import_meta_graph(meta_file)
        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
        with tf.Session() as sess:
            # Restore the checkpoint
            saver.restore(sess, cur_checkpoint)
            graph = tf.get_default_graph()
            # list_of_tuples = [op.values() for op in graph.get_operations()]
            if model.startswith('sensr') or model.startswith('clp'):
                post_idx = '3:0'
            else:
                post_idx = '1:0'
            
            logits = predict_from_checkpoint(sess, 'add_' + post_idx, X_test)
            preds = np.argmax(logits, axis = 1)    
            gender_race_consistency, spouse_consistency = get_consistency(X_test, lambda x: predict_from_checkpoint(sess, 'add_' + post_idx, x))
            # print(exp_descriptor)
            # print('gender/race combined consistency', gender_race_consistency)
            # print('spouse consistency', spouse_consistency)
            acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds, verbose=False)
            seed_metrics = [acc_temp, bal_acc_temp, spouse_consistency, gender_race_consistency,
                            gender_gap_rms_temp, race_gap_rms_temp, gender_max_gap_temp, race_max_gap_temp]
            
            result_exp[seed_idx] = np.array(seed_metrics)
            
    np.save(save_dir + exp_descriptor, result_exp)
    print(50*'==')
    print(exp_descriptor)
    print('acc', 'bal_acc', 'spouse_consistency', 'gender_race_consistency',
                            'gender_gap_rms', 'race_gap_rms', 'gender_max_gap', 'race_max_gap')
    print(result_exp.mean(axis=0))
    print(50*'==')