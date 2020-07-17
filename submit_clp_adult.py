from optparse import OptionParser
import os
import numpy as np
from itertools import product
from subprocess import call


def parse_args():
    
    parser = OptionParser()
    parser.set_defaults()
    
    parser.add_option("--n_seeds", type="int", dest="n_seeds")
    
    (options, args) = parser.parse_args()
 
    return options

def main(logs_dir='/dccstor/roxy1/fairness/metric_learning/logs/'):
    
    options = parse_args()
    print(options)

    n_seeds = options.n_seeds
    
    try:
        os.makedirs(logs_dir)
    except:
        pass

    seeds = list(range(n_seeds)) 

    # Grids
    eps_grid = [0.001,0.05,0.01,0.5,0.1] #5
    fe_grid = [20,40,60,80,100] #5
    slr_grid = [1,5,10,15,20] #5
    se_grid = [20,40,60,80,100] #5
    lr_grid = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5] #5

    # # Test grids
    # eps_grid = [0.001] #5
    # fe_grid = [20] #5
    # slr_grid = [1] #5
    # se_grid = [20] #5
    # lr_grid = [1e-3] #5
    
    # # Selection
    # eps_grid = [0.01]
    # fe_grid = [50]
    # slr_grid = [5., 10., 20.]
    # se_grid = [50]
    # lr_grid = [1e-5]
    
    hypers = [eps_grid, fe_grid, slr_grid, se_grid, lr_grid]
    names = ['eps', 'fe', 'slr', 'se', 'lr']

    names += ['flr', 'seed']
    
    exp_idx = 0
    
    for seed in seeds:
        for pack in product(*hypers):
            values = list(pack)

            (
                    eps,
                    full_epoch,
                    subspace_step,
                    subspace_epoch,
                    lr
            ) = pack
            
            full_step = eps/10
            values.append(full_step)
            values.append(seed)
            
            

            exp_descriptor = []
            for n, v in zip(names, values):
                exp_descriptor.append(':'.join([n,str(v)]))
                
            exp_name = '_'.join(exp_descriptor)
            print(exp_name)


            job_cmd='python ' +\
                    'adult_ccc.py ' +\
                    ' --eps ' + str(eps) +\
                    ' --fe ' + str(full_epoch) +\
                    ' --flr ' + str(full_step) +\
                    ' --se ' + str(subspace_epoch) +\
                    ' --slr ' + str(subspace_step) +\
                    ' --lr ' + str(lr) +\
                    ' --idx ' + str(exp_idx) +\
                    ' --seed ' + str(seed)
                        
            # queue = np.random.choice(['x86_6h','x86_12h', 'x86_24h'],1,p=[0.5, 0.3, 0.2])[0]
            queue = np.random.choice(['x86_12h','x86_24h', 'x86_7d'],1,p=[0.5, 0.3, 0.2])[0]
            # queue = np.random.choice(['x86_24h', 'x86_7d'],1,p=[0.65, 0.35])[0]
            
            call(['jbsub', '-proj', 'explore', '-cores', '1', '-mem', '10g', '-queue', queue,
                '-name', exp_name,
#                '-require', '(v100) && (hname != dccxc326)',
                '-out', logs_dir + exp_name + '.out',
                '-err', logs_dir + exp_name + '.err',
                job_cmd])
            
            exp_idx += 1
            
    return

if __name__ == '__main__':
    main()
