import tensorflow as tf
import glob
import re
from collections import defaultdict

def filter_event(event):
    for v in event.summary.value:
        if v.tag == "Test gap RMS ":
            if v.simple_value > 0.055:
                return False
        if v.tag == "spouse consistency":
            if v.simple_value < 0.93:
                return False
        if v.tag == "Test balanced TPR":
            if v.simple_value < 0.78:
                return False
    
    return True

def filter_experiment(name):
    vals = parse_name(name)
    if float(vals['adv-step']) < 5.:
        return False
    if int(vals['adv-epoch']) != 50:
        return False
    if int(vals['adv_epoch_full']) != 50:
        return False
    return True
    
def parse_name(name, result_dict=None):
    cur = {}
    values = re.match(TEMPLATE, name)
    for t,v in zip(TAGS, values.groups()):
        cur[t] = v
        if result_dict is not None:
            if v not in result_dict[t]:
                result_dict[t].append(v)
    return cur
    
TEMPLATE = r'(.*)_fair-dim:(.*)_adv-epoch:(.*)_batch_size:(.*)_adv-step:(.*)_l2_attack:(.*)_adv_epoch_full:(.*)_ro:(.*)_balanced:(.*)_lr:(.*)_clp:(.*)_start:(.*)_c_init:(.*)_arch:(.*)_(.*)'
TAGS = ['name'] + [s[1:-1] for s in TEMPLATE.split('(.*)')[1:-2]] + ['seed']
        
# tb_root = '/Users/roxy/Desktop/goku/fairness/metric_learning/adult_explore/tensorboard_adult/'
tb_root = './tensorboard_adult/'

all_experiments = glob.glob(tb_root + '*')

results = {}
tb_dict = defaultdict(list)
to_load = []

for exp in all_experiments:
    tb_path = glob.glob(exp + '/events*')[0]
    for event in tf.train.summary_iterator(tb_path):
        pass
    exp_name = exp.split('/')[-1]
    if filter_event(event) and event.step>1e4:
        results[exp_name] = event
        _ = parse_name(exp_name, tb_dict)
    if filter_experiment(exp_name) and event.step>1e4:
        to_load.append(exp_name)
        
# tb_query = '(' + '|'.join(results.keys()) + ')'
tb_query = ''
for t in TAGS:
    if t not in ['name', 'seed']:
        tb_query += t + '\^'
    tb_query += '(' + '|'.join(tb_dict[t]) + ')'
    if t != 'seed':
        tb_query += '_'
        
print(tb_query)
print('\n',50*'--','\n')
tb_run = 'tensorboard --logdir ' + ','.join(['^'.join(r.split(':'))+':'+tb_root+r for r in results.keys()])
# tb_run = 'tensorboard --logdir ' + ','.join(['^'.join(r.split(':'))+':'+tb_root+r for r in to_load])
print(tb_run)

