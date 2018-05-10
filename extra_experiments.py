import subprocess
import itertools

# Simple script for running some extra experiments.

def generate_param_combinations(hyper_params):
    '''Generate and return all combinations of the hyper parameters for the grid search.'''

    all_params = sorted(hyper_params)
    all_combs = itertools.product(*(hyper_params[name] for name in all_params))
    all_combs = list(all_combs)

    combinations_list = []
    for comb in all_combs:
        d = dict(zip(all_params, comb))
        combinations_list.append(d)

    return combinations_list

def words_forward_exp():

    words_forward = [1, 5, 10, 15, 20, 25, 30]
    subprocess_call = ['python', './train.py', '--context-len', '10', '--batch-size', '100', '--num-filters', '900',
                       '--num-layers', '4', '--num-negative-words', '50', '--num-residual', '2',
                       '--num-classes', '2', '--dataset', 'imdb', '--model', 'CNN_pad', '--max-iter', '31',
                       '--data-dir', '/home/shunan/Code/Data/', '--accuracy-file', './cache/words_forward_accs.pkl']

    for l in words_forward:
        print('Running experiment with {} words forward to predict'.format(l))
        new_call = subprocess_call + ['--num-positive-words', str(l)]
        subprocess.call(new_call)

def num_layers_exp():

    num_layers = [9]
    subprocess_call = ['python', './train.py', '--context-len', '10', '--batch-size', '100', '--num-filters', '900',
                       '--num-positive-words', '10', '--num-negative-words', '50', '--num-residual', '2',
                       '--num-classes', '2', '--dataset', 'imdb', '--model', 'CNN_pad', '--max-iter', '31',
                       '--data-dir', '/home/shunan/Code/Data/', '--accuracy-file', './cache/layer_accs.pkl']

    for l in num_layers:
        print('Running experiment with {} conv layers'.format(l - 1))
        new_call = subprocess_call + ['--num-layers', str(l)]
        subprocess.call(new_call)

def amazon_grid_search():
    '''
    Performing parameter sweeps on the Amazon dataset.
    '''

    hyper_params = {
        '--context-len': ['10'],
        '--num-filters': ['900'],
        '--num-positive-words': ['10'],
        '--num-negative-words': ['70'],
        '--num-residual': ['2'],
        '--num-layers': ['8', '10'],
        '--filter-size': ['5'],
        '--l2-coeff': ['0']
    }

    all_params = generate_param_combinations(hyper_params)
    subprocess_call = ['python', './train.py', '--batch-size', '100', '--num-classes', '2', '--dataset', 'amazon',
                       '--model', 'CNN_pad', '--max-iter', '36', '--data-dir', '/home/shunan/Data/',
                       '--accuracy-file', './cache/amazon_accs.pkl']

    i = 0
    while i < len(all_params):
        param = all_params[i]
        new_call = subprocess_call[:]
        for elem in param:
            new_call.append(elem)
            new_call.append(param[elem])

        print('Running experiment with: {}'.format(param))
        subprocess.call(new_call)

        i += 1

def wikipedia_grid_search():
    '''
    Perform parameter sweeps on the Wikipedia dataset.
    '''

    hyper_params = {
        '--context-len': ['10', '20'],
        '--num-filters': ['900'],
        '--num-positive-words': ['10'],
        '--num-negative-words': ['50', '70'],
        '--num-residual': ['1', '2'],
        '--num-layers': ['7', '9'],
        '--filter-size': ['5', '7']
    }

    all_params = generate_param_combinations(hyper_params)
    subprocess_call = ['python', './train.py', '--batch-size', '100', '--num-classes', '100', '--dataset', 'wikipedia',
                       '--model', 'CNN_pad', '--max-iter', '36', '--data-dir', '/home/shunan/Data/', '--gap-max', '4',
                       '--l2-coeff', '0.3', '--accuracy-file', './cache/wikipedia_grid_search.pkl']

    i = 0
    while i < len(all_params):
        param = all_params[i]
        new_call = subprocess_call[:]
        for elem in param:
            new_call.append(elem)
            new_call.append(param[elem])

        print('Running experiment with: {}'.format(param))
        subprocess.call(new_call)

        i += 1

if __name__ == '__main__':

    amazon_grid_search()
