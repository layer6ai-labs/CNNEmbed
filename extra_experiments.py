import subprocess

# Simple script for running some extra experiments.

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


if __name__ == '__main__':

    words_forward_exp()
    num_layers_exp()
