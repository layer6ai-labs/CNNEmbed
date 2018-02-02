import subprocess

# Simple script for running some extra experiments.

if __name__ == '__main__':

    words_forward = [1, 5, 7, 10, 15, 20, 30, 40]
    subprocess_call = ['python', './train.py', '--context-len', '10', '--batch-size', '100', '--num-filters', '300',
                       '--num-layers', '4', '--num-negative-words', '50', '--num-residual', '2',
                       '--num-classes', '2', '--dataset', 'imdb', '--model', 'CNN_pad', '--max-iter', '31',
                       '--data-dir', '/home/shunan/Code/Data/', '--accuracy-file', './cache/words_forward_accs.pkl']

    for l in words_forward:
        print('Running experiment with {} words forward to predict'.format(l - 1))
        new_call = subprocess_call + ['--num-positive-words', str(l)]
        subprocess.call(new_call)