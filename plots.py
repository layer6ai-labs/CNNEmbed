import matplotlib.pyplot as plt

def gating_no_gating_plot():

    gating = [87.63, 88.94, 89.41]
    no_gating = [87.09, 87.87, 88.29]
    epoch_nums = [10, 20, 30]

    plt.plot(epoch_nums, gating, marker='o', color='r', label='GLUs')
    plt.plot(epoch_nums, no_gating, marker='o', color='b', label='Relus')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def num_layers_plot():

    num_layers = [1, 3, 5, 7]
    accs = [88.18, 89.34, 89.5, 88.6]

    plt.plot(num_layers, accs, marker='o')
    plt.xlabel('Number of convolutional layers')
    plt.ylabel('Accuracy (%)')
    plt.show()

def words_forward_plot():

    words_forward = [1, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70]
    accs = [84.97, 88.14, 88.19, 88.55, 88.75, 88.6, 88.07, 87.72, 88.03, 87.59, 87.01]
    plt.plot(words_forward, accs, marker='o')
    plt.xlabel('Number of words forward to predict')
    plt.ylabel('Accuracy (%)')
    plt.show()

if __name__ == '__main__':

    words_forward_plot()