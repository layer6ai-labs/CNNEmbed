import matplotlib.pyplot as plt

def gating_no_gating_plot():

    gating = [87.63, 88.94, 89.41]
    no_gating = [87.09, 87.87, 88.29]
    epoch_nums = [10, 20, 30]

    plt.plot(epoch_nums, gating, marker='o', markersize=10, color='r', label='GLU')
    plt.plot(epoch_nums, no_gating, marker='o', markersize=10, color='b', label='ReLu')
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)
    plt.tick_params(labelsize=25)
    plt.legend(prop={'size': 25})
    plt.show()

def num_layers_plot():

    num_layers = [1, 2, 3, 4, 5, 6, 7]
    accs = [88.18, 89.03, 89.34, 89.2, 89.5, 89.61, 88.6]
    # the third value is wrong.

    plt.plot(num_layers, accs, marker='o', markersize=10, markevery=1)
    plt.xlabel('Number of GLU layers', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)
    # plt.ylim(87.5, 90)
    plt.tick_params(labelsize=25)
    plt.show()

def words_forward_plot():

    words_forward = [1, 5, 10, 15, 20, 25, 30]
    accs = [86, 88.86, 89.49, 89.39, 89.27, 89, 89.02 ]
    plt.plot(words_forward, accs, marker='o', markersize=10, markevery=1)
    plt.xlabel('Forward prediction window h', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)
    plt.tick_params(labelsize=25)
    plt.show()

if __name__ == '__main__':
    
    words_forward_plot()