import matplotlib.pyplot as plt
import numpy as np
import sys

def create_plots(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

        iters = []
        error_1 = []
        error_2 = []
        vals = []

        for i in range(len(lines)):
            lines[i] = lines[i].split()

            if len(lines[i]) == 1:
                iters.append(lines[i][0])
                error_1.append([])
                error_2.append([])
                vals.append([])
            else:
                error_1[-1].append(float(lines[i][0]))
                error_2[-1].append(float(lines[i][1]))
                vals[-1].append(float(lines[i][2]))

        for_plot = [{'iter': iters[l], 'data': [error_1[l], error_2[l], vals[l]]}
                    for l in range(len(iters))]

        print(for_plot)

        for plot in for_plot:
            plt.figure(plot['iter'])
            plt.plot(plot['data'][2], plot['data'][0], plot['data'][2], plot['data'][1])
            plt.ylabel('errors')
            plt.xlabel('thresh')

        plt.show()


if __name__ == '__main__':
    create_plots(filename='real_results/{}.txt'.format(sys.argv[1]))
