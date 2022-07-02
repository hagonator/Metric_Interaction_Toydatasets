import torch
import numpy
import matplotlib.pyplot as plt

path = 'test2.pt'
the_tabulatorium = torch.load(path)[0]

x = numpy.array([n for m, n in the_tabulatorium[4][0][0]])
for i in range(6):
    print(i)
    y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][0]]))
    plt.plot(x, y, label='log-Sens-200')
    y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][1]]))
    plt.plot(x, y, label='log-Sens-100')
    y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][2]]))
    plt.plot(x, y, label='log-Inf')
    plt.legend()
    plt.title(the_tabulatorium[2][i][0] + the_tabulatorium[0][0][1] + the_tabulatorium[1][0][1])
    plt.show()
    y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][3]])
    plt.plot(x, y, label='Eff-Comp-1e-5')
    y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][4]])
    plt.plot(x, y, label='Eff-Comp-1e-4')
    y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][5]])
    plt.plot(x, y, label='Eff-Comp-1e-3')
    y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][6]])
    plt.plot(x, y, label='Eff-Comp-1e-2')
    plt.legend()
    plt.title(the_tabulatorium[2][i][0] + the_tabulatorium[0][0][1] + the_tabulatorium[1][0][1])
    plt.show()

