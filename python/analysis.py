import torch
import numpy
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import TSNE

"""
second test
"""
# path = 'three_accuracies_one_dataset_twenty_models.pt'
# t = torch.load(path)[0]

# # complexity histogram for 60%, 70% and 80%
# t[2][3][0], t[2][4][0] = t[2][4][0], t[2][3][0]
# for i in range(20):
#     t[5][i][0][3], t[5][i][0][4] = t[5][i][0][4], t[5][i][0][3]
# figure, axes = plt.subplots(1, 5, figsize=(16, 3), constrained_layout=True)
# for k in range(5):
#     X60 = [numpy.mean(t[5][i][0][k][5][0]) for i in range(20)]
#     X70 = [numpy.mean(t[5][i][0][k][5][1]) for i in range(20)]
#     X80 = [numpy.mean(t[5][i][0][k][5][2]) for i in range(20)]
#     axes[k].hist(X60, label='60%')
#     axes[k].hist(X70, label='70%')
#     axes[k].hist(X80, label='80%')
#     axes[k].set_title(t[2][k][0])
# axes[-1].legend()
# plt.savefig('20models_3versions-complexity.png')
# plt.show()

# # avg sensitivity vs infidelity
# t[2][3][0], t[2][4][0] = t[2][4][0], t[2][3][0]
# for i in range(20):
#     t[5][i][0][3], t[5][i][0][4] = t[5][i][0][4], t[5][i][0][3]
# figure, axes = plt.subplots(1, 5, figsize=(16, 3), constrained_layout=True)
# for k in range(5):
#     X = [[], []]
#     for i in range(20):
#         for j in range(3):
#             X[0].append(numpy.log(numpy.array([numpy.mean([t[5][i][0][k][0][j]]), numpy.mean([t[5][i][0][k][1][j]])])))
#             X[1].append(t[4][i][0][j][1])
#     X[0] = numpy.array(X[0])
#     X[1] = numpy.array(X[1])
#     axes[k].scatter(X[0][:, 0], X[0][:, 1], c=X[1], cmap='winter')
#     axes[k].set_xlabel('log-Sensitivity')
#     axes[k].set_ylabel('log-Infidelity')
#     axes[k].set_title(t[2][k][0])
# plt.savefig('20models_3versions-sensitivity-infidelity.png')
# plt.show()

# # geometric visualisation of models
# # create vectors containing all model parameters (1d) and 'label' accuracy
# X = [[], []]
# for i in range(len(t[0])):
#     model_parameter_vectors = dc(t[4][i][0])
#     for j in range(3):
#         aux = []
#         for keyword in model_parameter_vectors[j][0]:
#             aux.append(torch.flatten(model_parameter_vectors[j][0][keyword]))
#         X[0].append(numpy.array(torch.cat(aux)))
#         X[1].append(model_parameter_vectors[j][1])
# X, Y = X
# X = numpy.array(X)
# Y = numpy.array(Y)
# # PCA
# U, W, _ = numpy.linalg.svd(X, full_matrices=False)
# Y0 = U[:, :2] * W[:2]
# plt.scatter(*Y0.T, c=Y, cmap='winter')
# plt.colorbar()
# plt.title('PCA with colored accuracy')
# plt.savefig('20models_3versions_PCA_colored.png')
# plt.show()
# # run TSNE starting with PCA embedding as an initial solution
# Y1 = TSNE.TSNE(X, Y0, 10.0, 100.0)
# plt.scatter(*Y1.T, c=Y, cmap='winter')
# plt.colorbar()
# plt.title('t-SNE with colored accuracy')
# plt.savefig('20models_3versions_tSNE_colored.png')
# plt.show()

"""
first test
"""
# path = 'test2.pt'
# the_tabulatorium = torch.load(path)[0]
#
# x = numpy.array([n for m, n in the_tabulatorium[4][0][1]])
# # for i in range(6):
# #     print(i)
# #     y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][0]]))
# #     plt.plot(x, y, label='log-Sens-200')
# #     y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][1]]))
# #     plt.plot(x, y, label='log-Sens-100')
# #     y = numpy.log(numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][2]]))
# #     plt.plot(x, y, label='log-Inf')
# #     plt.legend()
# #     plt.title(the_tabulatorium[2][i][0] + the_tabulatorium[0][0][1] + the_tabulatorium[1][0][1])
# #     plt.show()
# #     y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][3]])
# #     plt.plot(x, y, label='Eff-Comp-1e-5')
# #     y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][4]])
# #     plt.plot(x, y, label='Eff-Comp-1e-4')
# #     y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][5]])
# #     plt.plot(x, y, label='Eff-Comp-1e-3')
# #     y = numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][0][i][6]])
# #     plt.plot(x, y, label='Eff-Comp-1e-2')
# #     plt.legend()
# #     plt.title(the_tabulatorium[2][i][0] + the_tabulatorium[0][0][1] + the_tabulatorium[1][0][1])
# #     plt.show()
#
# the_tabulatorium[5][0][1][3], the_tabulatorium[5][0][1][4] = the_tabulatorium[5][0][1][4], the_tabulatorium[5][0][1][3]
# the_tabulatorium[2][3][0], the_tabulatorium[2][4][0] = the_tabulatorium[2][4][0], the_tabulatorium[2][3][0]
#
# figure, axes = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True, sharey=True)
# for i in range(4,5):
#     # plt.subplot(1, 6, i + 1)
#     y = (numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][1][i][3]]))
#     axes.plot(x, y, label='eps = 1e-5')
#     # axes[i].plot(x, y)
#     y = (numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][1][i][4]]))
#     axes.plot(x, y, label='eps = 1e-4')
#     y = (numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][1][i][5]]))
#     axes.plot(x, y, label='eps = 1e-3')
#     y = (numpy.array([numpy.mean(score) for score in the_tabulatorium[5][0][1][i][6]]))
#     axes.plot(x, y, label='eps = 1e-2')
#     # axes[i].plot(x, y)
#     # plt.ylabel('test accuracy')
#     axes.set_title(the_tabulatorium[2][i][0])
#     # print(axes[i].title)
# axes.legend()
# # figure.title('Effective Complexity')
# plt.savefig('simple_relu_complexity_FMNIST_local.png')
# plt.show()
