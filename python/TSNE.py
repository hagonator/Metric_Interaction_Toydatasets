import utils_TSNE
import numpy as np


def student(Y):
    # Calculates the probabilities q_ij defined above
    #
    # input:  Y - An Nx2 array containing the embedding
    # return: Q - An NxN array containing q_ij

    N = len(Y)

    Z = np.dot(Y, Y.T)
    z = np.diag(Z)
    Z = (z - 2 * Z).T + z + np.ones(N)
    Q = 1 / Z - np.identity(N)
    Q = Q / np.sum(Q)

    return Q


def objective(P, Q):
    # Calculates the objective of t-SNE to minimize. The objective is the
    # KL divergence C = KL(P||Q)
    #
    # inputs: P - An NxN array containing p_ij
    #         Q - An NxN array containing q_ij
    # return: C - The value of the objective

    N = len(P)

    C = np.sum(P * (np.log(P + np.identity(N)) - np.log(Q + np.identity(N))))

    return C


def gradient(P, Y):
    # Computes the gradient as described above.
    #
    # inputs: P     - An NxN array containing p_ij
    #        Y     - An Nx2 array containing the embedding
    # return: gradY - the gradient of the objective with respect to Y

    N = len(Y)

    Z = np.dot(Y, Y.T)
    z = np.diag(Z)
    Z = (z - 2 * Z).T + z + np.ones(N)
    Q = 1 / Z - np.identity(N)
    R = (P - Q / np.sum(Q)) * Q
    S = np.diag(np.sum(R, axis=0)) - R

    gradY = 4 * np.dot(S, Y)

    return gradY

def TSNE(X, Y0, perplexity, learningrate, nbiterations=1000):
    N, d = X.shape
    print('get affinity matrix')
    # get the affinity matrix in the original space
    P = utils_TSNE.getaffinity(X, perplexity)
    # create initial embedding and update direction
    Y = Y0 * 1
    dY = Y * 0
    print('run t-SNE')
    for t in range(nbiterations):
        # compute the pairwise affinities in the embedding space
        Q = student(Y)
        # monitor objective
        if t % 50 == 0: print('%3d %.3f' % (t, objective(P, Q)))
        # update
        dY = (0.5 if t < 100 else 0.9) * dY + learningrate * gradient(P, Y)
        Y = Y - dY
    return Y
