import numpy as np
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray
from numpy import genfromtxt
import matplotlib.animation as animation
import random
import numpy.matlib
import seaborn as sns
from sympy import symbols, Matrix, Transpose, sqrt, simplify, series
import progressbar
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import imageio
# import ffmpeg
# import pandas as pd
# import time
# import sys
# from io import StringIO

# programs for dynamic embedding
# Main functions are :
# pivot_torus_movie(N, eps, iter, k1, k2)
# Glauber_torus_movie(N, eps, iter, k1, k2)
# emb_cond_hom_torus(N, eps, steps, k1, k2) : computes cond.hom.den. of cycle/path within
# N by N torus + eps density random edges
# emb_cond_hom_csv(steps, k1, k2): computes cond.hom.den. of cycle/path within a network
# whose adjacency matrix is given by a csv file

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    # ind = r*cols + c
    r = np.floor(ind/array_shape[1])
    c = ind - r*array_shape[1]
    x = np.array([r, c])

    x = x.astype(int)
    return x


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def grid_adj(rows, cols):
    n = rows*cols
    mx = np.zeros((n, n), dtype=int)
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0:
                mx[i-1, i] = mx[i, i-1] = 1
            # Two outer diagonals
            if r > 0:
                mx[i-cols, i] = mx[i, i-cols] = 1
    return mx


def torus_adj(rows, cols):
    # Generates adjacency matrix of rows*cols torus

    a1: ndarray = np.zeros((rows * cols, rows * cols), dtype=int)
    for i in range(rows):
        a1[i * cols, i * cols + cols - 1] = 1
        a1[i * cols + cols - 1, i * cols] = 1

    for j in range(cols):
        a1[(rows - 1) * cols + j, j] = 1
        a1[j, (rows - 1) * cols + j] = 1

    a = grid_adj(rows, cols) + a1
    a[a == 2] = 1
    mx = a.copy()
    return mx


def torus_ER_adj(rows, cols, p):
    # Generates adjacency matrix of rows*cols + sprinkle of density p random edges
    A = torus_adj(rows, cols)
    U = np.random.rand(rows*cols, rows*cols)
    U = numpy.triu(U, k=1)
    U = U + np.transpose(U)
    U[U >= p] = 2
    U[U < p] = 1
    U[U == 2] = 0
    A = A + U
    A[A > 1] = 1
    return A


def torus_long_edges(rows, cols, p, alpha):
    # Generates adjacency matrix of rows*cols + additional edge b/w (a,b) and (c,d)
    # with probability p(|a-c|+|b-d|)^{-\alpha}

    A = torus_adj(rows, cols)
    U = np.random.rand(rows*cols, rows*cols)
    U = numpy.triu(U, k=1)
    U = U + np.transpose(U)

    for i in np.arange(rows*cols):
        for j in np.arange(rows*cols):
                [a, b] = ind2sub([rows, cols], i)
                [c, d] = ind2sub([rows, cols], j)

                aa = min(np.absolute(a-c) % rows, np.absolute(c-a) % rows)
                bb = min(np.absolute(b-d) % cols, np.absolute(d-b) % cols)

                if aa+bb > 1:
                    q = p * ((aa + bb) ** (-alpha))
                    if U[i, j] >= q:
                        U[i, j] = 2
                    if U[i, j] < q:
                        U[i, j] = 1

    U[U == 2] = 0
    A = A + U
    A[A > 1] = 1
    A.astype(int)

    return A


def barbell_adj(rows, cols, p1, p2):
    #  Generates adjacency matrix of a barbell graph obtained by a (torus + p1 density edges) and
    #  (torus + p2 density edges) by a single random edge
    A1 = torus_ER_adj(rows, cols, p1)
    A2 = torus_ER_adj(rows, cols, p2)
    N = rows*cols  # number of nodes in graph
    A = np.zeros((2*N,2*N), dtype=int)
    A[0:N, 0:N] = A1
    A[N:2*N, N:2*N] = A2
    y1 = np.random.choice(np.arange(0, N))
    y2 = np.random.choice(np.arange(N, 2*N))
    A[y1, y2] = 1
    A[y2, y1] = 1
    np.fill_diagonal(A, 0) # kill diagonal entries to remove loops
    return A


def barbell_sbm_adj(rows, cols, blowup, std=0.1, p1=0, p2=0.2):
    #  Generates adjacency matrix of a SBM barbell graph obtained by a (torus + p1 density edges) and
    #  (torus + p2 density edges) by a single random edge
    #  k = block size
    A1 = torus_ER_adj(rows, cols, p1)
    A2 = torus_ER_adj(rows, cols, p2)
    N = blowup*rows*cols  # number of nodes in graph
    A = np.zeros((2*N, 2*N), dtype=float)
    A[0:N, 0:N] = block_sample(A1, std, blowup)
    A[N:2*N, N:2*N] = block_sample(A2, std, blowup)

    # add a random edge connecting the two clusters
    y1 = np.random.choice(np.arange(0, N))
    y2 = np.random.choice(np.arange(N, 2*N))
    A[y1, y2] = np.max(A)
    A[y2, y1] = np.max(A)
    np.fill_diagonal(A, 0)  # kill diagonal entries to remove loops
    return A*(N**2)/sum(sum(A))


def path_adj(k1, k2):
    # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
    if k1 == 0:
        A = np.eye(k2 + 1, k=1, dtype=int)
    else:
        A = np.eye(k1+k2+1, k=1, dtype = int)
        A[k1,k1+1] = 0
        A[0,k1+1] = 1
    return A


def cycle_adj(k1, k2):
    # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
    if k1 == 0:
        A = np.eye(k2 + 1, k=1, dtype=int)
        A[0, k2] = 1
    else:
        A = np.eye(k1+k2+1, k=1, dtype=int)
        A[k1, k1+1] = 0
        A[k1, k1+k2] = 1
        A[0, k1+1] = 1
    return A


def WAN_network_gen(num):
    #  add up 211 by 211 WAN matrices of 1000 consecutive words to get a single network for each book
    #  generate file names
    s = [r'''C:\Users\colou\PycharmProjects\untitled\colourgraph\WAN\austen_sense\wan_austen_sense_''' + str(i) + r".txt" for i in range(1, num+1)]

    A = np.zeros((211, 211), dtype=int)
    for i in np.arange(num):
        A = A + genfromtxt(s[i], usecols=range(211))
    A = A / np.amax(A)

    np.savetxt('austen_sense.txt', A)

    return A


def block_sample(A, std, k):
    # Given a template matrix A, replace each entry by a k by k block of that entry + noise
    [N, M] = np.shape(A)
    B = np.zeros([N*k, M*k])
    for i in np.arange(N):
        for j in np.arange(M):
            if A[i,j] == 0:
                B[i * k:(i + 1) * k, j * k:(j + 1) * k] = 0
            else:
                B[i*k:(i+1)*k, j*k:(j+1)*k] = np.random.gamma(A[i, j]**2/std**2, std**2/A[i, j], [k, k])
                # Gamma distribution with mean A[i,j] and variance std^2
    return B


def sbm_sample(A, std, k, name):
    # sample stochastic block network from template A + Gaussian noise
    '''
    A = np.array([[5, 1, 1, 1, 1, 1],
                  [1, 5, 1, 1, 1, 1],
                  [1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 5, 1, 1],
                  [1, 1, 1, 1, 5, 1],
                  [1, 1, 1, 1, 1, 5]])

    A = np.array([[1, 1, 1, 5, 5, 1],
                  [1, 1, 1, 1, 1, 5],
                  [5, 1, 1, 5, 1, 5],
                  [5, 1, 5, 1, 1, 2],
                  [1, 5, 1, 1, 1, 1],
                  [1, 1, 5, 10, 1, 1]])
    '''

    B = block_sample(A, std, k)
    B = B / np.max(B)
    #np.savetxt('SBM_network' + str(name) + '.txt', B, fmt='%f\t')
    np.savetxt('SBM_network' + str(name) + '.txt', B)

    return B


def display_matrix():
    # display a matrix given by a csv or txt file
    A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\dickens_christmas.txt''', usecols=range(211))
    plt.spy(A, precision=0.001)
    plt.title('Charles Dickens - A Christmas Carol')
    a = np.where(A > 0.1)
    a = a[0]
    b = len(a) / 211 ** 2
    c = sum(sum(A))/211**2


def RW_update(A, x):
    # A = N by N matrix giving edge weights on networks
    # x = RW is currently at site x
    # RW kernel is modified by Metropolis-Hastings to converge to uniform dist. on [N]

    [N, N] = np.shape(A)
    dist1 = np.maximum(A[x, :], np.transpose(A[:, x]))
    deg_x = sum(dist1)
    dist2 = dist1 / sum(dist1)  # honest symmetric RW kernel
    dist = dist2

    '''
    for i in np.arange(N):
        deg_i = sum(np.maximum(A[i, :], np.transpose(A[:, i])))
        deg_out_i = sum((A[i, :]))
        if i != x and deg_i > 0 and deg_out_i > 0:
            dist[i] = dist2[i]*min(deg_x/deg_i, 1)  #M-H modification
    dist[x] = 0
    dist[x] = 1 - sum(dist)
    '''

    y = np.random.choice(np.arange(0, N), p=dist)
    return y


def Path_sample(A, x, k):
    # A = N by N matrix giving edge weights on networks
    # number of nodes in path
    # samples a path of length k from a given pivot x as the first node

    [N, N] = np.shape(A)
    emb = np.array([x]) # initialize path embedding

    for i in np.arange(0,k+1):
        dist = A[emb[i], :] / sum(A[emb[i], :])
        y = np.random.choice(np.arange(0, N), p=dist)
        emb = np.hstack((emb, y))

    return emb


def find_parent(B, i):
    # B = adjacency matrix of the tree motif rooted at first node
    # Nodes in tree B is ordered according to the depth-first-ordering
    # Find the index of the unique parent of i in B
    j = indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
    return min(j)


def tree_sample(A, B, x):
    # A = N by N matrix giving edge weights on networks
    # B = adjacency matrix of the tree motif rooted at first node
    # Nodes in tree B is ordered according to the depth-first-ordering
    # samples a tree B from a given pivot x as the first node

    [N, N] = np.shape(A)
    [k, k] = np.shape(B)
    emb = np.array([x])  # initialize path embedding

    if sum(sum(B)) == 0:
        y = np.random.randint(N, size=(1, k-1))
        y = y[0]
        emb = np.hstack((emb, y))
    else:
        for i in np.arange(1, k):
            j = find_parent(B, i)
            dist = A[emb[j], :] / sum(A[emb[j], :])
            y = np.random.choice(np.arange(0, N), p=dist)
            emb = np.hstack((emb, y))

    return emb


def Path_sample_gen_position(A, x, k1, k2):
    # A = N by N matrix giving edge weights on networks
    # number of nodes in path
    # samples k1 nodes to the left and k2 nodes to the right of pivot x

    [N, N] = np.shape(A)
    emb = np.array([x]) # initialize path embedding

    for i in np.arange(0, k2):
        dist = A[emb[i], :] / sum(A[emb[i], :])
        y1 = np.random.choice(np.arange(0, N), p=dist)
        emb = np.hstack((emb, [y1]))

    a = np.array([x])
    b = np.matlib.repmat(a, 1, k1+1)
    b = b[0, :]
    emb = np.hstack((b, emb[1:k2+1]))

    for i in np.arange(0, k1):
        dist = A[emb[i], :] / sum(A[emb[i], :])
        y2 = np.random.choice(np.arange(0, N), p=dist)
        emb[i+1] = y2

    return emb


def Pivot_update(A, emb, k1, k2):
    # A = N by N matrix giving edge weights on networks
    # emb = current embedding of a path in the network
    # k1 = length of left side chain from pivot
    # updates the current embedding using pivot rule

    [N, N] = np.shape(A)
    x0 = emb[0]  # current location of pivot
    x0 = RW_update(A, x0)  # new location of the pivot
    emb_new = Path_sample_gen_position(A, x0, k1, k2)  # new path embedding
    return emb_new


def pivot_gen_update(A, B, emb):
    # A = N by N matrix giving edge weights on networks
    # emb = current embedding of a path in the network
    # B = adjacency matrix of the tree motif
    # updates the current embedding using pivot rule

    [N, N] = np.shape(A)
    x0 = emb[0]  # current location of pivot
    x0 = RW_update(A, x0)  # new location of the pivot
    emb_new = tree_sample(A, B, x0)  # new tree embedding
    return emb_new


def Glauber_update(A, emb, k1, k2):
    # A = N by N matrix giving edge weights on networks
    # emb = current embedding of a path in the network
    # updates the current embedding using Glauber rule
    [N, N] = np.shape(A)
    if k1 + k2 == 0:
        emb[0] = RW_update(A, emb[0])
    else:
        j = random.randint(0, k1+k2)  # choose a random node to update
        dist = np.arange(N)
        if j == 0:
            dist = A[:, emb[1]] * A[:, emb[k1+1]]
        if 0 < j < k1:
            dist = A[emb[j-1], :] * np.transpose(A[:, emb[j+1]])
        if j == k1 and j > 0:
            dist = A[emb[j - 1], :]
        if j == k1 + 1:
            dist = A[emb[0], :]
        if k1 < j < k1 + k2:
            dist = A[emb[j - 1], :] * np.transpose(A[:, emb[j + 1]])
        if j == k1 + k2:
            dist = A[emb[j - 1], :]

        dist = dist / sum(dist)
        y = np.random.choice(np.arange(0, N), p=dist)
        emb[j] = y
    return emb


def glauber_gen_update(A, B, emb):
    # A = N by N matrix giving edge weights on networks
    # emb = current embedding of the tree motif with adj mx B
    # updates the current embedding using Glauber rule
    [N, N] = np.shape(A)
    [k, k] = np.shape(B)

    if k == 1:
        emb[0] = RW_update(A, emb[0])
    else:
        j = random.randint(0, k-1)  # choose a random node to update
        nbh_in = indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
        nbh_out = indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

        # build distribution for resampling emb[j]
        dist = np.ones(N, dtype=int)
        for r in nbh_in:
            dist = dist * A[emb[r], :]
        for r in nbh_out:
            dist = dist * np.transpose(A[:, emb[r]])
        dist = dist / sum(dist)

        # resample emb[j]
        y = np.random.choice(np.arange(0, N), p=dist)
        emb[j] = y
    return emb


def pivot_torus_movie(N, eps, iter, k1, k2):
    # returns an animation of Pivot chain of path with k1 and k2 nodes
    # to the left and right of pivot
    # into N by N torus for `iter' iterations
    # eps = density of random edges

    fig = plt.figure()
    A = torus_ER_adj(N, N, eps)  # adj matrix of N by N grid
    x = np.arange(0, N)
    y = np.arange(0, N).reshape(-1, 1)
    x0 = random.randint(0, N**2-1)  # random initial location of RW
    emb = Path_sample_gen_position(A, x0, k1, k2) # initial sampling of path embedding

    ims = []

    for k in np.arange(iter):
        emb = Pivot_update(A, emb, k1, k2) # new path embedding

        # turn emb into a configuration on torus
        config = np.zeros([N, N])
        for i in np.arange(0, k1):
            [a, b] = ind2sub([N, N], emb[i])
            # coordinate of ith node of path in torus
            config[a, b] = config[a, b] + 1

        for i in np.arange(k1+1, k1+k2+1):
            [a, b] = ind2sub([N, N], emb[i])
            # coordinate of ith node of path in torus
            config[a, b] = config[a, b] + 1

        # append new image
        ims.append((plt.pcolor(x, y, config, norm=plt.Normalize(0, 4)),))

    im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=0, blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im_pivot.mp4', metadata={'artist':'Han'})
    plt.show(im_ani)
    return im_ani


def Glauber_torus_movie(N, eps, steps, k1, k2):
    # returns an animation of Glauber chain of path on node set [k]
    # into N by N torus for `iter' iterations
    # eps = density of random edges

    fig = plt.figure()
    A = torus_ER_adj(N, N, eps)  # adj matrix of N by N grid
    x = np.arange(0, N)
    y = np.arange(0, N).reshape(-1, 1)
    x0 = random.randint(0, N**2-1)  # random initial location of RW
    emb = Path_sample_gen_position(A, x0, k1, k2) # initial sampling of path embedding

    ims = []

    for iter in np.arange(steps):
        emb = Glauber_update(A, emb, k1, k2)  # new path embedding

        # turn emb into a configuration on torus
        config = np.zeros([N, N])
        for i in np.arange(0, k1):
            [a, b] = ind2sub([N, N], emb[i])
            # coordinate of ith node of path in torus
            config[a, b] = config[a, b] + 1

        for i in np.arange(k1 + 1, k1 + k2 + 1):
            [a, b] = ind2sub([N, N], emb[i])
            # coordinate of ith node of path in torus
            config[a, b] = config[a, b] + 1

        # append new image
        ims.append((plt.pcolor(x, y, config, norm=plt.Normalize(0, 4)),))
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=0, blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im_Glauber.mp4', metadata={'artist':'Han'})

    plt.show(im_ani)
    return im_ani


def chd_gen(A, B, C, steps):
    # computes conditional homomorphism density t(H,A|F) using Pivot chain time average,
    # which is prob. of seeing all extra edges given that we have all edges
    # in tree motif with adj matrix B
    # steps = number of iteration
    # underlying graph = specified by A
    # F = path on node set [3]
    # H = any graph which contains F as a spanning subgraph
    # Here H = cycle, so there is only one extra edge (k1, k1+k2) to see
    '''
    #  adj matrix of the tree motif
    B = np.array([[0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    #  adj matrix of extra edges in H-F
    C = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    '''

    # A = torus_ER_adj(N, N, eps)  # adj matrix of N by N grid
    # specify network
    # A = barbell_adj(N, N, p1, p2)  # adj matrix of N by N grid
    # A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\network1.csv''', delimiter=',')
    # A = A/np.amax(A)  # normalize

    [N, N] = np.shape(A)
    x0 = random.randint(0, N)  # random initial location of RW
    emb1 = tree_sample(A, B, x0)  # initial sampling of path embedding
    emb2 = tree_sample(A, B, x0)  # initial sampling of path embedding

    hom1 = np.array([])
    hom2 = np.array([])
    bar = progressbar.ProgressBar()
    for i in bar(range(steps)):
        emb1 = pivot_gen_update(A, B, emb1)
        emb2 = glauber_gen_update(A, B, emb2)
        J = np.where(C - B == 1)  # count only edges in C-B
        a = J[0]
        b = J[1]

        wt1 = np.array([1])
        for w in np.arange(len(a)):
            wt1 = wt1 * A[emb1[a[w]], emb1[b[w]]]

        wt2 = np.array([1])
        for w in np.arange(len(a)):
            wt2 = wt2 * A[emb2[a[w]], emb2[b[w]]]

        hom1 = np.hstack((hom1, wt1))
        hom2 = np.hstack((hom2, wt2))

    # take time average
    y1 = np.zeros(steps)
    y2 = np.zeros(steps)
    for i in np.arange(1, steps):
        y1[i] = ((y1[i - 1] * i) + hom1[i]) / (i + 1)
        y2[i] = ((y2[i - 1] * i) + hom2[i]) / (i + 1)

    # draw figure
    sns.set_style('darkgrid', {'legend.frameon': True})
    fig = plt.figure()
    x = np.arange(steps)
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
    ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')
    plt.xlabel('iteration')
    plt.ylabel('con.hom.density')
    # plt.title('%i by %i Torus + %1.2f density random edges' % (N, N, eps))
    #  plt.title('%i by %i Torus, k1=%i and k2=%i ' % (N, N, k1, k2))

    plt.legend()
    plt.axis([0, steps, 0, 1])
    # ax1.set_aspect(steps)

    plt.show()
    return y1, y2


def chd_path(A, steps, k1, k2):
    B = path_adj(k1, k2)
    C = cycle_adj(k1, k2)
    y1, y2 = chd_gen(A, B, C, steps)
    t = (sum(y1) + sum(y2)) / (2*steps)  # t(H,G | F)
    return t


def hd_path(A, steps, k1, k2):
    # computes unconditioned homomorphism density of F in G
    B = np.zeros([k1+k2+1, k1+k2+1])
    C = path_adj(k1, k2)
    y1, y2 = chd_gen(A, B, C, steps)
    t = (sum(y1) + sum(y2)) / (2*steps)  # t(H,G | F)
    return t


def cond_hom_filt(A, B, C, steps, grid):
    # computes conditional homomorphism density profile f(H,A|F) using Pivot chain time average
    # Matrices for motifs H and F are given by C and B
    # F = directed tree rooted at node 0
    # steps = number of iteration
    # underlying network = given by A
    # gird = number of filtration levels

    A = A/np.amax(A)  # normalize
    [N, N] = np.shape(A)

    x0 = random.randint(0, N)  # random initial location of RW
    emb1 = tree_sample(A, B, x0)  # initial sampling of tree motif F
    emb2 = tree_sample(A, B, x0)  # initial sampling of tree motif F

    hom1 = np.zeros([grid, 1])
    hom2 = np.zeros([grid, 1])

    bar = progressbar.ProgressBar()
    for i in bar(range(steps)):
        emb1 = pivot_gen_update(A, B, emb1)
        emb2 = glauber_gen_update(A, B, emb2)
        J = np.where(C == 1)
        a = J[0]
        b = J[1]

        wt1 = np.ones([grid, 1])
        for w in np.arange(len(a)):
            for i in np.arange(grid):
                wt1[i] = wt1[i] * np.where(A[emb1[a[w]], emb1[b[w]]] > i/grid, 1, 0)

        wt2 = np.ones([grid, 1])
        for w in np.arange(len(a)):
            for i in np.arange(grid):
                wt2[i] = wt2[i] * np.where(A[emb2[a[w]], emb2[b[w]]] > i / grid, 1, 0)

        hom1 = np.hstack((hom1, wt1))
        hom2 = np.hstack((hom2, wt2))

    #  construct density profiles of filtration
    y1 = np.zeros(grid)
    y2 = np.zeros(grid)
    for i in np.arange(grid):
        y1[i] = sum(hom1[i, :]) / steps
        y2[i] = sum(hom2[i, :]) / steps

    return y1, y2


def chdf(steps, grid):
    # Specify matrices A, B, C here and draw conditional homomorphism density profile f(C,A|B).

    # Specify A
    A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\SBM_network3.txt''', usecols=range(60))
    # A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\twain_sawyer.txt''', usecols=range(211))
    for i in np.arange(60):
        A[i, :] = A[i, :] / sum(A[i, :])  # Markov kernel

    '''
    #  adj matrix of the tree motif
    B = np.array([[0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    #  adj matrix of extra edges in H-F
    C = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    '''
    k1 = 1
    k2 = 1
    B = path_adj(k1, k2)
    C = cycle_adj(k1, k2) - B
    y1, y2 = cond_hom_filt(A, B, C, steps, grid)

    # draw figure
    sns.set_style('darkgrid', {'legend.frameon': True})
    fig = plt.figure()
    x = np.arange(grid) / grid
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
    ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')

    plt.legend()
    plt.axis([0, 1, 0, 1])
    ax1.set_aspect(1)
    plt.xlabel('filtration level')
    plt.ylabel('density profile')
    # plt.title('Mark Twain - The Adventures of Tom Sawyer\n k1=%i and k2=%i' % (0, 0))
    # plt.title('Network 2 \n k1=%i and k2=%i' % (k1, k2))
    plt.show()

    return y1, y2


def chdf_path(A, steps, grid, k1, k2):
    # Specify matrices A, B, C here and draw conditional homomorphism density profile f(C,A|B).

    # Specify A
    # A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\SBM_network2.txt''', usecols=range(60))
    # A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\twain_tramp.txt''', usecols=range(211))
    for i in np.arange(211):
        if sum(A[i, :]) > 0:
            A[i,:] = A[i,:] / sum(A[i, :])  # Markov kernel

    # A = torus_ER_adj(10, 10, 0)

    B = path_adj(k1, k2)
    C = cycle_adj(k1, k2) - B
    y1, y2 = cond_hom_filt(A, B, C, steps, grid)

    # draw figure
    sns.set_style('darkgrid', {'legend.frameon': True})
    fig = plt.figure()
    x = np.arange(grid) / grid
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
    ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')

    plt.legend()
    plt.axis([0, 1, 0, 1])
    ax1.set_aspect(1)
    plt.xlabel('filtration level')
    plt.ylabel('density profile')
    plt.title('Mark Twain - A Tramp Abroad\n k1=%i and k2=%i' % (k1, k2))
    # plt.title('Network 2 \n k1=%i and k2=%i' % (k1, k2))
    plt.show()

    return y1, y2


def chdf_path_fast(steps, grid, k1, k2):
    # computes conditional homomorphism density profile f(H,A|F) using Pivot chain time average
    # Matrices for motifs H and F are given by C and B
    # F = directed tree rooted at node 0
    # steps = number of iteration
    # underlying network = given by A
    # gird = number of filtration levels

    # Specify A
    # A = genfromtxt(r'''embedding\SBM_network2.txt''', usecols=range(60))
    A = genfromtxt(r'''embedding\WAN\austen_pride.txt''', usecols=range(211))
    '''
    for i in np.arange(211):
        if sum(A[i,:]) > 0:
            A[i, :] = A[i, :] / sum(A[i, :])  # Markov kernel
    A = A/np.max(A)
    '''

    [N, N] = np.shape(A)

    x0 = random.randint(0, N)  # random initial location of RW
    emb1 = Path_sample_gen_position(A, x0, k1, k2)  # initial sampling of tree motif F
    emb2 = Path_sample_gen_position(A, x0, k1, k2)  # initial sampling of tree motif F

    hom1 = np.zeros([grid, 1])
    hom2 = np.zeros([grid, 1])

    bar = progressbar.ProgressBar()
    for i in bar(range(steps)):
        emb1 = Pivot_update(A, emb1, k1, k2)
        emb2 = Glauber_update(A, emb2, k1, k2)
        a = k1
        b = k1 + k2

        wt1 = np.ones([grid, 1])
        for i in np.arange(grid):
            wt1[i] = wt1[i] * np.where(A[emb1[a], emb1[b]] > i/grid, 1, 0)

        wt2 = np.ones([grid, 1])
        for i in np.arange(grid):
            wt2[i] = wt2[i] * np.where(A[emb2[a], emb2[b]] > i/grid, 1, 0)

        hom1 = np.hstack((hom1, wt1))
        hom2 = np.hstack((hom2, wt2))

    #  construct density profiles of filtration
    y1 = np.zeros(grid)
    y2 = np.zeros(grid)
    for i in np.arange(grid):
        y1[i] = sum(hom1[i, :]) / steps
        y2[i] = sum(hom2[i, :]) / steps

    # draw figure
    sns.set_style('darkgrid', {'legend.frameon': True})
    fig = plt.figure()
    x = np.arange(grid) / grid
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
    ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')

    plt.legend()
    plt.axis([0, 1, 0, 1])
    ax1.set_aspect(1)
    plt.xlabel('filtration level')
    plt.ylabel('density profile')
    plt.title('Jane Austen - Pride and Prejudice\n k1=%i and k2=%i' % (k1, k2))
    # plt.title('Network 2 \n k1=%i and k2=%i' % (k1, k2))
    plt.show()

    return y1, y2


'''
def path_transform(A, dist, k):
    # computes transform of the network (A,dist) by path of k nodes
    # dist = probability dist. on nodes. type = np.ndarray
    # A = matrix of edge weights

    N = len(dist)  # number of nodes in network
    D = np.diag(np.sqrt(dist))
    B = D@A@D

    C = np.linalg.matrix_power(B, k-1)
    C = D@C@D
    C = C/sum(sum(C)) # normalize
    return C
'''


def motif_transform_gen(A, B, C, steps):
    # computes motif transform A^H using Glauber/Pivot chain time average F-->G
    # Matrices for motif F is given by B
    # C gives the extra edge in H - F
    # F = directed tree rooted at node 0
    # steps = number of iteration
    # underlying network = given by A

    A = A/np.amax(A)  # normalize
    [N, N] = np.shape(A)
    [K, K] = np.shape(B)

    x0 = random.randint(0, N)  # random initial location of RW
    emb1 = tree_sample(A, B, x0)  # initial sampling of tree motif F
    # emb2 = tree_sample(A, B, x0)  # initial sampling of tree motif F

    hom1 = np.zeros([N, N])
    # hom2 = np.zeros([N, N])

    bar = progressbar.ProgressBar()
    for i in bar(range(steps)):
        emb1 = pivot_gen_update(A, B, emb1)
        # emb2 = glauber_gen_update(A, B, emb2)
        J = np.where(C == 1)
        a = J[0]
        b = J[1]

        wt1 = np.array([1])
        for w in np.arange(len(a)):
            wt1 = wt1 * A[emb1[a[w]], emb1[b[w]]]

        # wt2 = np.array([1])
        # for w in np.arange(len(a)):
        #    wt2 = wt2 * A[emb2[a[w]], emb2[b[w]]]

        hom1[emb1[0], emb1[K - 1]] = hom1[emb1[0], emb1[K - 1]] + wt1
        # hom2[emb2[0], emb2[K - 1]] = hom2[emb2[0], emb2[K - 1]] + wt2

    return hom1*(N**2)/sum(sum(hom1))


def motif_transform_movie(A, steps):
    # computes motif transform A^H using Glauber/Pivot chain time average F-->G
    # Matrices for motif F is given by B
    # C gives the extra edge in H - F
    # F = directed tree rooted at node 0
    # steps = number of iteration
    # underlying network = given by A
    k1 = 0
    k2 = 2
    B = path_adj(k1, k2)
    C = cycle_adj(k1, k2) - B

    A = A/np.amax(A)  # normalize
    [N, N] = np.shape(A)
    [K, K] = np.shape(B)

    x0 = random.randint(0, N)  # random initial location of RW
    emb1 = tree_sample(A, B, x0)  # initial sampling of tree motif F
    # emb2 = tree_sample(A, B, x0)  # initial sampling of tree motif F

    hom1 = np.zeros([N, N])
    # hom2 = np.zeros([N, N])

    fig = plt.figure()
    ims = []
    x = np.arange(0, N)
    y = np.arange(0, N).reshape(-1, 1)

    #bar = progressbar.ProgressBar()
    #for i in bar(range(steps)):
    for i in np.arange(steps):
        emb1 = pivot_gen_update(A, B, emb1)
        # emb2 = glauber_gen_update(A, B, emb2)
        J = np.where(C == 1)
        a = J[0]
        b = J[1]

        wt1 = np.array([1])
        for w in np.arange(len(a)):
            wt1 = wt1 * A[emb1[a[w]], emb1[b[w]]]

        hom1[emb1[0], emb1[K - 1]] = hom1[emb1[0], emb1[K - 1]] + wt1
        # hom2[emb2[0], emb2[K - 1]] = hom2[emb2[0], emb2[K - 1]] + wt2

        # append new image
        ims.append((plt.pcolor(x, y, hom1, norm=plt.Normalize(0, 4)),))

    im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=0, blit=True)

    plt.show(im_ani)
    return im_ani


def motif_transform(A, steps):
    # Specify matrices A, B, C here and compute motif transform A^{C} by Glauber/pivot chain avg B-->G

    # Specify A
    # A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\SBM_network1.txt''', usecols=range(60))
    # A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\twain_sawyer.txt''', usecols=range(211))
    # A = torus_ER_adj(10, 10, 0.2)
    '''
        for i in np.arange(211):
        A[i, :] = A[i, :] / sum(A[i, :])  # Markov kernel

    #  adj matrix of the tree motif
    B = np.array([[0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    #  adj matrix of extra edges in H-F
    C = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    '''
    k1 = 0
    k2 = 2
    B = path_adj(k1, k2)
    C = cycle_adj(k1, k2) - B
    y1 = motif_transform_gen(A, B, C, steps)

    # draw figure
    # sns.set_style('darkgrid', {'legend.frameon': True})


    # plt.title('Mark Twain - The Adventures of Tom Sawyer\n k1=%i and k2=%i' % (0, 0))
    # plt.title('Network 2 \n k1=%i and k2=%i' % (k1, k2))
    # plt.show()

    return y1


def path_transform(k):
    # computes path transform of A via path of k nodes
    # k = 0 means k = infty
    x, y = symbols('x y')
    z = (1-x)/2
    w = y*sqrt(x*(1-x)/2)
    a = Matrix([[sqrt(z), sqrt(x), sqrt(z)]])

    D = Matrix([[sqrt(z), 0, 0],
            [0, sqrt(x), 0],
            [0, 0, sqrt(z)]])

    A = Matrix([[z, w, 0],
            [w, x, w],
            [0, w, z]])

    Eval = list(A.eigenvals().keys())
    Evec = A.eigenvects()

    V = Matrix([[-1, 1, 1], [0, 0, 0], [1, 1, 1]])
    for i in range(3):
        v = Evec[i]
        v = v[2]
        v = v[0]
        v = simplify(v)
        V[:, i] = v

    if k == 0:
        v = V[:, 2]
        r = (a*V[:, 2])[0]
        P = (r ** (-2)) * D * v * Transpose(v) * D
    else:
        U = Matrix([[Eval[0] ** (k - 1), 0, 0],
                    [0, Eval[1] ** (k - 1), 0],
                    [0, 0, Eval[2] ** (k - 1)]])
        w = a * V
        t = (w * U * Transpose(w))[0]
        P = (D * V * U * Transpose(V) * D) / t

    P0 = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    bar = progressbar.ProgressBar()
    for i in bar(range(3)):
        for j in range(3):
            q = P[i, j]
            # q = series(q, y, 0, 2)
            P0[i, j] = series(q, x, 0, 2)

    return P0


def dist_mx(A):
    # compute distance matrix from a given pairwise weight matrix using Floyd-Warshall algorithm
    # input A is a n by n matrix of entries between 0 and 1.
    [n, n] = np.shape(A)
    # inf = n/np.min(A[A > 0])  # make sure inf is large enough
    B = A.copy()
    B = B.astype(float)
    B[B == 0] = -n + np.max(B)
    B = np.max(B) - B
    np.fill_diagonal(B, 0)  # this is the initial distance matrix

    for k in np.arange(n):
        for i in np.arange(n):
            for j in np.arange(n):
                if B[i, j] > B[i, k] + B[k, j]:
                    B[i, j] = B[i, k] + B[k, j]

    # symmetrization
    C = np.minimum(B, np.transpose(B))

    return C


def plot_dendro(A):
    # plot dendrogram of an input network A
    # A = torus_ER_adj(10,10, 0.2)
    # A = barbell_adj(10, 10, 0, 0)
    # A = barbell_sbm_adj(5, 5, 5, std, p1, p2)
    # A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\austen_pride.txt''', usecols=range(211))
    # A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\SBM_network2.txt''', usecols=range(60))
    B = squareform(dist_mx(A))  # condensed distance matrix from A
    Z = linkage(B, 'single')
    # Make the dendrogram
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance (single)')
    dendrogram(Z, leaf_rotation=90)
    return Z

'''
def plot_dendro_graphon_ex(s=0.1, eps=0.2):
    s=0.1
    eps=0.2
    a = (1/2) + (s**2 - 1/2)*eps
    b = s*(1+eps)/2
    c = (s**2)*eps
    A1 = np.array([[1, s, 0],
                  [s, 0, s],
                  [0, s, 1]])
    A2 = np.array([[a, b, c],
                  [b, 0, b],
                  [c, b, a]])
    A3 = np.array([[a, b*s, 0],
                  [b*s, 0, b*s],
                  [0, b*s, 1]])
    plot_dendro(A)
'''

def image_transform(steps):
    # Read an image as a triple of matrices, normalize each, and do triangle transform for 'step' iterations

    # Read image
    img = imageio.imread(r'''C:\Users\colou\PycharmProjects\colorgraph1\image\forest.jpg''')
    [n, m, l] = np.shape(img)
    r = np.min([n, m])
    img_new = np.zeros([r, r, 3])

    # Build a network for each color RGB
    for i in np.arange(3):
        '''
        # Make image matrices square by appending zeros
        if n > m:
            B = np.zeros([n, n])
            B[:, 0:m] = img[:, :, i]
        else:
            B = np.zeros([m, m])
            B[0:n, :] = img[:, :, i]
        '''
        r = np.min([n, m])
        B = img[0:r, 0:r, i]
        # Perform triangle transform for each Ai
        C = motif_transform(B, steps)
        img_new[:, :, i] = C


    #  forest1 = np.stack((C0, C1, C2), axis=-1)
    imgplot = plt.imshow(img_new)
    plt.show()

    return img_new


# run specific program here

#  emb_cond_hom_csv(3000, 0, 0)

#  emb_cond_hom_torus(10, 0, 1, 1000, 0, 1)
#  emb_cond_hom_barbell(10, 0, 0.5, 50000, 1, 1)
#  cond_hom_filt_csv(1000, 1000, 0, 1)
# y1 = chdf(100, 100)
# y1 = chdf_path_fast(1000, 100, 0, 0)
# y1 = chdf_path_fast(1000, 100, 1, 1)
# y1 = chdf_path_fast(10000, 1000, 4, 5)

#chdf_path_fast(1000, 100, 0, 1)


