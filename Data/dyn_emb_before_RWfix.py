import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import random
import numpy.matlib
import seaborn as sns
from sympy import symbols, Matrix, Transpose, sqrt, simplify, series
import progressbar
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
# import time
# import sys
# from io import StringIO

'''
# programs for dynamic embedding
# Main functions are :
# pivot_torus_movie(N, eps, iter, k1, k2)
# Glauber_torus_movie(N, eps, iter, k1, k2)
# emb_cond_hom_torus(N, eps, steps, k1, k2) : computes cond.hom.den. of cycle/path within
# N by N torus + eps density random edges
# emb_cond_hom_csv(steps, k1, k2): computes cond.hom.den. of cycle/path within a network
# whose adjacency matrix is given by a csv file
'''

DEBUG = False


class Dyn_Emb():
    def __init__(self, A):

        self.A = A   # network weight matrix

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def sub2ind(self, array_shape, rows, cols):
        ind = rows*array_shape[1] + cols
        # ind[ind < 0] = -1
        # ind[ind >= array_shape[0]*array_shape[1]] = -1
        return ind

    def ind2sub(self, array_shape, ind):
        # ind = r*cols + c
        r = np.floor(ind/array_shape[1])
        c = ind - r*array_shape[1]
        x = np.array([r, c])

        x = x.astype(int)
        return x

    def path_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        if k1 == 0 or k2 == 0:
            k3 = max(k1,k2)
            A = np.eye(k3 + 1, k=1, dtype=int)
        else:
            A = np.eye(k1+k2+1, k=1, dtype = int)
            A[k1,k1+1] = 0
            A[0,k1+1] = 1
        return A

    def cycle_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        A = self.path_adj(k1, k2)
        if k1 == 0 or k2 == 0:
            k3 = max(k1,k2)
            A[0,k3] = A[0,k3] + 1
        else:
            A[k1, k1 + k2] = A[k1, k1 + k2]+1
        '''
        (!!!Below is implemented directly in programs below -- 9/30/19)
        if k1 == 0 and k2 == 1: # important special case for edge motif -- Otherwise C = path(0,1)-cycle(0,1) = zeros
            A = np.array([[0, 2], [0, 0]])
        elif k1 == 0:
            A = np.eye(k2 + 1, k=1, dtype=int)
            A[0, k2] = 1
        else:
            A = np.eye(k1+k2+1, k=1, dtype=int)
            A[k1, k1+1] = 0
            A[k1, k1+k2] = 1
            A[0, k1+1] = 1
        '''
        return A

    '''
    def RW_kernel(self):
        # Compute RW kernel on a given network
        # A = N by N matrix giving edge weights on networks
        A = self.A
        [N, N] = np.shape(A)
        K = np.zeros([N,N])
        for x in np.arange(N):
            K[x,:] = np.maximum(A[x, :], np.transpose(A[:, x]))
            if sum(K[x, :]) > 0:
                K[x, :] = K[x, :] / sum(K[x, :])

        #  Now modify it using MH-rule so that the stat. dist. is uniform.
        K_new = np.zeros([N,N])
        for x in np.arange(N):
            for y in np.arange(N):
                if y != x:
                    K_new[x,y] = K[x,y]*min(K[y,x], K[x,y])
                else:
                    K_new[x,y] = 1 - sum(K_new[x,:])
        
        return K_new

    def RW_update_kernel(self, kernel, x):
        # RW update from x to y according to a prescribed kernel
        [N,N] = kernel.shape
        if sum(kernel[x,:]) > 0:
            y = np.random.choice(np.arange(0, N), p=kernel[x,:])
        else:
            y = np.random.choice(np.arange(0, N))
        return y   
    '''

    def RW_update(self, x):
        # A = N by N matrix giving edge weights on networks
        # x = RW is currently at site x
        # RW kernel is modified by Metropolis-Hastings to converge to uniform dist. on [N]

        A = self.A
        [N, N] = np.shape(A)
        dist_x = np.maximum(A[x, :], np.transpose(A[:, x]))
        #  dist_x = np.where(dist_x > 0, 1, 0)  # make 1 if positive, otherwise 0
        # (!!! The above line seem to cause disagreement b/w Pivot and Glauber chains in WAN data for
        # k1=0 and k2=1 case and other inner edge CHD cases -- 9/30/19)

        if sum(dist_x) > 0:  # this holds if the current location x of pivot is not isolated
            dist_x_new = dist_x / sum(dist_x)  # honest symmetric RW kernel
            y = np.random.choice(np.arange(0, N), p=dist_x_new)  # proposed move

            # Use MH-rule to accept or reject the move
            dist_y = np.maximum(A[y, :], np.transpose(A[:, y]))
            prop_accept = min(1, dist_x_new[y]*sum(dist_y)/max(A[x,y],A[y,x]) )
            if np.random.rand() > prop_accept:
                y = x
        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.arange(0, N))

        '''
            # (!!! MH-rule doesn't seem to affect the stationary measure, it only slows down the algorithm -9/30/19)
            #  Now modify it using MH-rule so that the stat. dist. is uniform.
            dist = np.zeros(len(dist2))
            for i in np.arange(N):
                dist_i = np.maximum(A[i, :], np.transpose(A[:, i]))
                # dist_i = np.where(dist_i > 0, 1, 0)
                if i != x and sum(dist_i) > 0:
                    dist[i] = dist2[i] * min(sum(dist_x) / sum(dist_i), 1)  # M-H modification
                    #  LHS equals P(x,i)
            dist[x] = 1 - sum(dist)
            '''
        return y



    def Path_sample(self, x, k):
        # A = N by N matrix giving edge weights on networks
        # number of nodes in path
        # samples a path of length k from a given pivot x as the first node

        A = self.A
        [N, N] = np.shape(A)
        emb = np.array([x]) # initialize path embedding

        for i in np.arange(0,k+1):
            dist = A[emb[i], :] / sum(A[emb[i], :])
            y = np.random.choice(np.arange(0, N), p=dist)
            emb = np.hstack((emb, y))

        return emb

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        return min(j)

    def tree_sample(self, B, x):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node

        A = self.A
        [N, N] = np.shape(A)
        [k, k] = np.shape(B)
        emb = np.array([x])  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N, size=(1, k-1))
            y = y[0]
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                if sum(A[emb[j], :]) > 0:
                    dist = A[emb[j], :] / sum(A[emb[j], :])
                    y = np.random.choice(np.arange(0, N), p=dist)
                else:
                    y = emb[j]
                    print('isolated')
                emb = np.hstack((emb, y))

        return emb

    def Path_sample_gen_position(self, x, k1=0, k2=2):
        # A = N by N matrix giving edge weights on networks
        # number of nodes in path
        # samples k1 nodes to the left and k2 nodes to the right of pivot x

        A = self.A
        [N, N] = np.shape(A)
        emb = np.array([x]) # initialize path embedding

        for i in np.arange(0, k2):
            if sum(A[emb[i], :]) > 0:
                dist = A[emb[i], :] / sum(A[emb[i], :])
                y1 = np.random.choice(np.arange(0, N), p=dist)
            else:
                y1 = emb[i]
                # if the new location of pivot makes embedding the path impossible,
                # just contract the path onto the pivot
            emb = np.hstack((emb, [y1]))

        a = np.array([x])
        b = np.matlib.repmat(a, 1, k1+1)
        b = b[0, :]
        emb = np.hstack((b, emb[1:k2+1]))

        for i in np.arange(0, k1):
            if sum(A[emb[i], :]) > 0:
                dist = A[emb[i], :] / sum(A[emb[i], :])
                y2 = np.random.choice(np.arange(0, N), p=dist)
                emb[i+1] = y2
            else:
                emb[i + 1] = emb[i]

        return emb

    def Pivot_update(self, emb, k1, k2):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule

        [N, N] = np.shape(self.A)
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0)  # new location of the pivot
        B = self.path_adj(k1, k2)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding
        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def pivot_gen_update(self, B, emb):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of a path in the network
        # B = adjacency matrix of the tree motif
        # updates the current embedding using pivot rule

        A = self.A
        [N, N] = np.shape(A)
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0)  # new location of the pivot
        emb_new = self.tree_sample(B, x0)  # new tree embedding
        # print([emb_new[0], emb_new[1], A[emb_new[0], emb_new[1]]])
        return emb_new

    def Glauber_update(self, emb, k1, k2):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of a path in the network
        # updates the current embedding using Glauber rule

        A = self.A
        [N, N] = np.shape(A)
        if k1 + k2 == 0:
            emb[0] = self.RW_update(emb[0])
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

            if sum(dist) > 0:
                dist = dist / sum(dist)
                y = np.random.choice(np.arange(0, N), p=dist)
                emb[j] = y
            #  possibly the initial embedding is not valid so that Glauber update may not be well-defined at some node.
            #  In that case do not attempt to move the node
        return emb

    def glauber_gen_update(self, B, emb):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of the tree motif with adj mx B
        # updates the current embedding using Glauber rule

        A = self.A
        [N, N] = np.shape(A)
        [k, k] = np.shape(B)

        if k == 1:
            # emb[0] = np.random.choice(np.arange(0, N))
            # If B has no edge, conditional measure is uniform over the nodes

            '''
            For the WAN data, there is a giant connected component and the Pivot chain only explores that component. 
            In order to match the Glauber chain, we can let the single node case k1=k2=0 to behave like a RW. 
            '''
            emb[0] = self.RW_update(emb[0])
        else:
            j = random.randint(0, k-1)  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            dist = np.ones(N, dtype=int)
            for r in nbh_in:
                dist = dist * A[emb[r], :]
            for r in nbh_out:
                dist = dist * np.transpose(A[:, emb[r]])
            if sum(dist) > 0:
                dist = dist / sum(dist)
                y = np.random.choice(np.arange(0, N), p=dist)
                emb[j] = y
            else:
                emb[j] = np.random.choice(np.arange(0, N))
                print('Glauber move rejected')  # Won't happen
        return emb

    def chd_gen(self, B, C, iterations=1000):
        # computes conditional homomorphism density t(H,A|F) using Pivot chain time average,
        # which is prob. of seeing all extra edges given that we have all edges
        # in tree motif with adj matrix B
        # iterations = number of iteration
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

        A = self.A
        [N, N] = np.shape(A)
        [k,k] = np.shape(B)
        x0 = random.randint(0, N)  # random initial location of RW
        emb1 = self.tree_sample(B, x0)  # initial sampling of path embedding
        emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding

        hom1 = np.array([])
        hom2 = np.array([])
        hom_mx1 = np.zeros([k,k])
        hom_mx2 = np.zeros([k,k])
        bar = progressbar.ProgressBar()
        for i in bar(range(iterations)):
            emb1 = self.pivot_gen_update(B, emb1)
            emb2 = self.glauber_gen_update(B, emb2)
            # print([emb1[0], emb2[0]])
            J = np.where(C - B == 1)  # count only edges in C-B
            a = J[0]
            b = J[1]

            wt1 = np.array([1])
            for w in np.arange(len(a)):
                wt1 = wt1 * A[emb1[0], emb1[1]]

            wt2 = np.array([1])
            for w in np.arange(len(a)):
                wt2 = wt2 * A[emb2[0], emb2[1]]
            # print(wt1, wt2)
            # print(A[emb1[a[w]], emb1[b[w]]])
            # print(A[emb2[a[w]], emb1[b[w]]])
            hom1 = np.hstack((hom1, wt1))
            hom2 = np.hstack((hom2, wt2))

            # full adjacency matrix over the path motif
            a1 = np.zeros([k,k])
            a2 = np.zeros([k,k])
            for q in np.arange(k):
                for r in np.arange(k):
                    a1[q, r] = A[emb1[q], emb1[r]]
                    a2[q, r] = A[emb2[q], emb2[r]]

            hom_mx1 = ((hom_mx1 * i) + a1 ) / (i+1)
            hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            #  progress status
            if 100 * i / iterations % 1 == 0:
                print(i / iterations * 100)

        # take time average
        y1 = np.zeros(iterations)
        y2 = np.zeros(iterations)
        for i in np.arange(1, iterations):
            y1[i] = ((y1[i - 1] * i) + hom1[i]) / (i + 1)
            y2[i] = ((y2[i - 1] * i) + hom2[i]) / (i + 1)

        '''
        # take time average of the induced adj mx
        z1 = np.zeros([k,k, iterations])
        z2 = np.zeros([k,k, iterations])
        for i in np.arange(1, iterations):
            z1[:,:, i] = ((z1[:,:,i - 1] * i) + hom_mx1[:,:,i]) / (i + 1)
            z2[:,:, i] = ((z2[:,:,i - 1] * i) + hom_mx2[:,:,i]) / (i + 1)
        '''

        # draw figure
        sns.set_style('darkgrid', {'legend.frameon': True})
        fig = plt.figure()
        x = np.arange(iterations)
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
        ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')
        plt.xlabel('iteration')
        plt.ylabel('con.hom.density')
        # plt.title('%i by %i Torus + %1.2f density random edges' % (N, N, eps))
        #  plt.title('%i by %i Torus, k1=%i and k2=%i ' % (N, N, k1, k2))

        plt.legend()
        plt.axis([0, iterations, 0, 1])
        # ax1.set_aspect(iterations)

        plt.show()
        return hom_mx1, hom_mx2

    def chd_path(self, iterations=1000, k1=0, k2=1):
        B = self.path_adj(k1, k2)
        C = self.cycle_adj(k1, k2)
        z1, z2 = self.chd_gen(B, C, iterations)
        # t = (sum(y1) + sum(y2)) / (2*iterations)  # t(H,G | F)
        # print(z1, z2)
        return z1, z2

    def hd_path(self, iterations=1000, k1=0, k2=1):
        A = self.A
        # computes unconditioned homomorphism density of F in G
        B = np.zeros([k1+k2+1, k1+k2+1], dtype=int)
        C = self.path_adj(k1, k2)
        y1, y2 = self.chd_gen(B, C, iterations)
        t = (sum(y1) + sum(y2)) / (2*iterations)  # t(H,G | F)
        return t

    def cond_hom_filt(self, B, C, iterations = 1000, filt_lvl=100):
        # computes conditional homomorphism density profile f(H,A|F) using Pivot chain time average
        # Matrices for motifs H and F are given by C and B
        # F = directed tree rooted at node 0
        # iterations = number of iteration
        # underlying network = given by A
        # filt_lvl = number of filtration levels

        A = self.A
        A = A/np.amax(A)  # normalize
        [N, N] = np.shape(A)

        x0 = random.randint(0, N)  # random initial location of RW
        emb1 = self.tree_sample(B, x0)  # initial sampling of tree motif F
        emb2 = self.tree_sample(B, x0)  # initial sampling of tree motif F

        hom1 = np.zeros([filt_lvl, 1])
        hom2 = np.zeros([filt_lvl, 1])

        bar = progressbar.ProgressBar()
        for j in bar(range(iterations)):
            emb1 = self.pivot_gen_update(B, emb1)
            emb2 = self.glauber_gen_update(B, emb2)
            J = np.where(C == 1)
            a = J[0]
            b = J[1]

            wt1 = np.ones([filt_lvl, 1])
            for w in np.arange(len(a)):
                for i in np.arange(filt_lvl):
                    wt1[i] = wt1[i] * np.where(A[emb1[a[w]], emb1[b[w]]] > i/filt_lvl, 1, 0)

            wt2 = np.ones([filt_lvl, 1])
            for w in np.arange(len(a)):
                for i in np.arange(filt_lvl):
                    wt2[i] = wt2[i] * np.where(A[emb2[a[w]], emb2[b[w]]] > i / filt_lvl, 1, 0)

            hom1 = np.hstack((hom1, wt1))
            hom2 = np.hstack((hom2, wt2))

            #  progress status
            if 100*j/iterations % 1 == 0:
                print(j/iterations*100)

        #  construct density profiles of filtration
        y1 = np.zeros(filt_lvl)
        y2 = np.zeros(filt_lvl)
        for i in np.arange(filt_lvl):
            y1[i] = sum(hom1[i, :]) / iterations
            y2[i] = sum(hom2[i, :]) / iterations

        return y1, y2

    def chdp_path(self, iterations=1000, filt_lvl=100, k1=0, k2=1):
        # Specify matrices A, B, C here and draw conditional homomorphism density profile f(C,A|B).

        A = self.A
        B = self.path_adj(k1, k2)
        C = self.cycle_adj(k1, k2) - B
        print(C)
        y1, y2 = self.cond_hom_filt(B, C, iterations, filt_lvl)

        # draw figure
        sns.set_style('darkgrid', {'legend.frameon': True})
        fig = plt.figure()
        x = np.arange(filt_lvl) / filt_lvl
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

    def chdp_path_fast(self, iterations=1000, filt_lvl=100, k1=0, k2=1):
        # computes conditional homomorphism density profile f(H,A|F) using Pivot chain time average
        # Matrices for motifs H and F are given by C and B
        # F = directed tree rooted at node 0
        # iterations = number of iteration
        # underlying network = given by A
        # gird = number of filtration levels

        # Specify A
        # A = genfromtxt(r'''C:\Users\colou\PycharmProjects\colorgraph1\SBM_network2.txt''', usecols=range(60))
        # A = genfromtxt(r'''C:\Users\colou\Google Drive\pylib_lyu\austen_pride.txt''', usecols=range(211))
        '''
        for i in np.arange(211):
            if sum(A[i,:]) > 0:
                A[i, :] = A[i, :] / sum(A[i, :])  # Markov kernel
        A = A/np.max(A)
        '''

        A = self.A
        [N, N] = np.shape(A)

        x0 = random.randint(0, N)  # random initial location of RW
        emb1 = self.Path_sample_gen_position(x0, k1, k2)  # initial sampling of tree motif F
        emb2 = self.Path_sample_gen_position(x0, k1, k2)  # initial sampling of tree motif F

        hom1 = np.zeros([filt_lvl, 1])
        hom2 = np.zeros([filt_lvl, 1])

        bar = progressbar.ProgressBar()
        for i in bar(range(iterations)):
            emb1 = self.Pivot_update(emb1, k1, k2)
            emb2 = self.Glauber_update(emb2, k1, k2)
            b = k1
            a = k1 + k2
            ##  should be from k1+k2 to k1

            wt1 = np.ones([filt_lvl, 1])
            for j in np.arange(filt_lvl):
                wt1[j] = wt1[j] * np.where(A[emb1[a], emb1[b]] > j/filt_lvl, 1, 0)

            wt2 = np.ones([filt_lvl, 1])
            for j in np.arange(filt_lvl):
                wt2[j] = wt2[j] * np.where(A[emb2[a], emb2[b]] > j/filt_lvl, 1, 0)

            hom1 = np.hstack((hom1, wt1))
            hom2 = np.hstack((hom2, wt2))
            if i % 100 == 0:
                print(100*i/iterations, flush=True)

        #  construct density profiles of filtration
        y1 = np.zeros(filt_lvl)
        y2 = np.zeros(filt_lvl)
        for i in np.arange(filt_lvl):
            y1[i] = sum(hom1[i, :]) / iterations
            y2[i] = sum(hom2[i, :]) / iterations

        # draw figure
        sns.set_style('darkgrid', {'legend.frameon': True})
        fig = plt.figure()
        x = np.arange(filt_lvl) / filt_lvl
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

    def motif_transform_gen(self, B, C, iterations=1000):
        # computes motif transform A^H using Glauber/Pivot chain time average F-->G
        # Matrices for motif F is given by B
        # C gives the extra edge in H - F
        # F = directed tree rooted at node 0
        # iterations = number of iteration
        # underlying network = given by A

        A = self.A
        A = A/np.amax(A)  # normalize
        [N, N] = np.shape(A)
        [K, K] = np.shape(B)

        x0 = random.randint(0, N)  # random initial location of RW
        emb1 = self.tree_sample(B, x0)  # initial sampling of tree motif F
        # emb2 = tree_sample(B, x0)  # initial sampling of tree motif F

        hom1 = np.zeros([N, N])
        # hom2 = np.zeros([N, N])

        bar = progressbar.ProgressBar()
        for i in bar(range(iterations)):
            emb1 = self.pivot_gen_update(B, emb1)
            # emb2 = glauber_gen_update(B, emb2)
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

    def motif_transform_movie(self, iterations=1000, k1=0, k2=2):
        # computes motif transform A^H using Glauber/Pivot chain time average F-->G
        # Matrices for motif F is given by B
        # C gives the extra edge in H - F
        # F = directed tree rooted at node 0
        # iterations = number of iteration
        # underlying network = given by A

        A = self.A
        B = self.path_adj(k1, k2)
        C = self.cycle_adj(k1, k2) - B

        A = A/np.amax(A)  # normalize
        [N, N] = np.shape(A)
        [K, K] = np.shape(B)

        x0 = random.randint(0, N)  # random initial location of RW
        emb1 = self.tree_sample(B, x0)  # initial sampling of tree motif F
        # emb2 = tree_sample(B, x0)  # initial sampling of tree motif F

        hom1 = np.zeros([N, N])
        # hom2 = np.zeros([N, N])

        fig = plt.figure()
        ims = []
        x = np.arange(0, N)
        y = np.arange(0, N).reshape(-1, 1)

        #bar = progressbar.ProgressBar()
        #for i in bar(range(iterations)):
        for i in np.arange(iterations):
            emb1 = self.pivot_gen_update(B, emb1)
            # emb2 = self.glauber_gen_update(B, emb2)
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

    def motif_transform(self, k1=0, k2=2):
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

        A = self.A
        B = self.path_adj(k1, k2)
        C = self.cycle_adj(k1, k2) - B
        y1 = self.motif_transform_gen(B, C)

        # draw figure
        # sns.set_style('darkgrid', {'legend.frameon': True})


        # plt.title('Mark Twain - The Adventures of Tom Sawyer\n k1=%i and k2=%i' % (0, 0))
        # plt.title('Network 2 \n k1=%i and k2=%i' % (k1, k2))
        # plt.show()

        return y1

    def path_transform(self, k):
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

    def dist_mx(self, cap):
        # compute distance matrix from a given pairwise weight matrix using Floyd-Warshall algorithm
        # input A is a n by n matrix of entries between 0 and 1.
        A = self.A
        [n, n] = np.shape(A)
        # inf = n/np.min(A[A > 0])  # make sure inf is large enough
        B = A.copy()
        B = B.astype(float)
        B[B == 0] = -cap + np.max(B)
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

    def plot_dendro(self, cap):
        # plot dendrogram of an input network A
        A = self.A
        D = self.dist_mx(cap)
        B = squareform(D)  # condensed distance matrix from A
        Z = linkage(B, 'single')
        # Make the dendrogram
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance (single)')
        dendrogram(Z, leaf_rotation=90)
        plt.show()
        return Z