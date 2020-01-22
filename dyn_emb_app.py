from embedding.dyn_emb import Dyn_Emb
import numpy as np
import csv
import seaborn as sns
from matplotlib import pyplot as plt
import progressbar
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable


class dyn_emb_app():
    def __init__(self, sources, iterations=500, filt_lvl=100, k1=0, k2=1):

        self.sources = sources
        self.k1 = k1  # for path or cyc motif case, length of the left arm
        self.k2 = k2  # for path or cyc motif case, length of the left arm
        self.iterations = iterations  # iterations in each MCMC sampling
        self.filt_lvl = filt_lvl  # number of filtration levels


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

    a1: np.ndarray = np.zeros((rows * cols, rows * cols), dtype=int)
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
    U = np.triu(U, k=1)
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
    U = np.triu(U, k=1)
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


def load_WAN_dataset():
    # initializing the titles and row list
    fields = []
    data_rows = []

    with open('WAN_list.csv', 'r') as csvFile:
        # creating a csv reader object
        reader = csv.DictReader(csvFile)

        # Get field names
        fields = reader.fieldnames

        # extracting each data row one by one
        for row in reader:
            data_rows.append(row)
        # rows is a list object.

    num_files = reader.line_num - 1
    return data_rows, num_files


def wan_generate_dist_mx_frobenius(iter=100):
    # computes classification rate of the WAN data set using CHD profiles
    # data uses filt_lvl = 500

    distance_mx = np.zeros(shape=(9, 9, iter))  # reference * validation * motifs * iteration
    data_rows, num_files = load_WAN_dataset()

    for step in np.arange(iter):
        # sample indices for the validation set for each of the 9 authors
        index_validation = np.random.randint(5, size=(1, 9))
        index_validation = index_validation[0]

        # compute the L1 distance matrix between the reference and validation profiles -- 9 by 9 by 3 by iter matrix
        for x in itertools.product(np.arange(9), repeat=2):  # x[0] = reference author, x[1] = validation author
            i = index_validation[x[0]]
            for j in np.arange(5):  # validation article
                if j != index_validation[x[0]]:  # j th article of x[0]th author is not in the validation set

                    A1 = np.genfromtxt("WAN/" + data_rows[5 * x[0] + j]['filename'] + ".txt", usecols=range(211))
                    A2 = np.genfromtxt("WAN/" + data_rows[5 * x[1] + i]['filename'] + ".txt", usecols=range(211))
                    distance_mx[x[0], x[1], step] += np.linalg.norm(A1 - A2, ord=None)
        print(step)
    # distance_mx = np.sum(distance_mx, axis=2) / (4*distance_mx.shape[2])
    distance_mx = distance_mx / 4
    np.save("distance_mx_frobenius", distance_mx)
    return distance_mx


def wan_generate_dist_mx(iter=100):
    # computes classification rate of the WAN data set using CHD profiles
    # data uses filt_lvl = 500
    a00 = np.load('chd_mx_00.npy')  #a00.shape = (500,45)
    a01 = np.load('chd_mx_01.npy')
    a11 = np.load('chd_mx_11.npy')

    distance_mx = np.zeros(shape=(9, 9, 3, iter))  # reference * validation * motifs * iteration
    for step in np.arange(iter):
        # sample indices for the validation set for each of the 9 authors
        index_validation = np.random.randint(5, size=(1, 9))
        index_validation = index_validation[0]

        # compute the L1 distance matrix between the reference and validation profiles -- 9 by 9 by 3 by iter matrix
        for x in itertools.product(np.arange(9), repeat=2):  # x[0] = reference author, x[1] = validation author
            i = index_validation[x[0]]
            for j in np.arange(5):  # validation article
                if j != index_validation[x[0]]:  # j th article of x[0]th author is not in the validation set
                    distance_mx[x[0], x[1], 0, step] += np.linalg.norm(a00[:, 5 * x[0] + j] - a00[:, 5*x[1]+i], ord=1)/500
                    distance_mx[x[0], x[1], 1, step] += np.linalg.norm(a01[:, 5 * x[0] + j] - a01[:, 5*x[1]+i], ord=1)/500
                    distance_mx[x[0], x[1], 2, step] += np.linalg.norm(a11[:, 5 * x[0] + j] - a11[:, 5*x[1]+i], ord=1)/500

    distance_mx = distance_mx / 4

    np.save("distance_mx", distance_mx)
    return distance_mx


def wan_chd_distance_mx():
    # a = np.load('distance_mx.npy')
    a = np.load('distance_mx_frobenius.npy')
    # c00 = np.sum(a[:, :, 0, :], axis=2) / a.shape[3]
    # c01 = np.sum(a[:, :, 1, :], axis=2) / a.shape[3]
    # c11 = np.sum(a[:, :, 2, :], axis=2) / a.shape[3]

    '''
    # Make the figure:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)

    # Make 4 subplots:
    im1 = axs[0].imshow(c00)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    im2 = axs[1].imshow(c01)
    divider = make_axes_locatable(axs[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)

    im3 = axs[2].imshow(c11)
    divider = make_axes_locatable(axs[2])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3)
    plt.show()
    '''

    # Make the figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05)

    # Make 4 subplots:
    # im1 = ax.imshow(c11)
    im1 = ax.imshow(a)
    ax.set_xticks(np.arange(9))
    # ax.set_yticks(np.arange(9))
    # axs.set_yticks(np.arange(9))
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cax1.tick_params(labelsize=20)
    fig.colorbar(im1, cax=cax1)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    fig.savefig('WAN/classification/classification_frobenius')
    plt.show()

def wan_dendrogram():
    a = np.load('distance_mx.npy')
    c00 = np.sum(a[:, :, 0, :], axis=2) / a.shape[3]
    c01 = np.sum(a[:, :, 1, :], axis=2) / a.shape[3]
    c11 = np.sum(a[:, :, 2, :], axis=2) / a.shape[3]
    authors_list = np.loadtxt('authors_list.txt', delimiter=',', dtype='str')
    demb = Dyn_Emb(c11)
    demb.plot_dendro(cap=1.5, labels=authors_list)


def wan_compute_classification_rate():
    # a = np.load('distance_mx.npy')  # a.shape = (9,9,3,1000)
    a = np.load('distance_mx_frobenius.npy')  # a.shape = (9,9,3,1000)

    '''
    rate = np.zeros(shape=(9,3))  # author * motif
    for i in np.arange(a.shape[3]):
        for motif in np.arange(3):
            for author in np.arange(9):
                c = a[:,author, motif, i]
                idx = np.where(c == np.min(c))
                idx = idx[0]
                prediction = idx[0]  # just to extract the index -- could use the custom index function here
                if prediction == author:
                    rate[author, motif] += 1
                    print(c)
                    print(prediction)
    rate = rate/a.shape[3]
    np.save("classification_rate", rate)
    '''
    rate = np.zeros(shape=(9, 3))  # author * motif
    for i in np.arange(a.shape[2]):
            for author in np.arange(9):
                c = a[:, author, i]
                idx = np.where(c == np.min(c))
                idx = idx[0]
                prediction = idx[0]  # just to extract the index -- could use the custom index function here
                if prediction == author:
                    rate[author] += 1
                    print(c)
                    print(prediction)
    rate = rate / a.shape[2]
    np.save("classification_rate_fronenius", rate)

    return rate

def main():
    # sources = [r'''WAN\abbott_1.txt''']
    # sources = [r'''network3.csv''']
    # sources = [r'''SBM_network1.txt''']

    data_rows, num_files = load_WAN_dataset()
    iterations = 5000
    filt_lvl = 500
    k1 = 1
    k2 = 1
    chd_mx = np.zeros((filt_lvl, 1))
    # A = torus_ER_adj(10, 10, 0)
    # np.savetxt('torus_add', A)

    bar = progressbar.ProgressBar()
    for i in bar(np.arange(35,36)):
        # A = barbell_adj(10, 10, 0, 0.2)
        # A = np.genfromtxt(path, delimiter=',')
        # A = np.genfromtxt(path, usecols=range(60))
        A = np.genfromtxt("WAN/"+data_rows[i]['filename'] + ".txt", usecols=range(211))
        # A = np.genfromtxt("torus_adj.txt", usecols = range(100))
        # A = np.log(A+1)
        # A = A + A.transpose()
        A = A / np.max(A)

        demb = Dyn_Emb(A)
        # hd_cycle_exact = demb.hd_edge_exact()
        # hd_cycle_exact = demb.hd_2path_exact()
        # demb.chdp_edge_exact(filt_lvl=100)
        # y = demb.chdp_path_exact(filt_lvl=filt_lvl, k1=k1, k2=k2)

        # demb.hd_path(iterations=500, k1, k2)
        # z1, z2 = demb.chd_path(iterations=iterations, k1=k1, k2=k2)
        # print(z1, z2)
        y1, y2 = demb.chdp_path(iterations=iterations, filt_lvl=filt_lvl, k1=k1, k2=k2)
        '''
        For WAN data, the two chains only explore the giant connected component. 
        plt.spy(A, precision=0.001)
        '''


        ### Draw dendrogram
        # functionwords_list = np.loadtxt('functionwords_list.txt', dtype='str')
        # demb.plot_dendro(cap=1.5, labels=functionwords_list)
        ###


        # Draw network plot after log scaling       

        '''
        cmap = plt.cm.jet
        image = cmap(A)
        save_filename = data_rows[i]['filename'] + ".png"
        plt.title(data_rows[i]['Author'] + " - " + data_rows[i]['Title'])
        plt.imsave('WAN/network_plot/' + "_test_" + save_filename, image)
        '''

        '''
        Draw CHDP
        '''

        '''
        # draw figure
        sns.set_style('darkgrid', {'legend.frameon': True})
        fig = plt.figure(figsize=(5,5))
        x = np.arange(filt_lvl) / filt_lvl
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, s=3, c='b', marker="s", label='exact')
        # ax1.scatter(x, y1, s=3, c='b', marker="s", label='pivot')
        # ax1.scatter(x, y2, s=3, c='r', marker="s", label='Glauber')
        plt.legend()
        plt.axis([0, 1, 0, 1])
        ax1.set_aspect(1)
        plt.xlabel('filtration level', fontsize=15)
        plt.ylabel('density profile', fontsize=15)
        # plt.title(data_rows[i]['Author'] + " - " + data_rows[i]['Title'] + "\n k1=%i and k2=%i" % (k1, k2))
        plt.title(data_rows[i]['Title'], fontsize=20)
        # save_filename = data_rows[i]['filename'] + "_" + str(k1) + str(k2) + ".png"
        # save_filename = "mcmc_" + data_rows[i]['filename'] + "_" + str(k1) + str(k2)  + ".png"
        plt.subplots_adjust(left=0.12, bottom=0.1, right=0.88, top=0.9, wspace=0, hspace=0)
        save_filename = data_rows[i]['filename'] + "_" + str(k1) + str(k2) + ".png"
        fig.savefig('WAN/profiles_mcmc_11_tight/' + save_filename)
        # plt.show()
        '''


        '''
        # Save CHD profiles as an array
        

        # chd_mx = np.hstack((chd_mx, y))  # for exact chd profiles
        y = (y1+y2)/2  # y.shape = (filt_lvl, )
        y = y.reshape(filt_lvl, -1)   # y.shape = (filt_lvl, 1)
        chd_mx = np.hstack((chd_mx, y))   # for approximate chd profiles

    chd_mx = np.delete(chd_mx, 0, 1)  # delete the first column of zeros
    np.save("chd_mx_11", chd_mx)
    '''

    # dist = wan_generate_dist_mx(iter=1000)
    # dist = wan_generate_dist_mx_frobenius(iter=1000)
    # rate = wan_compute_classification_rate()
    # wan_chd_distance_mx()
    # wan_dendrogram()

if __name__ == '__main__':
    main()




