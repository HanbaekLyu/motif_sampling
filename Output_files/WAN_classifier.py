import numpy as np
import scipy.io
import csv
# import mat73
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
from dyn_emb import Dyn_Emb
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def load_WAN_dataset():
    # initializing the titles and row list
    fields = []
    data_rows = []

    with open('WAN_list2.csv', 'r') as csvFile:
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


def compute_KL_div(A, B, is_L2=False, is_MC=True):
    '''
    Computes KL divergence between two WANs
    '''
    A1 = A.copy()
    B1 = B.copy()
    # print('A1', np.sum(np.sum(A1)))
    if is_MC:
        ### Normalized row-wise to make the frequency matrices Markov kernels
        for row in np.arange(A1.shape[0]):
            if np.sum(A1[row, :]) > 0:
                A1[row, :] = A1[row, :] / np.sum(A1[row, :])
            # else:  ### this is used in the original paper (rather not use it)
            #    A1[row, :] = np.ones(shape=A1[row, :].shape) / len(A1[row, :])
        for row in np.arange(B1.shape[0]):
            if np.sum(B1[row, :]) > 0:
                B1[row, :] = B1[row, :] / np.sum(B1[row, :])
            # else:
            #    B1[row, :] = np.ones(shape=B1[row, :].shape) / len(B1[row, :])
    else:
        A1 = A1 / np.max(A1)
        B1 = B1 / np.max(B1)

    if not is_L2:
        ### Compute stationary distribution of A1 by power method
        pi = np.sum(A1, axis=1) / np.sum(np.sum(A1))
        # print('A_row_sum', np.sum(A11, axis=1))
        for i in np.arange(50):
            pi = pi @ A1
        # print('pi', pi)
        ### Now compute relative entropy

        H = A1 / B1
        H = np.where(H == np.inf, 0, H)  ### is this right? Shouldn't we set this to be a large value, not zero?
        H = np.where(H > 0, np.log2(H), 0)
        H = (np.expand_dims(pi, axis=1) @ np.ones(shape=(1, pi.shape[0]))) * A1 * H
        d = np.sum(np.sum(H))

        '''
        d2 = 0
        for y in itertools.product(np.arange(A1.shape[0]), repeat=2):
            ### !!! diagonal is added twice
            if A1[y] * B1[y] > 0:
                d2 += pi[y[0]] * A1[y] * np.log2(A1[y] / B1[y])
        for i in np.arange(A1.shape[0]):
            if A1[i,i] * B1[i,i] > 0:
                d2 -= pi[i] * A1[i,i] * np.log2(A1[i,i] / B1[i,i])
        '''
    else:
        d = np.linalg.norm(A1 - B1, ord=2)
    return d


def WAN_dist_mx(filename, num_top_fwords=50, is_L2=False, is_MC=True):
    '''
    num_top_fwords   ### number of top most used function words to be used in entropy computation
    '''
    data_rows, num_files = load_WAN_dataset()
    num_texts = len(data_rows)

    idx = np.load("fwords_sorting_idx.npy")  ### index that would sort the 211 function words
    data_rows, num_files = load_WAN_dataset()
    dist_mx = np.zeros(shape=(len(data_rows), len(data_rows)))
    for x in itertools.product(np.arange(len(data_rows)), repeat=2):
        A = np.genfromtxt("WAN/" + data_rows[x[0]]['filename'] + ".txt", usecols=range(211))
        A1 = np.zeros(shape=(num_top_fwords, num_top_fwords))
        for y in itertools.product(np.arange(num_top_fwords), repeat=2):
            A1[y] = A[idx[y[0]], idx[y[1]]]

        B = np.genfromtxt("WAN/" + data_rows[x[1]]['filename'] + ".txt", usecols=range(211))
        B1 = np.zeros(shape=(num_top_fwords, num_top_fwords))
        for y in itertools.product(np.arange(num_top_fwords), repeat=2):
            B1[y] = B[idx[y[0]], idx[y[1]]]
        # print('A1.shape', A1.shape)
        dist_mx[x] = compute_KL_div(A, B, is_L2=is_L2, is_MC=is_MC)

        print('current iteration (%i, %i) out of (%i, %i)' % (x[0], x[1], num_texts, num_texts))
    np.save("WAN/classification3/distance_mx_KL_" + "L2_" + str(is_L2) + "_" + filename, dist_mx)
    return dist_mx


def WAN_CHDP_dist_mx(filename, num_top_fwords=50, k1=1, k2=1, iterations=500, filt_lvl=500, is_MC=False):
    '''
    num_top_fwords   ### number of top most used function words to be used in entropy computation
    '''
    data_rows, num_files = load_WAN_dataset()
    num_texts = len(data_rows)
    chd_mx = compute_CHD_profiles(num_top_fwords=num_top_fwords,
                                  iterations=iterations,
                                  filt_lvl=filt_lvl,
                                  k1=k1,
                                  k2=k2,
                                  is_MC=is_MC)
    print('chd_mx.shape', chd_mx.shape)
    dist_mx = np.zeros(shape=(num_texts, num_texts))
    for x in itertools.product(range(num_texts), repeat=2):
        dist_mx[x] = np.linalg.norm(chd_mx[:, x[0]] - chd_mx[:, x[1]], ord=1)/filt_lvl
        # dist_mx[x] = np.abs(np.linalg.norm(chd_mx[:, x[0]],ord=1) - np.linalg.norm(chd_mx[:, x[1]],ord=1))

        print('current iteration (%i, %i) out of (%i, %i)' % (x[0], x[1], num_texts, num_texts))
    np.save("WAN/classification3/distance_mx_CHDP_" + str(k1) + str(k2) + "_" + filename, dist_mx)
    return dist_mx


def find_most_used_words():
    f_list = np.loadtxt('functionwords_list.txt', dtype='str')
    data_rows, num_files = load_WAN_dataset()
    ### get frequency vector of the entire corpus
    freq_vec = np.zeros(211)
    for i in np.arange(len(data_rows)):
        A = np.genfromtxt("WAN/" + data_rows[i]['filename'] + ".txt", usecols=range(211))
        freq_vec += np.sum(A, axis=1)
        freq_vec += np.sum(A, axis=1)
    idx = np.argsort(freq_vec)
    idx = np.flip(idx)
    f_list_sorted = []
    for e in idx:
        f_list_sorted.append(f_list[e])
    np.save("functionwords_list_sorted", f_list_sorted)
    np.save("fwords_sorting_idx", idx)
    return f_list_sorted


def wan_generate_classification_dist_mx(filename, dist='KL', num_top_fwords=50, iter=100, MCMC_iter=500, filt_lvl=500,
                                        is_MC=True, k1=1, k2=1, num_training=4):
    # computes classification rate of the WAN data set using CHD profiles
    # data uses filt_lvl = 500
    data_rows, num_files = load_WAN_dataset()
    k = np.floor((len(data_rows) / 5)).astype(int)
    if dist is 'KL':
        # a00 = np.load('WAN/classification3/distance_mx_KL_10_top_10.npy')  #a00.shape = (500,10)
        a00 = WAN_dist_mx(filename=filename, num_top_fwords=num_top_fwords, is_L2=False, is_MC=is_MC)
    elif dist is 'L2':
        a00 = WAN_dist_mx(filename=filename, num_top_fwords=num_top_fwords, is_L2=True, is_MC=is_MC)
    else:
        a00 = WAN_CHDP_dist_mx(filename, num_top_fwords=50, k1=k1, k2=k2, is_MC=is_MC, iterations=MCMC_iter,
                               filt_lvl=filt_lvl)

    distance_mx = 10 * np.ones(shape=(k, k, iter))  # reference * validation * motifs * iteration
    for step in np.arange(iter):
        # compute the L1 distance matrix between the reference and validation profiles -- 9 by 9 by 3 by iter matrix
        for x in itertools.product(np.arange(k), repeat=2):  # x[0] = reference author, x[1] = validation author
            ix = np.random.choice(5, num_training + 1, replace=False)
            for j in np.arange(num_training):  # validation article
                distance_mx[x[0], x[1], step] = np.minimum(distance_mx[x[0], x[1], step], np.linalg.norm(
                    a00[:, 5 * x[0] + ix[j]] - a00[:, 5 * x[1] + ix[-1]], ord=np.inf) / 500)
        print('current iteration %i out of %i' % (step, iter))

    np.save("WAN/classification3/distance_mx_validation_" + str(dist) + str(k1) + str(k2) + "_" + "num_training_" + str(
        num_training) + "_" + str(filename), distance_mx)
    return distance_mx


def wan_compute_classification_rate(filename, dist='KL', mx=None, num_top_fwords=50, iter=1000, is_L2=False, is_MC=True,
                                    iterations=500, filt_lvl=500, k1=1, k2=1, num_training=4):
    if mx is None:
        # a = np.load('WAN/classification/distance_mx_KL_validation')  # a.shape = (9,9,1000)
        a = wan_generate_classification_dist_mx(filename=filename, dist=dist, num_top_fwords=num_top_fwords,
                                                is_MC=is_MC, iter=iter, k1=k1, k2=k2, num_training=num_training)
    else:
        a = mx
    data_rows, num_files = load_WAN_dataset()
    k = np.floor((len(data_rows) / 5)).astype(int)  ### number of authors

    rate = np.zeros(shape=(k, 1))  # author * motif
    for i in np.arange(a.shape[2]):
        for author in np.arange(k):
            c = a[:, author, i]
            idx = np.where(c == np.min(c))
            idx = idx[0]
            prediction = idx[0]  # just to extract the index -- could use the custom index function here
            if prediction == author:
                rate[author] += 1
                print(c)
                print(prediction)
    rate = rate / a.shape[2]
    # np.save("classification_rate_fronenius", rate)
    np.save("WAN/classification3/classification_rate_" + str(filename), rate)
    return rate


def compute_CHD_profiles(num_top_fwords, k1=1, k2=1, iterations=500, filt_lvl=500, is_MC=False, if_compute_fresh=True, if_plot=False):
    ### compute CHD profiles of the 15 novels WAN data
    data_rows, num_files = load_WAN_dataset()
    print('!!!data_rows', len(data_rows))
    chd_mx = np.zeros((filt_lvl, 1))
    chd_mx1 = np.zeros((filt_lvl, 1))
    k = np.floor((len(data_rows) / 5)).astype(int)
    if if_compute_fresh:
        idx = np.load("fwords_sorting_idx.npy")

        for i in np.arange(0, len(data_rows)):
            A = np.genfromtxt("WAN/" + data_rows[i]['filename'] + ".txt", usecols=range(211))
            ### Restrict on take top k function words
            A1 = np.zeros(shape=(num_top_fwords, num_top_fwords))
            for y in itertools.product(np.arange(num_top_fwords), repeat=2):
                A1[y] = A[idx[y[0]], idx[y[1]]]

            if not is_MC:
                A1 = A1 / np.max(A1)
            else:
                # row-wise normalize to make it a Markov transition matrix
                for row in np.arange(A1.shape[0]):
                    if np.sum(A1[row, :]) > 0:
                        A1[row, :] = A1[row, :] / np.sum(A1[row, :])

            # print('A1', np.max(A1))
            # print('A1.shape', A1.shape)

            demb = Dyn_Emb(A1)
            if k1 == 0 and k2 == 0:
                y = demb.chdp_path_exact(filt_lvl=filt_lvl, k1=k1, k2=k2)
                chd_mx = np.hstack((chd_mx, y))  # use Glauber chain to generate profiles
            else:
                y, y1 = demb.chdp_path(iterations=iterations, filt_lvl=filt_lvl, k1=k1, k2=k2)
                print('y.shape', y.shape)
                chd_mx = np.hstack((chd_mx, np.expand_dims(y, axis=1)))  # use Glauber chain to generate profiles
                chd_mx1 = np.hstack((chd_mx1, np.expand_dims(y1, axis=1)))
                # chd_mx = np.append(chd_mx, y)
            print('chd_mx.shape', chd_mx.shape)

            # demb.hd_path(iterations=500, k1, k2)
            # z1, z2 = demb.chd_path(iterations=iterations, k1=k1, k2=k2)
            # print(z1, z2)
            # y1, y2 = demb.chdp_path(iterations=iterations, filt_lvl=filt_lvl, k1=k1, k2=k2)
            # y = (y1+y2)/2  # y.shape = (filt_lvl, )
            # y = y.reshape(filt_lvl, -1)   # y.shape = (filt_lvl, 1)
            # chd_mx = np.hstack((chd_mx, y))   # for approximate chd profiles
            print('current iteration %i out of %i' % (i, len(data_rows)))

        chd_mx = np.delete(chd_mx, 0, 1)  # delete the first column of zeros
        chd_mx1 = np.delete(chd_mx1, 0, 1)  # delete the first column of zeros
        np.save("WAN\classification3\chd_mx_" + str(k1) + str(k2), chd_mx)

    if if_plot:
        fig, axs = plt.subplots(nrows=k, ncols=5, figsize=(15, 15))
        if not if_compute_fresh:
            chd_mx = np.load("WAN/classification3/chd_mx_00.npy")
        for ax, j in zip(axs.flat, range(len(data_rows))):
            x = np.arange(filt_lvl) / filt_lvl
            if k1 == 0 and k2 == 0:
                ax.scatter(x, chd_mx[:,j], s=3, c='b', marker="s", label = data_rows[j]['Title'])
                print('j, title', j, data_rows[j]['Title'])
                # ax.scatter(x, chd_mx[:,j], '', label = data_rows[j]['Title'])
                plt.axis([0, 1, 0, 1])
                ax.set_aspect(1)
                plt.xlabel('filtration level', fontsize=15)
                plt.ylabel('density profile', fontsize=15)
                # plt.title(data_rows[j]['Title'], fontsize=20)
            else:
                ax.scatter(x, chd_mx[:, j], s=3, c='b', marker="s", label='pivot')
                ax.scatter(x, chd_mx1[:, j], s=3, c='r', marker="s", label='Glauber')
                plt.legend(fontsize=15)
                plt.axis([0, 1, 0, 1])
                ax.set_aspect(1)
                plt.xlabel('filtration level', fontsize=15)
                plt.ylabel('density profile', fontsize=15)

        # plt.title(data_rows[i]['Title'], fontsize=20)
        plt.subplots_adjust(left=0.12, bottom=0.1, right=0.88, top=0.9, wspace=0.1, hspace=0)
        fig.savefig("WAN/classification3/profiles_" + str(k1) + str(k2))
        plt.show()


    return chd_mx


def WAN_classification_comparison():
    list = ['CHDP', 'KL', 'L2']
    for k in range(3):
        mx = np.zeros(shape=(4, 4, 6))  ### num_training * methods * authors + 1
        for i in np.arange(1, 5):
            rate = wan_compute_classification_rate(filename="top_211_rate_comparison",
                                                   dist=list[k],
                                                   mx=None,
                                                   is_MC=True,
                                                   iterations=5000,
                                                   filt_lvl=500,
                                                   num_top_fwords=211,
                                                   iter=1000,
                                                   k1=0, k2=0,
                                                   num_training=i)
            rate = np.vstack((rate, np.sum(rate) / len(rate)))
            rate = rate[0]
            print('rate.shape', rate.shape)
            mx[i - 1, k, :] = rate
    np.save("WAN\classification3\rate_comparision", mx)
    print('mx', mx)
    return mx


def wan_texts_dendrogram(path=None, k1=1, k2=1):
    if path is not None:
        a = np.load(path)
    else:
        a = np.load("WAN/classification3/distance_mx_CHDP_" + str(k1) + str(k2) + "_top_211_hom_noMC.npy")
        # a = np.load("WAN/classification3/distance_mx_KL_L2_False_top_211_rate_comparison.npy")

    data_rows, num_files = load_WAN_dataset()
    text_labels = []
    for i in range(len(data_rows)):
        text_labels.append(data_rows[i]['filename'])

    Z = linkage(a, 'single')
    # Make the dendrogram

    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    dendrogram(Z, ax=axes, orientation='right', leaf_rotation=0, labels=text_labels,
               color_threshold=1)  # orientation="left"
    # plt.title("Hierarchical Clustering Dendrogram" + "\n Jane Austen - Pride and Prejudice")
    # plt.ylabel('sample index')
    plt.xlabel('distance')
    # Add horizontal line.
    # plt.axvline(x=1, c='grey', lw=1, linestyle='dashed')
    # plt.xticks(np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5]), np.array([1, 0.75, 0.50, 0.25, 0, -0.25, -0.5]))
    plt.tight_layout()
    plt.show()
    return Z

class WAN_classification_old:
    def __init__(self,
                 source,
                 source_combined,
                 num_texts):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.source = source
        self.num_texts = num_texts

        # read in data
        self.data_dict = scipy.io.loadmat(source)
        # self.wans_and_freqs_only = mat73.loadmat(source_combined)
        self.key_list = [e for e in self.data_dict.keys()]
        ### key_list[26] = 'wan_abbott_alexander_1' is the first article key
        ### key_list[11155] = 'wan_wharton_summer_9' is the last article key
        self.article_list = self.key_list[26:11155]
        self.article_idx = np.random.choice(self.article_list, num_texts)

    def compute_KL_div(self, A, B):
        '''
        Computes KL divergence between two WANs
        '''
        A1 = A.copy()
        B1 = B.copy()
        # print('A1', np.sum(np.sum(A1)))
        ### Normalized row-wise to make the frequency matrices Markov kernels
        for row in np.arange(A1.shape[0]):
            if np.sum(A1[row, :]) > 0:
                A1[row, :] = A1[row, :] / np.sum(A1[row, :])
            # else:  ### this is used in the original paper (rather not use it)
            #     A1[row, :] = np.ones(shape=A1[row, :].shape) / len(A1[row, :])
        for row in np.arange(B1.shape[0]):
            if np.sum(B1[row, :]) > 0:
                B1[row, :] = B1[row, :] / np.sum(B1[row, :])
            # else:
            #     B1[row, :] = np.ones(shape=B1[row, :].shape) / len(B1[row, :])

        ### Compute stationary distribution of A1 by power method
        pi = np.sum(A1, axis=1) / np.sum(np.sum(A1))
        # print('A_row_sum', np.sum(A11, axis=1))
        for i in np.arange(50):
            pi = pi @ A1
        # print('pi', pi)
        ### Now compute relative entropy

        '''
        H = A1 / B1
        H = np.where(H == np.inf, 0, H)
        H = np.where(H > 0, np.log(H), 0)
        H = (np.expand_dims(pi, axis=1)  @ np.ones(shape=(1, pi.shape[0]))) * A1 * H
        d = np.sum(np.sum(H))
        '''

        d = 0
        for y in itertools.product(np.arange(A1.shape[0]), repeat=2):
            if pi[y[0]] * A1[y] > 0 | B1[y] > 0:
                d += pi[y[0]] * A1[y] * np.log2(A1[y] / B1[y])
        # print('KL_divergence=',d)

        return d

    def wan_dist_mx_KL_divergence(self):
        '''
        Computes KL divergence between WANs in the given list of indices idx
        '''
        num_texts = len(self.article_idx)
        key_list = [e for e in self.data_dict.keys()]
        ### key_list[26] = 'wan_abbott_alexander_1' is the first article key

        distance_mx = np.zeros(shape=(num_texts, num_texts))
        # computes relative entropy between the texts
        for x in itertools.product(np.arange(num_texts), repeat=2):  # x[0] = reference author, x[1] = validation author
            print('article_list', self.article_idx[x[0]])
            A1 = self.data_dict.get(self.article_idx[x[0]])
            A2 = self.data_dict.get(self.article_idx[x[1]])
            distance_mx[x] = self.compute_KL_div(A1, A2)
            print('d', distance_mx[x])
            print('current iteration (%i, %i) out of (num_texts, num_texts)' % (x[0], x[1]))
        np.save("WAN/classification3/distance_max_KL_min_MC", distance_mx)
        print('distance_mx', distance_mx)
        return distance_mx

    def wan_dist_mx_KL_combined(self):
        '''
        Computes KL divergence between WANs in the given list of indices idx
        '''
        num_texts = self.num_texts
        key_list = [e for e in self.wans_and_freqs_only.keys()]
        idx = np.random.choice(337, num_texts)
        distance_mx = np.zeros(shape=(num_texts, num_texts))
        # computes relative entropy between the texts
        for x in itertools.product(np.arange(num_texts), repeat=2):  # x[0] = reference author, x[1] = validation author
            A1 = self.wans_and_freqs_only.get('all_wans')
            A2 = self.wans_and_freqs_only.get(self.article_idx[x[1]])
            A11 = A1.copy()
            A22 = A2.copy()
            print('A1', np.sum(np.sum(A1)))
            ### Normalized row-wise to make the frequency matrices Markov kernels
            for row in np.arange(A11.shape[0]):
                if np.sum(A11[row, :]) > 0:
                    A11[row, :] = A11[row, :] / np.sum(A11[row, :])
            for row in np.arange(A22.shape[0]):
                if np.sum(A22[row, :]) > 0:
                    A22[row, :] = A22[row, :] / np.sum(A22[row, :])

            ### Compute stationary distribution of A1 by power method
            pi = np.sum(A11, axis=1) / np.sum(np.sum(A11))
            print('A_row_sum', np.sum(A11, axis=1))
            for i in np.arange(50):
                pi = pi @ A11
            # print('pi', pi)
            ### Now compute relative entropy
            d = 0
            for y in itertools.product(np.arange(211), repeat=2):
                if pi[y[0]] * A11[y] > 0:
                    if A22[y] == 0:
                        A22[y] = 0.001
                    d += pi[y[0]] * A11[y] * np.log2(A11[y] / A22[y])
            distance_mx[x] = d
            # print('d',d)
            print('current iteration (%i, %i) out of (num_texts, num_texts)' % (x[0], x[1]))
        np.save("WAN/classification3/distance_mx_KL", distance_mx)
        print('distance_mx', distance_mx)
        return distance_mx


def main():
    compute_rate_single = True

    if compute_rate_single:
        rate = wan_compute_classification_rate(filename="top_211_hom_noMC",
                                               dist='CHDP',
                                               mx=None,
                                               is_MC=False,
                                               iterations=5000,
                                               filt_lvl=500,
                                               num_top_fwords=211,
                                               iter=1000,
                                               k1=0, k2=0,
                                               num_training=4)

        print('rate', rate)
        print('rate.shape', rate.shape)
        print('avg_rate', np.sum(rate) / len(rate))

    compute_rate_comparision = False

    if compute_rate_comparision:
        mx = WAN_classification_comparison()

    plot_dendrogram = True

    if plot_dendrogram:
        wan_texts_dendrogram(path=None, k1=0, k2=0)

    plot_profiles = False

    if plot_profiles:
        compute_CHD_profiles(num_top_fwords=211,
                             k1=1, k2=1,
                             iterations=500,
                             filt_lvl=500,
                             is_MC=False,
                             if_compute_fresh=True,
                             if_plot=True)


if __name__ == '__main__':
    main()














