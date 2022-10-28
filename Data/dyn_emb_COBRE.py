import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from os import listdir
from os.path import isfile, join
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from time import time
from NNetwork import NNetwork as nn
import plotly.graph_objects as go
from numpy import genfromtxt
from tqdm import trange


def collect_MACC_fb(k=21):
    # myfolder = "Data/Facebook/"
    # onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    list = np.arange(124)
    MACC_mx = np.zeros(shape=(k**2, len(list)))
    for i in range(len(list)):
        MACC_mx[:,i] = np.load("COBRE_sim/" + str(list[i]) + ".npy").reshape(k**2,)

    np.save("MACC_mx_COBRE", MACC_mx)
    return MACC_mx

def plot_MDS_fb(dim=2, kmeans_num_clusters=5):

    network_labels = genfromtxt("Data/COBRE_src/labels.csv", delimiter=' ')
    list = np.arange(124)
    list_new = []
    #for f in list:
    #    list_new.append(f.replace('.txt', ''))
    #list = list_new
    A = np.load("MACC_mx_COBRE.npy")
    mds2 = manifold.MDS(2, max_iter=100, n_init=1)
    mds3 = manifold.MDS(3, max_iter=100, n_init=1)
    trans_data2 = mds2.fit_transform(A.T).T
    trans_data3 = mds3.fit_transform(A.T).T

    ### compute kmeans cluster labels
    #kmeans = KMeans(n_clusters=kmeans_num_clusters, random_state=0).fit(A.T)
    #labels = kmeans.labels_
    labels = genfromtxt("Data/COBRE_src/labels.csv", delimiter=' ')  #1 or -1
    labels = np.asarray(2*labels - 1, dtype='int')

    random_colors = []
    for i in range(len(labels)):
        random_colors.append(np.random.rand(3))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []})

    if dim==2:
        ax = fig.add_subplot(111)
        for i in range(len(list)):
            ax.scatter(trans_data2[0][i], trans_data2[1][i], marker='o', color=random_colors[labels[i]])
            ax.text(trans_data2[0][i] - .05, trans_data2[1][i] + .05, list[i], fontsize=5)
        plt.tight_layout()
    else:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(list)):
            ax.scatter(trans_data3[0][i], trans_data3[1][i], trans_data3[2][i], marker='o', color=random_colors[labels[i]])
            ax.text(trans_data3[0][i] - .05, trans_data3[1][i] + .03, trans_data3[2][i]+0.05, list[i], fontsize=5)
        plt.tight_layout()

    # ax[1] = fig.add_subplot(122, projection='3d')
    # ax[1].scatter(trans_data3[0], trans_data3[1], trans_data3[2], label=list)
    # for i, txt in enumerate(list):
    #    ax[1].annotate(list[i], (trans_data3[0][i], trans_data3[1][i], trans_data3[2][i]))

    fig.savefig('COBRE_sim/' + 'MACC_MDS_' + str(dim) + ".png", bbox_inches='tight')
    plt.show()

def plot_fb_baseline_clustering(num_schools=100, kmeans_num_clusters=5, show_dendrogram=False):
    ### Use number of nodes and avg degree for baseline k-means clutering
    a = np.load("COBRE_sim/data_ntwk_computation_full.npy", allow_pickle=True)
    b = a.item()
    c = [f for f in b.keys()]  ## list of schools
    d = np.random.choice(c, num_schools, replace=False)  ## sublist of schools
    ct = []
    for i in d:
        ct.append(b.get(i).get('computation time'))
    idx = np.argsort(ct)

    d_sorted = []
    comp_time = []
    num_nodes = []
    avg_deg = []
    for j in range(len(idx)):
        d_sorted.append(d[idx[j]].replace('.txt', ''))
        comp_time.append(2 * np.round(b.get(d[idx[j]]).get('computation time'), decimals=2))
        num_nodes.append(np.round(b.get(d[idx[j]]).get('num nodes'), decimals=2))
        avg_deg.append(np.round(b.get(d[idx[j]]).get('avg deg'), decimals=2))

    A = np.hstack((np.asarray(num_nodes).reshape(-1,1), np.asarray(avg_deg).reshape(-1,1)))
    ### Normalize A to compare networks relatively
    A[:, 0] = A[:, 0]/np.max(A[:,0])
    A[:, 1] = A[:, 1] / np.max(A[:, 1])

    if show_dendrogram:
        dist_mx = np.zeros(shape=(num_schools, num_schools))
        for x in itertools.product(np.arange(len(d_sorted)), repeat=2):
            A1 = A[x[0], :]
            A2 = A[x[1], :]
            dist_mx[x] = np.linalg.norm(A1/np.max(A[:,0]) - A2/np.max(A[:,1]), ord=2)
        Z = linkage(dist_mx, 'single')
        # Make the dendrogram

        fig, axes = plt.subplots(1, 1, figsize=(10, 3))
        R = dendrogram(Z, ax=axes, orientation='top', leaf_rotation=90, labels=d_sorted, leaf_font_size=6,
                       color_threshold=0.25)  # orientation="left"
        # plt.title("Hierarchical Clustering Dendrogram" + "\n Jane Austen - Pride and Prejudice")
        # plt.ylabel('sample index')
        plt.ylabel('distance')
        # Add horizontal line.
        # plt.axvline(x=1, c='grey', lw=1, linestyle='dashed')
        # plt.xticks(np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5]), np.array([1, 0.75, 0.50, 0.25, 0, -0.25, -0.5]))
        plt.tight_layout()
        plt.show()

    ### compute kmeans cluster labels
    #kmeans = KMeans(n_clusters=kmeans_num_clusters, random_state=0).fit(A)
    #labels = kmeans.labels_
    labels = genfromtxt("Data/COBRE_src/labels.csv", delimiter=' ')
    labels = np.asarray(2*labels - 1, dtype='int')

    random_colors = []
    for i in range(len(labels)):
        random_colors.append(np.random.rand(3))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []})

    ax = fig.add_subplot(111)
    for i in range(len(d_sorted)):
        ax.scatter(num_nodes[i], avg_deg[i], marker='o', color=random_colors[labels[i]])
        ax.text(num_nodes[i] - 1, avg_deg[i] + 1, d_sorted[i], fontsize=5)
    ax.set_xlabel('num nodes')
    ax.set_ylabel('avg degree')
    plt.tight_layout()

    fig.savefig('COBRE_sim/' + 'baseline_MDS_' + ".png", bbox_inches='tight')
    plt.show()


def compute_dist_mx_fb():
    # myfolder = "Data/Facebook/"
    # onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    #list = np.arange(124)
    list = np.arange(124)
    dist_mx = np.zeros(shape=(len(list), len(list)))
    for x in itertools.product(np.arange(len(list)), repeat=2):
        A1 = np.load("COBRE_sim/" + str(x[0]) + ".npy")
        A11 = np.flip(A1)
        A2 = np.load("COBRE_sim/" + str(x[1]) + ".npy")
        dist_mx[x] = np.min((np.linalg.norm(A1-A2, ord=2), np.linalg.norm(A11-A2, ord=2)))

        print('current iteration (%i, %i) out of (num_texts, num_texts)' % (x[0], x[1]))
    print("!!! computed distance matrix")
    np.save("dist_mx_COBRE", dist_mx)
    return dist_mx

def fb_dendrogram(path=None):
    if path is not None:
        a = np.load(path)
    else:
        a = np.load("dist_mx_COBRE.npy")
    #list = np.arange(124)
    list_new = np.arange(124)
    #list_new = []
    #for f in list:
    #    list_new.append(f.replace('.txt', ''))

    Z = linkage(a, 'single')
    # Make the dendrogram

    labels = genfromtxt("Data/COBRE_src/labels.csv", delimiter=' ')  #1 or -1
    #labels = np.asarray(2*labels - 1, dtype='int')
    list_new = np.asarray(list_new * labels, dtype="int")

    fig, axes = plt.subplots(1, 1, figsize=(10, 3))
    R = dendrogram(Z, ax=axes, orientation='top', leaf_rotation=90, labels=list_new, leaf_font_size=6, color_threshold=1)  # orientation="left"
    # plt.title("Hierarchical Clustering Dendrogram" + "\n Jane Austen - Pride and Prejudice")
    # plt.ylabel('sample index')
    plt.ylabel('distance')
    # Add horizontal line.
    # plt.axvline(x=1, c='grey', lw=1, linestyle='dashed')
    # plt.xticks(np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5]), np.array([1, 0.75, 0.50, 0.25, 0, -0.25, -0.5]))
    plt.tight_layout()
    plt.show()
    return R

def fb_display_clustering_mx(path=None):
    #  display learned dictionary
    #list = np.arange(124)
    list_new = np.arange(124)
    #list_new = []
    #for f in list:
    #      list_new.append(f.replace('.txt', ''))
    idx = np.argsort(list_new)

    # fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(7, 5),
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(10, 11),
                            subplot_kw={'xticks': [], 'yticks': []})

    for ax, j in zip(axs.flat, range(100)):
        # ax.set_xlabel('%1.2f' % importance, fontsize=15)
        # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
        if path is not None:
            a = np.load(path)
        else:
            a = np.load("COBRE_sim/" + str(list_new[idx[j]])+".npy")
        ax.imshow(np.sqrt(a), cmap="viridis", interpolation='nearest')
        ax.set_xlabel(list_new[idx[j]], fontsize=7)
        ax.xaxis.set_label_coords(0.5, -0.05)
        # use gray_r to make black = 1 and white = 0

    # plt.suptitle(title)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
    fig.savefig('COBRE_sim/' + 'fb_full_MACC' + ".png", bbox_inches='tight')
    # plt.show()

def make_dict_computation_time():
    a = np.load("COBRE_sim/computation_time.npy", allow_pickle=True)
    list = [f for f in a.item().keys()]
    g = {}
    for i in list:
        print('Current school', i)
        edgelist = np.genfromtxt("Data/Facebook/"+i, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G = nx.Graph(edgelist)
        l_sub = {'computation time': a.item().get(i), 'num nodes': len(G.nodes), 'avg deg': 2*len(G.edges())/len(G.nodes)}
        g.update({i:l_sub})
        np.save("COBRE_sim/data_ntwk_computation_full", g)
    print(g)

def print_computation_time(num_schools=20):
    a = np.load("COBRE_sim/data_ntwk_computation_full.npy", allow_pickle=True)
    b = a.item()
    c = [f for f in b.keys()]  ## list of schools
    d = np.random.choice(c, num_schools, replace=False) ## sublist of schools
    ct = []
    for i in d:
        ct.append(b.get(i).get('computation time'))
    idx = np.argsort(ct)

    # 5.271885395050049  -- my laptop, Caltech36
    # 10.472821474075317 -- cluster, Caltech36

    d_sorted = []
    comp_time = []
    num_nodes = []
    avg_deg = []
    for j in range(len(idx)):
        d_sorted.append(d[idx[j]].replace('.txt', ''))
        comp_time.append(2*np.round(b.get(d[idx[j]]).get('computation time'), decimals=2))
        num_nodes.append(np.round(b.get(d[idx[j]]).get('num nodes'), decimals=2))
        avg_deg.append(np.round(b.get(d[idx[j]]).get('avg deg'), decimals=2))

    fig = go.Figure(data=[go.Table(
                            columnorder=[1, 2, 3, 4],
                            columnwidth=[80, 80, 80, 80],
                            header=dict(values=['School', 'Computation time (sec)', 'Number of nodes', 'Average degree']),
                            cells=dict(values=[d_sorted, comp_time, num_nodes, avg_deg]))
                          ])
    fig.show()

def read_COBRE():

    print("Reading in COBRE networks ...")
    dataset = "COBRE_src"
    school = "COBRE_edges.txt"

    directory = "Data/" + dataset + "/"
    path = directory + school

    edgelist = []
    with open(path) as f:
        for line in f:
            #line = line[0:len(line)-1]
            edgelist.append([float(x) for x in line.split(" ")])

    ## Read in COBRE concatenated brain network edgelists and split them into different networks
    ## Delimiter == [-1,-1]

    network_labels = genfromtxt("Data/COBRE_src/labels.csv", delimiter=' ')
    # 1 == schizophrenic subjects, -1 == healthy subjects

    list_edgelist = []
    edgelist_new = []
    network_number = 0
    for i in trange(len(edgelist)):
        edge = np.asarray([np.round(edgelist[i][0],0), np.round(edgelist[i][1],0)], dtype=int)
        if (edge[0] == -1) and (edge[1] == -1):
            list_edgelist.append([edgelist_new, network_labels[network_number]])
            #np.savetxt("Data/COBRE/COBRE_ntwk_" + str(network_number), edgelist_new)
            edgelist_new = []
            network_number += 1
        else:
            edgelist_new.append([edge[0], edge[1], np.abs(edgelist[i][-1])])
            ## edge weights in COBRE brain networks can be negative
            ## can also consider adding a global constant to make everything nonnegative.

    return list_edgelist



def main_COBRE(k = 20,
               read_COBRE_MACC=False,
               compute_dendrogram=True,
               ):
    ### set motif arm lengths
    k1 = 0
    k2 = k


    """
    dataset = "COBRE_src"
    school = "COBRE_edges.txt"

    directory = "Data/" + dataset + "/"
    path = directory + school
    """

    if read_COBRE_MACC:

        list_edgelist = read_COBRE()
        dict_comp_time = {}
        # r = np.load("COBRE_sim/computation_time.npy", allow_pickle='TRUE')
        # computation_time = r.item()

        for i in np.arange(124):
            #path = "Data/COBRE/COBRE_ntwk_" + str(i)
            #edgelist = list(np.genfromtxt("Data/COBRE/COBRE_ntwk_0", delimiter=' ', dtype=float))
            G = nn.NNetwork()
            G.add_edges(list_edgelist[i][0])
            """
            G = nx.DiGraph()
            for e in edgelist:
                G.add_edge(e[0], e[1], weight=e[2])
                if e[0] != e[1]:
                    G.add_edge(e[1], e[0], weight=e[2]) # symmetric edge weight
            #print('Currently sampling a path motif from COBRE network No. ' +str(i) + "...")


            motif_MCMC = Network_Motif_MCMC(source=None,
                                            G = G,
                                            MCMC_iterations=20,   # MCMC steps (macro, grow with size of ntwk)
                                            k1=k1, k2=k2,  # left and right arm lengths for chain motif
                                            is_glauber_dict=True,  # keep true to use Glauber chain for motif sampling
                                            drop_edge_prob=0)
            """
            title = str(i)
            t0 = time()
            X, emb = G.get_patches(k=k2, sample_size=1000, skip_folded_hom=False, sampling_alg='glauber')
            MACC = np.sum(X, axis=1)/X.shape[1]
            MACC = MACC.reshape(k2,k2)
            t1 = time() - t0


            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), subplot_kw={'xticks': [], 'yticks': []})
            # hom_mx = hom_mx - np.diag(np.ones(k-1), k=1) - np.diag(np.ones(k-1), k=-1
            axs.imshow(MACC, cmap="viridis", interpolation='nearest')

            plt.suptitle("COBRE_" + str(i))
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            fig.savefig('COBRE_sim/' + str(title) + ".png")
            np.save("COBRE_sim/" + str(title), MACC)


            #chd_mx = motif_MCMC.chd_path_mx(k1=k1, k2=k2, iterations=None, is_glauber=True, title=title)


            ### record network sizes and computation time
            A_adj = G.get_adjacency_matrix()
            l_sub = {'computation time': t1-t0,
                     'num nodes': len(G.nodes()),
                     'avg deg': np.linalg.norm(A_adj, 1)/len(G.nodes())}
            dict_comp_time.update({str(i):l_sub})
            # np.save("COBRE_sim/dict_computation_time", dict_comp_time)
            print('time spent:', t1)
            # print(chd_mx)


    if compute_dendrogram:
        print('!!!')
        dist_mx = compute_dist_mx_fb()
        fb_dendrogram(path=None)
        fb_display_clustering_mx()
        # collect_MACC_fb(k=k2)
        plot_MDS_fb(dim=2, kmeans_num_clusters=2)
        # plot_computation_time()
        # print_computation_time(num_schools=30)
        plot_fb_baseline_clustering(num_schools=100, kmeans_num_clusters=5, show_dendrogram=False)




if __name__ == '__main__':
    main_COBRE()
