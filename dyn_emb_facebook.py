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
import plotly.graph_objects as go

class Network_Motif_MCMC():
    def __init__(self,
                 source,
                 MCMC_iterations=500,
                 k1=1,
                 k2=2,
                 patches_file='',
                 is_glauber_dict=True,
                 drop_edge_prob=None):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.source = source
        self.MCMC_iterations = MCMC_iterations
        self.k1 = k1
        self.k2 = k2
        self.patches_file = patches_file
        self.is_glauber_dict = is_glauber_dict   ### if false, use pivot chain for dictionary learning
        self.drop_edge_prob = drop_edge_prob
        self.edges_deleted = []

        # read in networks
        G = self.read_networks_as_DiGraph(source)
        self.G = G
        print('number of nodes=', len(G))

    def read_networks_as_DiGraph(self, path):
        p = self.drop_edge_prob
        edgelist = np.genfromtxt(path, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G1 = nx.DiGraph()
        for e in edgelist:
            G1.add_edge(e[0], e[1], weight=1)
            if e[0] != e[1]:
                G1.add_edge(e[1], e[0], weight=1) # symmetric edge weight

        edgelist_sym = [e for e in nx.Graph(G1.edges).edges]
        print('num of edges', len(edgelist_sym))
        if p == None:
            G = G1
        elif p > 0:
            self.G_complete = G1
            edgelist_thinned = []
            self.edges_deleted = []
            for i in range(len(edgelist_sym)):
                edge = edgelist_sym[i]
                if np.random.rand()<p:
                    self.edges_deleted.append(edge)
                else:
                    edgelist_thinned.append(edge)

            G=nx.DiGraph()
            for e in edgelist_thinned:
                G.add_edge(e[0], e[1], weight=1)
                G.add_edge(e[1], e[0], weight=1)

            G.add_nodes_from(G1.nodes)
            # print(self.edges_deleted)
            print('num edges right after thinning', len(nx.Graph(G.edges).edges))
        else: ### Need to add density p edges globally
            self.G_complete = G1
            G = G1.copy()
            self.edges_added = []
            node_list = [v for v in G1.nodes]
            for i in range(len(node_list)):
                for j in range(i, len(node_list)):
                    if np.random.rand() < -p:
                        G.add_edge(node_list[i], node_list[j], weight=1)
                        G.add_edge(node_list[j], node_list[i], weight=1)
                        if G.has_edge(node_list[i], node_list[j]):
                            self.edges_added.append([node_list[i], node_list[j]])
            print('num edges right after adding', len(nx.Graph(G.edges).edges))
            print('num edges added', len(self.edges_added))
        return G

    def read_networks_as_graph(self, path):
        edgelist = np.genfromtxt(path, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G1 = nx.Graph(edgelist)
        print('num of edges', len(edgelist))
        if self.drop_edge_prob == None:
            G = G1
        else:
            self.G_original = G1
            edgelist_dropped = []
            p = self.drop_edge_prob
            self.edges_deleted = []
            for i in range(len(edgelist)):
                edge = edgelist[i]
                if np.random.rand()<p:
                    self.edges_deleted.append(edge)
                else:
                    edgelist_dropped.append(edge)
            G=nx.Graph(edgelist_dropped)
            G.add_nodes_from(G1.nodes)
            print(self.edges_deleted)
        return G

    def read_networks(self, path):
        G = nx.read_edgelist(path, delimiter=',')
        A = nx.to_numpy_matrix(G)
        A = np.squeeze(np.asarray(A))
        print(A.shape)
        # A = A / np.max(A)
        return A

    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def np2nx(self, x):
        ### Gives bijection from np array node ordering to G.node()
        G = self.G
        a = np.asarray([v for v in G])
        return a[x]

    def nx2np(self, y):
    ### Gives bijection from G.node() to nx array node ordering
        G = self.G
        a = np.asarray([v for v in G])
        return np.min(np.where(a == y))

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

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        # (!!! Also finds self-loop)
        return min(j)

    def tree_sample(self, B, x):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node

        G = self.G
        N = len(G)
        k = np.shape(B)[0]
        emb = np.array([x])  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N, size=(1, k-1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                nbs_j = np.asarray([i for i in G[emb[j]]])
                if len(nbs_j) > 0:
                    y = np.random.choice(nbs_j)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))
        # print('emb', emb)
        return emb

    def glauber_gen_update(self, B, emb):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of the tree motif with adj mx B
        # updates the current embedding using Glauber rule

        G = self.G
        k = np.shape(B)[0]

        if k == 1:
            # emb[0] = np.random.choice(np.arange(0, N))
            # If B has no edge, conditional measure is uniform over the nodes

            '''
            For the WAN data, there is a giant connected component and the Pivot chain only explores that component. 
            In order to match the Glauber chain, we can let the single node case k1=k2=0 to behave like a RW. 
            '''
            emb[0] = self.RW_update(emb[0])
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            cmn_nbs = [i for i in G]
            for r in nbh_in:
                nbs_r = [i for i in G[emb[r]]]
                cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
            for r in nbh_out:
                nbs_r = [i for i in G[emb[r]]]
                cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
            if len(cmn_nbs) > 0:
                y = np.random.choice(np.asarray(cmn_nbs))
                emb[j] = y
            else:
                emb[j] = np.random.choice(np.asarray([i for i in G]))
                print('Glauber move rejected')  # Won't happen once valid embedding is established
        return emb

    def glauber_weighted_update(self, B, emb):
        # Glauber chain update rule for weighted graphs
        # probability of a homomorphism is proportional to the product of all edge weights in G
        # emb = current embedding of the tree motif with adj mx B

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
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
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
                print('Glauber move rejected')  # Won't happen once valid embedding is established
        return emb

    def RW_update(self, x):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform

        G = self.G
        nbs_x = np.asarray([i for i in G[x]])  # array of neighbors of x in G
        #  dist_x = np.where(dist_x > 0, 1, 0)  # make 1 if positive, otherwise 0
        # (!!! The above line seem to cause disagreement b/w Pivot and Glauber chains in WAN data for
        # k1=0 and k2=1 case and other inner edge CHD cases -- 9/30/19)

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # choose a uniform element in nbs_x

            # Use MH-rule to accept or reject the move
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            nbs_y = np.asarray([i for i in G[y]])
            prop_accept = min(1, len(nbs_x)/len(nbs_y))

            if np.random.rand() > prop_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray([i for i in G]))
        return y

    def pivot_acceptance_prob(self, x, y):
        # approximately compute acceptance probability for moving the pivot of B from x to y
        G = self.A
        k = self.k1 + self.k2 + 1
        nbs_x = np.asarray([i for i in G[x]])
        nbs_y = np.asarray([i for i in G[y]])
        accept_prob = len(nbs_x) ** (k - 2) / len(nbs_y) ** (k - 2)  # to be modified

        return accept_prob

    def RW_update_gen(self, x):
        # A = N by N matrix giving edge weights on networks
        # x = RW is currently at site x
        # Acceptance prob will be computed by conditionally embedding the rest of B pivoted at x and y

        G = self.G
        nbs_x = np.asarray([i for i in G[x]])

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # proposed move

            accept_prob = self.pivot_acceptance_prob(x, y)
            if np.random.rand() > accept_prob:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray([i for i in G]))
        return y

    def Path_sample_gen_position(self, x):
        # A = N by N matrix giving edge weights on networks
        # number of nodes in path
        # samples k1 nodes to the left and k2 nodes to the right of pivot x

        G = self.G
        k1 = self.k1
        k2 = self.k2
        emb = np.array([x]) # initialize path embedding

        for i in np.arange(0, k2):
            nbs_emb_i = np.asarray([j for j in G[emb[i]]])
            if len(nbs_emb_i) > 0:
                y1 = np.random.choice(nbs_emb_i)
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
            nbs_emb_i = np.asarray([j for j in G[emb[i]]])
            if len(nbs_emb_i) > 0:
                y2 = np.random.choice(nbs_emb_i)
                emb[i+1] = y2
            else:
                emb[i + 1] = emb[i]

        return emb

    def Pivot_update(self, emb):
        # G = underlying simple graph
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule

        k1 = self.k1
        k2 = self.k2
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0)  # new location of the pivot
        B = self.path_adj(k1, k2)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding
        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def chd_gen_mx(self, B, emb, iterations=1000, is_glauber=True):
        # computes B-patches of the input network G using Glauber chain to evolve embedding of B in to G
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif

        G = self.G
        emb2 = emb
        N = len(G)
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding
        hom2 = np.array([])
        hom_mx2 = np.zeros([k, k])

        for i in range(iterations):
            if is_glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)

            # full adjacency matrix over the path motif
            a2 = np.zeros([k, k])
            for q in np.arange(k):
                for r in np.arange(k):
                    a2[q, r] = int(G.has_edge(emb2[q], emb2[r]) == True)

            hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            #  progress status
            print('current iteration %i out of %i' % (i, iterations))

        return hom_mx2, emb2

    def chd_path_mx(self, k1=0, k2=20, iterations=1000, is_glauber=True, if_save=True, title=None):
        G = self.G
        B = self.path_adj(k1=k1, k2=k2)
        print('B.shape', B.shape)
        k = k1 + k2 + 1
        print('k', k)
        x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)

        if iterations is None:
            node_list = [v for v in G]
            iter = np.floor(2 * len(node_list) * np.log(len(node_list))).astype(int)
        else:
            iter = iterations

        hom_mx, emb = self.chd_gen_mx(B, emb, iterations=iter, is_glauber=is_glauber)

        if if_save:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), subplot_kw={'xticks': [], 'yticks': []})
            # hom_mx = hom_mx - np.diag(np.ones(k-1), k=1) - np.diag(np.ones(k-1), k=-1
            axs.imshow(hom_mx, cmap="gray_r", interpolation='nearest')


            file_title = title
            if title is None:
                file_title = 'test'

            plt.suptitle(file_title)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            fig.savefig('Facebook_sim/' + str(title) + ".png")
            np.save("Facebook_sim/" + str(title), hom_mx)
        return hom_mx

    def show_array(self, arr):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5),
                               subplot_kw={'xticks': [], 'yticks': []})
        ax.xaxis.set_ticks_position('bottom')
        ax.imshow(arr)
        plt.show()

def collect_MACC_fb():
    # myfolder = "Data/Facebook/"
    # onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    list = np.load("list_schools.npy")
    MACC_mx = np.zeros(shape=(441, len(list)))
    for i in range(len(list)):
        MACC_mx[:,i] = np.load("Facebook_sim/" + list[i] + ".npy").reshape(441,)

    np.save("MACC_mx_fb", MACC_mx)
    return MACC_mx

def plot_MDS_fb(dim=2, kmeans_num_clusters=5):
    list = np.load("list_schools.npy")
    list_new = []
    for f in list:
        list_new.append(f.replace('.txt', ''))
    list = list_new
    A = np.load("MACC_mx_fb.npy")
    mds2 = manifold.MDS(2, max_iter=100, n_init=1)
    mds3 = manifold.MDS(3, max_iter=100, n_init=1)
    trans_data2 = mds2.fit_transform(A.T).T
    trans_data3 = mds3.fit_transform(A.T).T

    ### compute kmeans cluster labels
    kmeans = KMeans(n_clusters=kmeans_num_clusters, random_state=0).fit(A.T)
    labels = kmeans.labels_

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

    fig.savefig('Facebook_sim/' + 'MACC_MDS_' + str(dim) + ".png", bbox_inches='tight')
    plt.show()

def plot_fb_baseline_clustering(num_schools=100, kmeans_num_clusters=5, show_dendrogram=False):
    ### Use number of nodes and avg degree for baseline k-means clutering
    a = np.load("Facebook_sim/data_ntwk_computation_full.npy", allow_pickle=True)
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
    kmeans = KMeans(n_clusters=kmeans_num_clusters, random_state=0).fit(A)
    labels = kmeans.labels_

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

    fig.savefig('Facebook_sim/' + 'baseline_MDS_' + ".png", bbox_inches='tight')
    plt.show()


def compute_dist_mx_fb():
    # myfolder = "Data/Facebook/"
    # onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    list = np.load("list_schools.npy")
    dist_mx = np.zeros(shape=(len(list), len(list)))
    for x in itertools.product(np.arange(len(list)), repeat=2):
        A1 = np.load("Facebook_sim/" + list[x[0]] + ".npy")
        A11 = np.flip(A1)
        A2 = np.load("Facebook_sim/" + list[x[1]] + ".npy")
        dist_mx[x] = np.min((np.linalg.norm(A1-A2, ord=2), np.linalg.norm(A11-A2, ord=2)))

    print('current iteration (%i, %i) out of (num_texts, num_texts)' % (x[0], x[1]))
    np.save("dist_mx_fb", dist_mx)
    return dist_mx

def fb_dendrogram(path=None):
    if path is not None:
        a = np.load(path)
    else:
        a = np.load("dist_mx_fb.npy")
    list = np.load("list_schools.npy")
    list_new = []
    for f in list:
        list_new.append(f.replace('.txt', ''))

    Z = linkage(a, 'single')
    # Make the dendrogram

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
    list = np.load("list_schools.npy")
    list_new = []
    for f in list:
        list_new.append(f.replace('.txt', ''))
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
            a = np.load("Facebook_sim/" + list_new[idx[j]]+".txt.npy")
        ax.imshow(np.sqrt(a), cmap="gray_r", interpolation='nearest')
        ax.set_xlabel(list_new[idx[j]], fontsize=7)
        ax.xaxis.set_label_coords(0.5, -0.05)
        # use gray_r to make black = 1 and white = 0

    # plt.suptitle(title)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
    fig.savefig('Facebook_sim/' + 'fb_full_MACC' + ".png", bbox_inches='tight')
    # plt.show()

def make_dict_computation_time():
    a = np.load("Facebook_sim/computation_time.npy", allow_pickle=True)
    list = [f for f in a.item().keys()]
    g = {}
    for i in list:
        print('Current school', i)
        edgelist = np.genfromtxt("Data/Facebook/"+i, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G = nx.Graph(edgelist)
        l_sub = {'computation time': a.item().get(i), 'num nodes': len(G.nodes), 'avg deg': 2*len(G.edges())/len(G.nodes)}
        g.update({i:l_sub})
        np.save("Facebook_sim/data_ntwk_computation_full", g)
    print(g)

def print_computation_time(num_schools=20):
    a = np.load("Facebook_sim/data_ntwk_computation_full.npy", allow_pickle=True)
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

def main():
    ### set motif arm lengths
    k1 = 0
    k2 = 20
    n_components = 25
    foldername = 'Facebook_sim'

    myfolder = "Data/Facebook"
    onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    onlyfiles = onlyfiles[15:]
    # onlyfiles.remove('desktop.ini')
    onlyfiles = ['Caltech36.txt']

    # dist_mx = compute_dist_mx_fb()
    fb_dendrogram(path=None)
    # fb_display_clustering_mx()
    # collect_MACC_fb()
    # plot_MDS_fb(dim=2, kmeans_num_clusters=5)
    # plot_computation_time()
    # print_computation_time(num_schools=30)
    # plot_fb_baseline_clustering(num_schools=100, kmeans_num_clusters=5, show_dendrogram=False)

    dict_comp_time = {}
    # r = np.load("Facebook_sim/computation_time.npy", allow_pickle='TRUE')
    # computation_time = r.item()

    for school in onlyfiles:
        directory = "Data/Facebook/"
        path = directory + school
        print('Currently sampling a path motif from ' + school)

        motif_MCMC = Network_Motif_MCMC(source=path,
                                        MCMC_iterations=20,   # MCMC steps (macro, grow with size of ntwk)
                                        k1=k1, k2=k2,  # left and right arm lengths for chain motif
                                        is_glauber_dict=True,  # keep true to use Glauber chain for motif sampling
                                        drop_edge_prob=0)

        title = str(school)
        t0 = time()
        chd_mx = motif_MCMC.chd_path_mx(k1=k1, k2=k2, iterations=None, is_glauber=True, title=title)
        t1 = time() - t0

        ### record network sizes and computation time
        l_sub = {'computation time': t1-t0,
                 'num nodes': len(motif_MCMC.G.nodes),
                 'avg deg': 2 * len(motif_MCMC.G.edges()) / len(motif_MCMC.G.nodes)}
        dict_comp_time.update({school:l_sub})
        # np.save("Facebook_sim/dict_computation_time", dict_comp_time)
        print('time spent:', t1)
        # print(chd_mx)


if __name__ == '__main__':
    main()





