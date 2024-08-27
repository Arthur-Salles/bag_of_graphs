import numpy as np 
import pyts
import pyts.bag_of_words
import pyts.datasets
import pyts.transformation 
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_array
import itertools
import string

class Bag2Graph():

    def __init__(
        self,
        window_size, 
        word_size,
        n_bins,
    ):
        
        self.bow = pyts.bag_of_words.BagOfWords(window_size, word_size, n_bins)

        self.dict_words = np.apply_along_axis("".join, axis=1,arr=list(itertools.product(*([[i for i in string.ascii_letters[:n_bins]]] * word_size))))

        print(self.dict_words, len(self.dict_words))

    
    def get_filtered_words(self):
        return self.selected_words

    def __get_count_from_bow(self, x_bow): 
        # added to make compatibility with variable length, otherwise just apply regular pyts BoP
        
        X_BoP = dict(zip(self.dict_words, [0] * len(self.dict_words)))
        _sxBow = x_bow.split(' ')

        for i, word in enumerate(_sxBow):
            X_BoP[word] += 1
    
        return list(X_BoP.values())
            

    def __filter_words(self, x_bop, y, K=10):
        f_chi2, p_values = chi2(x_bop, y)
        f_chi2_sorted = np.argsort(f_chi2)
        selected_fchi2 = f_chi2_sorted[:K]

        self.filtered_masks = selected_fchi2
        self.selected_words = self.dict_words[selected_fchi2]
        self.filtered_bop = x_bop[:, selected_fchi2]
        
        print(f'Filtered {K} top words :', self.selected_words)


    def apply_bow_uni(self, x, y, inx=0):
        # fix this later , check the TODO 
        x = x[:, inx] # TODO : add compatibility for other channels and variable length 
        X = check_array(x)
        X_bow = self.bow.fit_transform(X)
        x_bop = np.array([ self.__get_count_from_bow(xb) for xb in X_bow])

        self.__filter_words(x_bop, y)

        filtered_xbow = []
        for inx, bow in enumerate(X_bow):
            splited_bow = np.array(bow.split(' '))
            mask = np.isin(splited_bow, self.selected_words)
            filtered_bow = splited_bow[mask]
            filtered_xbow.append(filtered_bow)

        self.filtered_xbow = filtered_xbow

        # return filtered_xbow

    def get_cooc_matrix(self, inx, include_diagonal=False):

        # # get words to build the fera graph
        f_bow = self.filtered_xbow[inx]

        # # making the maximum possible matrix to be able to sum later
        max_words = np.unique((self.selected_words.flatten()))
        indx_words = dict(zip(max_words, range(len(max_words))))
        r_indx_words = dict(zip(range(len(max_words)), max_words))
        n = len(max_words) ## the adj will be ->  n x n matrix

        # # creating the adjancey words : this could be usefull in the future to experiment K-connections        
        neighbors = np.empty((len(f_bow) - 1, 2), dtype=f_bow.dtype)
        neighbors[:, 0] = f_bow[:-1]
        neighbors[:, 1] = f_bow[1:]

        ## vectorized + hash 
        def __transform_word(a, b): 
            n1, n2 = hash(a) % n, hash(b) % n 
            # n1, n2 = indx_words[a] % n, indx_words[b] % n 
            return n1, n2

        v_func = np.vectorize(__transform_word)

        adj_m = np.zeros((n,n))

        edges = v_func(neighbors[:, 0], neighbors[:, 1])
        print(edges)
        edges_list = np.stack((edges[0], edges[1]), axis=-1) 
        pairs, counts = np.unique(edges_list, axis = 0, return_counts=True)
        adj_m[pairs[:, 0], pairs[:, 1]] = counts 

        if not include_diagonal: 
            np.fill_diagonal(adj_m, 0)

        row_sum = np.sum(adj_m, axis=1)
        normalized_adj = adj_m / row_sum[:, np.newaxis]
        normalized_adj = np.nan_to_num(normalized_adj, nan = 0)

        return adj_m, neighbors, edges_list

