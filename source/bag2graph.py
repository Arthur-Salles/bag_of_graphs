import numpy as np 
import pyts
import pyts.bag_of_words
import pyts.datasets
import pyts.transformation 
from sklearn.feature_selection import chi2
from sklearn.utils.validation import check_array
import itertools
import hashlib
import string

class Bag2Graph():

    def __init__(
        self,
        window_size, 
        word_size,
        n_bins,
        channel_size,
    ):
        
        self.bow = pyts.bag_of_words.BagOfWords(window_size, word_size, n_bins)
        self.dict_words = np.apply_along_axis("".join, axis=1,arr=list(itertools.product(*([[i for i in string.ascii_letters[:n_bins]]] * word_size))))
        self.channel_size = channel_size
    
    def get_filtered_words(self):
        return self.selected_words

    def __get_count_from_bow(self, x_bow): 
        # added to make compatibility with variable length, otherwise just apply regular pyts BoP
        
        X_BoP = dict(zip(self.dict_words, [0] * len(self.dict_words)))
        _sxBow = x_bow.split(' ')

        for i, word in enumerate(_sxBow):
            X_BoP[word] += 1
    
        return list(X_BoP.values())
            

    def __select_agg(self, aggregation_type):
        return np.sum


    def __filter_words(self, x_bop, y, K=10):
        f_chi2, p_values = chi2(x_bop, y)
        f_chi2_sorted = np.argsort(f_chi2)
        selected_fchi2 = f_chi2_sorted[:K]

        self.filtered_masks = selected_fchi2
        self.selected_words = self.dict_words[selected_fchi2]
        self.filtered_bop = x_bop[:, selected_fchi2]

        self.word_to_number = {self.selected_words[i] : i for i in range(len(self.selected_words))}

        # self.word_to_number = {self.selected_words[i] : int(hashlib.sha256(self.selected_words[i].encode()).hexdigest(), 16) % 1000 for i in range(len(self.selected_words))}
        self.number_to_word = {i : self.selected_words[i]  for i in range(len(self.selected_words))}
        print(self.word_to_number)
        print(f'Filtered {K} top words :', self.selected_words,'\n')


    def apply_bow(self, x, y, aggregation_type='sum'):

        t_xbop, t_Xbow = [], []
        filtered_xbow = []

        # TODO : select aggregation method for multivariable here or maybe in another object lol
        agg = self.__select_agg(aggregation_type)
        
        for ch in range(self.channel_size):
            
            X = x[:, ch]
            X = check_array(X)
            X_bow = self.bow.fit_transform(X)
            x_bop = np.array([ self.__get_count_from_bow(xb) for xb in X_bow])
            t_xbop.append(x_bop)
            t_Xbow.append(X_bow)
        
        t_xbop = np.array(t_xbop)
        t_xbop = agg(t_xbop, axis=0)

        self.__filter_words(t_xbop, y)        

        t_Xbow = np.array(t_Xbow).T
        print(t_Xbow.shape) # CANAL, 40 vetores de palavras, 

        # # precisa filtrar todas palavras no bow por canal a agregacao vai ser dar no grafo 
        for ch, word_per_ch in enumerate(t_Xbow):
            split_w = [np.array(w.split(' ')) for w in word_per_ch]
            f_w = [p[np.isin(p, self.selected_words)] for p in split_w]

            filtered_xbow.append(f_w)

        filtered_xbow = np.array(filtered_xbow, dtype='object')
        print('shape', filtered_xbow.shape)

        self.filtered_xbow = filtered_xbow

        return filtered_xbow

    def __np_to_number(self, a, b):
        return self.word_to_number[a], self.word_to_number[b]

    def get_inx_cooc_matrix(self, inx, ch, include_diagonal=False):

        # # get words to build the fera graph
        f_bow = self.filtered_xbow[0, ch]

        # # creating the adjancey words : this could be usefull in the future to experiment K-connections        
        neighbors = np.empty((len(f_bow) - 1, 2), dtype=f_bow.dtype)
        neighbors[:, 0] = f_bow[:-1]
        neighbors[:, 1] = f_bow[1:]

        v_func = np.vectorize(self.__np_to_number)
        edges = v_func(neighbors[:,0], neighbors[:,1])
        # print(edges)

        # print(neighbors)
        edges_list = np.stack((edges[0], edges[1]), axis=-1) 
        # print(edges_list)
        pairs, counts = np.unique(edges_list, axis = 0, return_counts=True)

        edges_with_weights = np.column_stack((np.unique(edges_list, axis = 0, return_counts=True)))

        return pairs, counts, edges_with_weights

    def get_cooc_matrix(self, inx, aggregation_type='sum', include_diagonal=False):

        # # making the maximum possible matrix to be able to sum later
        max_words = np.unique((self.selected_words.flatten()))
        n = len(max_words) ## the adj will be ->  n x n matrix

        adj_m = np.zeros((self.channel_size, n, n))

        agg = self.__select_agg(aggregation_type)

        for ch in range(self.channel_size):

            pairs, counts, edges_with_weight = self.get_inx_cooc_matrix(inx, ch)

            # adj_m[ch, pairs[:, 0], pairs[:, 1]] = counts
            # print(edges_with_weight)
            # print(edges_with_weight)
            # print('----')
        
        adj_m = agg(adj_m, axis=0)

        print(adj_m)
        
        return adj_m
