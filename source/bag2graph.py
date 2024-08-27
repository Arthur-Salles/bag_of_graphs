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
        channel_size,
    ):
        
        self.bow = pyts.bag_of_words.BagOfWords(window_size, word_size, n_bins)

        self.dict_words = np.apply_along_axis("".join, axis=1,arr=list(itertools.product(*([[i for i in string.ascii_letters[:n_bins]]] * word_size))))

        print(self.dict_words, len(self.dict_words))
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
        
        print(f'Filtered {K} top words :', self.selected_words,'\n')



    # def apply_bow_uni(self, x, y, inx=0):
    #     x = x[:, inx] # TODO : add compatibility for variable length 
    #     X = check_array(x)
    #     X_bow = self.bow.fit_transform(X)
    #     x_bop = np.array([ self.__get_count_from_bow(xb) for xb in X_bow])

    #     self.__filter_words(x_bop, y)

    #     filtered_xbow = []
    #     for inx, bow in enumerate(X_bow):
    #         splited_bow = np.array(bow.split(' '))
    #         # print(splited_bow)
    #         mask = np.isin(splited_bow, self.selected_words)
    #         filtered_bow = splited_bow[mask]
    #         filtered_xbow.append(filtered_bow)

    #         print(filtered_xbow)
        
    #     self.filtered_xbow = filtered_xbow

    #     return filtered_xbow



    def apply_bow(self, x, y, aggregation_type='sum'):

        t_xbop = []
        t_Xbow = []
        filtered_xbow = []

        # TODO : select aggregation method for multivariable here or maybe in another object lol
        agg = self.__select_agg(aggregation_type)
        
        for ch in range(self.channel_size):
            
            X = x[:, ch]
            X = check_array(X)
            X_bow = self.bow.fit_transform(X)
            # print(X_bow.shape)
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
        print('shape',filtered_xbow.shape)



        

    def __get_inx_cooc_matrix(self, inx, include_diagonal=False):

        # # get words to build the fera graph
        f_bow = self.filtered_xbow[inx]

        # # making the maximum possible matrix to be able to sum later
        max_words = np.unique((self.selected_words.flatten()))
        n = len(max_words) ## the adj will be ->  n x n matrix

        # # creating the adjancey words : this could be usefull in the future to experiment K-connections        
        neighbors = np.empty((len(f_bow) - 1, 2), dtype=f_bow.dtype)
        neighbors[:, 0] = f_bow[:-1]
        neighbors[:, 1] = f_bow[1:]

        ###  vectorized + hash 
        def __transform_word(a, b): 
            n1, n2 = hash(a) % n, hash(b) % n 
            return n1, n2

        v_func = np.vectorize(__transform_word)

        adj_m = np.zeros((n, n))

        edges = v_func(neighbors[:, 0], neighbors[:, 1])
        print(edges)
        edges_list = np.stack((edges[0], edges[1]), axis=-1) 
        pairs, counts = np.unique(edges_list, axis = 0, return_counts=True)

        adj_m[pairs[:, 0], pairs[:, 1]] = counts  # CAN THIS BE DONE WITHOUT CREATING AN ADJ MATRIX ?????

        if not include_diagonal: 
            np.fill_diagonal(adj_m, 0)

        row_sum = np.sum(adj_m, axis=1)
        normalized_adj = adj_m / row_sum[:, np.newaxis]
        normalized_adj = np.nan_to_num(normalized_adj, nan = 0)

        return normalized_adj, neighbors, edges_list

    def get_cooc_matrix(self, x, aggregation_type='sum', include_diagonal=False):

        for ch in range(self.channel_size):
            
            X = x[:, ch]
            n_adj, _, edges_list = self.__get_inx_cooc_matrix(ch, False)
