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
        K=None, 
        n_neighbors=1,
    ):
        
        self.bow = pyts.bag_of_words.BagOfWords(window_size, word_size, n_bins)
        self.dict_words = np.apply_along_axis("".join, axis=1,arr=list(itertools.product(*([[i for i in string.ascii_letters[:n_bins]]] * word_size))))
        self.K = K
        self.n_neighbors = n_neighbors
        self.is_fitted = False
    
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
        self.word_to_number = {self.selected_words[i] : i for i in range(len(self.selected_words))}
        self.number_to_word = {i : self.selected_words[i]  for i in range(len(self.selected_words))}

        return x_bop[:, selected_fchi2]
   
   
    def __np_to_number(self, a, b):
        return self.word_to_number[a], self.word_to_number[b]

 
    def fit(self, X, y):

        # discretize the TS into list of words
        x_bow = self.bow.fit_transform(X,y)
        
        # transforms the words in a histogram
        x_bop = np.array([ self.__get_count_from_bow(xb) for xb in x_bow])

        # filter words using chi2_square test (decided to let this feature be optional)  
        filtered_xbop = x_bop
        if self.K:
            filtered_xbop = self.__filter_words(x_bop, y, self.K)

        # split and filter the words in the x_bow 

        print(x_bow)
        split_xbow = [np.array(w.split(' ')) for w in x_bow]
        filtered_xbow = [word[np.isin(word, self.selected_words)] for word in split_xbow]

        self.is_fitted = True

        return filtered_xbow

    def transform(self, filtered_xbow):

        if not self.is_fitted:
            raise("Model is not fitted") 

        graphs_list = []
        
        # builds an array with [i - n_neighbors, i + n_neighbors] for all i words   
        for f_w in filtered_xbow:
            print(f_w)
            neighbors = np.empty((len(f_w) - self.n_neighbors, 2), dtype=f_w.dtype)
            neighbors[:, 0] = np.roll(f_w, -self.n_neighbors)[:-self.n_neighbors]
            neighbors[:, 1] = np.roll(f_w, self.n_neighbors)[self.n_neighbors:]

            v_func = np.vectorize(self.__np_to_number)
            edges = v_func(neighbors[:,0], neighbors[:,1])
            edge_list = np.stack((edges[0], edges[1]), axis=-1) 
            edge_with_weights = np.column_stack((np.unique(edge_list, axis = 0, return_counts=True)))
            graphs_list.append(edge_with_weights)
            
        return graphs_list

    def fit_transform(self, X, y):

        return self.transform(self.fit(X,y))