import sys
sys.path.append(r'C:\matlabpy\Lib\site-packages')
import numpy as np
import matlab
import matlab.engine
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from datetime import datetime
from scipy.optimize import linprog
import random
import gensim.downloader as api

base_len = 1000

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self):
        self.epoch = 0

    # def on_epoch_begin(self, model):
    #     self.epoch += 1

    # def on_epoch_end(self, model):
    #     output_path = model_path.format('epoch_{}'.format(self.epoch))
    #     model.save(output_path)
    #     self.epoch

eng = matlab.engine.start_matlab()
s = eng.genpath(r'.\matlab')
eng.addpath(s, nargout=0)

# MODEL_PATH = 'C:/Users/mstek/Google Drive/models/word2vec/SemEval/All/semeval_2010-task_14-window_2-normalized.model'
# model = Word2Vec.load(MODEL_PATH)

model = api.load('glove-wiki-gigaword-100')

print(f'The vocabulary contains {model.vectors.shape[0]} word embeddings of length {model.vectors.shape[1]}')

lst = model.vectors.tolist()
print(f'{len(lst)}, {len(lst[0])}')
mat = matlab.double(lst, (len(lst), len(lst[0])))

print(f'{datetime.now().strftime("%H:%M:%S")}: Started computing the convex hull')
indices = [int(i) for i in eng.AVTA_K(mat, base_len)[0]]

print(f'{datetime.now().strftime("%H:%M:%S")}: Finished computing the convex hull')
print(indices)

#### SUPER IMPORTANT - the matlab indices are 1 based and therefore in python they must
#### be reduced by 1
basic_vectors = np.array([np.array(mat[i - 1]) for i in indices])
basic_words = [model.similar_by_vector(v, topn=1)[0][0] for v in basic_vectors]


def get_weights(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    print('*************************************************************************************************************')
    print(A)
    print('*************************************************************************************************************')
    print(b)
    print('*************************************************************************************************************')
    print(c)
    print('*************************************************************************************************************')
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.x


query = 'king'
weights = get_weights(basic_vectors, model[query])
print(f'Weights sum to {sum(weights)}')
print(f'{query} = {" + ".join([str(round(weights[i], 3)) + " * " + basic_words[i] for i in range(len(weights)) if weights[i] != 0])}\n')

for w in basic_words:
    print(w)