import sys

sys.path.append(r'C:\matlabpy\Lib\site-packages')
import numpy as np
import matlab.engine
from gensim.models.callbacks import CallbackAny2Vec
from datetime import datetime
from scipy.optimize import linprog
import gensim.downloader as api

base_len = 10

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
# df = pd.read_csv('./model/depDocNNSE50.tab', delimiter='\s+')
# df['vector'] = df[df.columns.drop('#target')].agg(np.array, axis=1)
# print(df[['#target', 'vector']])
# exit(0)
# model = namedtuple('_', 'vectors')()

print(f'The vocabulary contains {model.vectors.shape[0]} word embeddings of length {model.vectors.shape[1]}')

lst = model.vectors.tolist()
print(f'{len(lst)}, {len(lst[0])}')
mat = matlab.double(lst, (len(lst), len(lst[0])))

print(f'{datetime.now().strftime("%H:%M:%S")}: Started computing the convex hull')
indices = [int(i) for i in eng.AVTA_K(mat, base_len)[0]]
print(f'{datetime.now().strftime("%H:%M:%S")}: Finished computing the convex hull')

#### SUPER IMPORTANT - the matlab indices are 1 based and therefore in python they must
#### be reduced by 1
indices = [i - 1 for i in indices]

basic_vectors = np.array([model.vectors[i] for i in indices])
basic_words = [model.index_to_key[i] for i in indices]

def get_weights(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.x


query = 'king'
weights = get_weights(basic_vectors, model[query])
print(f'Weights sum to {sum(weights)}')
print(f'{query} = {" + ".join([str(round(weights[i], 3)) + " * " + basic_words[i] for i in range(len(weights)) if weights[i] != 0])}\n')

for w in basic_words:
    print(w)