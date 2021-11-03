import random
import sys
sys.path.append(r'C:\matlabpy\Lib\site-packages')
import numpy as np
import matlab.engine
from scipy.optimize import linprog

eng = matlab.engine.start_matlab()
s = eng.genpath(r'.\matlab')
eng.addpath(s, nargout=0)

def softmax(numbers):
    exponentials = np.exp(numbers)
    sum_exponentials = sum(exponentials)
    return exponentials / sum_exponentials

vertices = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
])

vertices *= 3

vertices += 10

#print(vertices)

num_of_nonvertices = 100

nonvertices = []

for _ in range(num_of_nonvertices):
    genvertices = random.sample(vertices.tolist(), random.randint(2, len(vertices)))
    factors = softmax(np.random.rand(len(genvertices)))
    nonvertex = np.sum([factors[i] * np.array(genvertices[i]) for i in range(len(factors))], axis=0)
    nonvertices.append(nonvertex)

points = np.append(nonvertices, vertices, axis=0)
points = points.tolist()
np.random.shuffle(points)
matlab_points = matlab.double(points, (len(points), len(points[0])))

#print(np.array(points))

indices = [int(i) for i in eng.AVTA_eps(matlab_points, 0.00001)[0]]
#indices = [int(i) for i in eng.AVTA_K(matlab_points, 10)[0]]
indices = [i - 1 for i in indices]
#print(indices)
basic_points = np.array([np.array(matlab_points[i]) for i in indices])
print('The convex hull is:')
print(basic_points)
#print(len(basic_points))

def get_weights(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    # print('*************************************************************************************************************')
    # print(A)
    # print('*************************************************************************************************************')
    # print(b)
    # print('*************************************************************************************************************')
    # print(c)
    # print('*************************************************************************************************************')
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.x

query = nonvertices[0]
weights = get_weights(basic_points, query)
print('Factorizing:')
print(query)
print(f'Weights sum to {sum(weights)}')
print(f'{query} = {" + ".join([str(round(weights[i], 3)) + " * " + str(basic_points[i]) for i in range(len(weights)) if weights[i] != 0.0])}\n')
print(np.sum([weights[i] * basic_points[i] for i in range(len(weights))], axis=0))