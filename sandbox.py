import random
import sys
sys.path.append(r'C:\matlabpy\Lib\site-packages')
import numpy as np
import matlab
import matlab.engine
from matplotlib import pyplot as plt

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

print(vertices)

num_of_nonvertices = 100

nonvertices = []

for _ in range(num_of_nonvertices):
    genvertices = random.sample(vertices.tolist(), random.randint(2, len(vertices)))
    factors = softmax(np.random.rand(len(genvertices)))
    print(factors)
    print(sum(factors))
    nonvertex = np.sum([factors[i] * np.array(genvertices[i]) for i in range(len(factors))], axis=0)
    # nonvertex2str = [f'{factors[i]} * {np.array(genvertices[i])} = {factors[i] * np.array(genvertices[i])}' for i in range(len(factors))]
    # print(nonvertex2str)
    nonvertices.append(nonvertex)
points = np.append(nonvertices, vertices, axis=0)
points = points.tolist()
np.random.shuffle(points)
matlab_points = matlab.double(points, (len(points), len(points[0])))

print(np.array(points))

indices = [int(i) for i in eng.AVTA_eps(matlab_points, 0.00001)[0]]
print(indices)
basic_points = np.array([np.array(matlab_points[i - 1]) for i in indices])

print(basic_points)
print(len(basic_points))

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# data = np.random.random(size=(3, 3, 3))
# z, x, y = np.array(points)
# ax.scatter(x, y, z, c=z, alpha=1)
# plt.show()
