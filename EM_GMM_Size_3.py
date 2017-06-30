
''' Original implementation of the code was done by McDickenson available here - https://github.com/mcdickenson/em-gaussian
considering two gaussian mixture model. This code modified the original work by
extending to three gaussian mixture and shows a way of how to use the same code for n number of 
gaussian mixtures '''


import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from sys import maxint


### Setup
# set random seed
rand.seed(24)


'''sigma is actually covariance, which is either spherical or diagonal type 

Covariance indicates the level to which two variables vary together
From the multivariate normal distribution, we draw N-dimensional samples, 
X = [x_1, x_2, ... x_N]. The covariance matrix element C_{ij} is the covariance 
of x_i and x_j.           
The element C_{ii} is the variance of x_i (i.e. its “spread”).


Diagonal covariance (cov has non-negative elements, and only on the diagonal)
Diagonal covariance means that points are oriented along x or y-axis:
Note that the covariance matrix must be positive semidefinite    
'''

# Considering three clusters 
#initializing the parameters for choosing the samples from gaussians
mu1 = [0, -2]
sig1 = [ [2, 0], [0, 2] ]
mu2 = [1, -3]
sig2 = [ [2, 0], [0, 1] ]
mu3 = [3, 2]
sig3 = [ [1, 0], [0, 2] ]

# generate samples
x1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T
x3, y3 = np.random.multivariate_normal(mu3, sig3, 100).T                                                                           


''' the first 100 data belongs to one gaussian distribution, and the next 100 
samples are taken from another gauss oistri, now joining them to form a common
distri for us to cluster it '''
                                         
xs = np.concatenate((x1, x2, x3))
ys = np.concatenate((y1, y2, y3))
labels = ([1] * 100) + ([2] * 100) +([3] * 100)

data = {'x': xs, 'y': ys, 'label': labels}
df = pd.DataFrame(data=data)

# inspect the data
df.head()
df.tail()

fig = plt.figure()
plt.scatter(data['x'], data['y'], 17, c=data['label'])
fig.savefig("true-values.png")

### Expectation-maximization

# initial guesses for the gaussian mixture, parameters of which have to be tuned later
guess = { 'mu1': [1,1],
          'sig1': [ [1, 0], [0, 1] ],
          'mu2': [4,4],
          'sig2': [ [1, 0], [0, 1] ],
          'mu3': [3,3],
          'sig3': [ [1, 0], [0, 1] ],
          'lambda': [0.3, 0.4, 0.3]
        }


#print lambda[0]

#  lambda is the probablility that the point comes from that particular gaussian
# note that the covariance must be diagonal for this to work

# Probability of data point Val belongingg to a cluster 
def prob(val, mu, sig, lam):
  p = lam
  for i in range(len(val)-1):
    p *= norm.pdf(val[i], mu[i], sig[i][i])
  return p


# Expectation step - checking to which cluster the data point is expected to be came from given the initial parameter setting
def expectation(dataFrame, parameters):
  for i in range(dataFrame.shape[0]):
    x = dataFrame['x'][i]
    y = dataFrame['y'][i]
    p_cluster1 = prob([x, y], list(parameters['mu1']), list(parameters['sig1']), parameters['lambda'][0] )
    p_cluster2 = prob([x, y], list(parameters['mu2']), list(parameters['sig2']), parameters['lambda'][1] )
    p_cluster3 = prob([x, y], list(parameters['mu3']), list(parameters['sig3']), parameters['lambda'][2] )
    
    if (p_cluster1 >= p_cluster2) & (p_cluster1 >= p_cluster3):
      dataFrame['label'][i] = 1
    elif  (p_cluster2 >= p_cluster1) & (p_cluster2 >= p_cluster3):
      dataFrame['label'][i] = 2
    elif  (p_cluster3 >= p_cluster1) & (p_cluster3 >= p_cluster2):
        dataFrame['label'][i] = 3
    else: dataFrame['label'][i] = 3
  return dataFrame


# Maximization step - Given the parameters and the model, whther the parameter maximizes the likelihood of being sampled correctly from the gaussian distribution 
# Alternatively this step finds the parameters that maximizes/suits the given setting
def maximization(dataFrame, parameters):
  #print 'mu1 parameters printing', '\n' 
  #print parameters['mu1']
  #print '\n'
  #print parameters['sig1']
  points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
  points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]
  points_assigned_to_cluster3 = dataFrame[dataFrame['label'] == 3]
  #print points_assigned_to_cluster3
  percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
  percent_assigned_to_cluster2 = len(points_assigned_to_cluster2) / float(len(dataFrame))
  percent_assigned_to_cluster3 = 1 - percent_assigned_to_cluster1 - percent_assigned_to_cluster2
  parameters['lambda'] = [percent_assigned_to_cluster1, percent_assigned_to_cluster2, percent_assigned_to_cluster3 ]
  parameters['mu1'] = [points_assigned_to_cluster1['x'].mean(), points_assigned_to_cluster1['y'].mean(), None]
  parameters['mu2'] = [points_assigned_to_cluster2['x'].mean(), points_assigned_to_cluster2['y'].mean(),None]
  parameters['mu3'] = [points_assigned_to_cluster3['x'].mean(), points_assigned_to_cluster3['y'].mean(), None]
  parameters['sig1'] = [ [points_assigned_to_cluster1['x'].std(), 0 ], [ 0, points_assigned_to_cluster1['y'].std() ], None]
  parameters['sig2'] = [ [points_assigned_to_cluster2['x'].std(), 0 ], [ 0, points_assigned_to_cluster2['y'].std() ], None]
  parameters['sig3'] = [ [points_assigned_to_cluster3['x'].std(), 0 ], [ 0, points_assigned_to_cluster3['y'].std() ], None]
  return parameters

# get the distance between points
# used for determining if params have converged or not
def distance(old_params, new_params):
  dist = 0
  for param in ['mu1', 'mu2', 'mu3']:
    for i in range(len(old_params)-1):
      dist += (old_params[param][i] - new_params[param][i]) ** 2
  return dist ** 0.5

# loop until parameters converges
shift = maxint
epsilon = 0.07
iters = 0
df_copy = df.copy()
# randomly assign points to their initial clusters
df_copy['label'] = map(lambda x: x+1, np.random.choice(3, len(df)))

#params = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in guess.iteritems()])) 
params = pd.DataFrame.from_dict(guess, orient = 'index')
params = params.transpose()
#params = pd.DataFrame([guess])

#print params

#print shape(params)

while shift > epsilon:
  iters += 1
  # E-step
  updated_labels = expectation(df_copy.copy(), params)
  #print updated_labels

  # M-step
  updated_parameters = maximization(updated_labels, params.copy())

  # see if our estimates of mu have changed
  # could incorporate all params, or overall log-likelihood
  shift = distance(params, updated_parameters)

  # Printing the mean shift output
  print("iteration_new {}, shift_new {}".format(iters, shift))

  # update labels and params for the next iteration
  df_copy = updated_labels
  params = updated_parameters

  fig = plt.figure()
  plt.scatter(df_copy['x'], df_copy['y'], 24, c=df_copy['label'])
  fig.savefig("iteration_new{}.png".format(iters))
