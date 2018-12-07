'''Libraries for Prototype selection'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
from cvxpy import *
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances
import time
import pdb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class classifier():
	'''Contains functions for prototype selection'''
	def __init__(self, X, y, epsilon_, lambda_ ):
		'''Store data points as unique indexes, and initialize 
		the required member variables eg. epsilon, lambda, 
		interpoint distances, points in neighborhood'''
		self.data = X
		self.target = y
		self.epsilon_ = epsilon_
		self.lambda_ = lambda_
		self.num_of_cls = max(self.target)+1

		# the number of data within a class
		self.data_per_cls = []
		self.cls_idx = []
		for i in range(self.num_of_cls):
			self.data_per_cls.append(sum([1 for j in range(len(self.data)) if self.target[j] == i]))
			self.cls_idx.append([j for j in range(len(self.data)) if self.target[j] == i])

		self.pairwise_dist = euclidean_distances(self.data, self.data)
		# initialize pointwise neighbor
		self.neighbor = []
		for i in range(len(self.data)):
			self.neighbor.append([j for j in range(len(self.data)) if self.pairwise_dist[i][j]<=self.epsilon_])
		
		self.classwise_obj = [0]*self.num_of_cls
		self.rounded_obj = [0]*self.num_of_cls

		# initialize classwise matrices
		self.C = [np.zeros(len(self.data)) for i in range(self.num_of_cls)]
		for i in range(len(self.C)):
			for j in range(len(self.C[i])):
				self.C[i][j] = sum([1 for N in self.neighbor[j] if self.target[N] != i])+self.lambda_

		#pdb.set_trace()
		self.M = [np.zeros((self.data_per_cls[i], len(self.data))) for i in range(self.num_of_cls)]
		for i in range(self.num_of_cls):
			for j in range(len(self.cls_idx[i])):
				for N in self.neighbor[self.cls_idx[i][j]]:
					self.M[i][j][N] = 1

		self.opt_alph = []
		self.opt_E = []

		self.rounded_alph = []
		self.rounded_E = []

		self.prototype = []


	'''Implement modules which will be useful in the train_lp() function
	for example
	1) operations such as intersection, union etc of sets of datapoints
	2) check for feasibility for the randomized algorithm approach
	3) compute the objective value with current prototype set
	4) fill in the train_lp() module which will make 
	use of the above modules and member variables defined by you
	5) any other module that you deem fit and useful for the purpose.'''

	
	def train_lp(self, verbose = False):
		'''Implement the linear programming formulation 
		and solve using cvxpy for prototype selection'''
		start = time.time()
		n = len(self.data)
		for i in range(self.num_of_cls):
			Alph = Variable(n)
			E = Variable(self.data_per_cls[i])
			ones = np.ones(self.data_per_cls[i])

			objective = Minimize(self.C[i]*Alph + ones*E)
			constraints = [self.M[i]*Alph>=1-E, Alph>=0, Alph<=1, E>=0]
			#constraints = [self.M[i]*Alph>=1-E, Alph>=0, Alph<=1, E>=0]
			prob = Problem(objective, constraints)
			self.classwise_obj[i] = prob.solve()
			if verbose:
				print('The optimal strategy: {} \r\n\r\n {}'.format(Alph.value, E.value))
				print('Problem status: {} Optimal value: {:.3f}'.format(prob.status, prob.value))
			self.opt_alph.append(Alph.value)
			self.opt_E.append(E.value)
			#pdb.set_trace()
		end = time.time()
		print('Time for solving LP: {:.3f} sec'.format(end-start))

	def is_feasible(self, Alph, E, i):
		ones = np.ones((self.data_per_cls[i],1))
		result = np.dot(self.M[i],Alph)-ones+E
		#pdb.set_trace()
		for res in result:
			if res < 0:
				return False
		return True

	def objective_value(self):
		'''Implement a function to compute the objective value of the integer optimization
		problem after the training phase'''
		start = time.time()
		num_of_trial = 500

		for i in range(self.num_of_cls):
			for l in range(num_of_trial):
				for j in range(int(2*mt.log(self.data_per_cls[i]))):
					temp_alph, temp_e = np.zeros((len(self.data), 1)), np.zeros((self.data_per_cls[i], 1))
					for k in range(len(temp_alph)):
						#print(self.opt_alph[i][k])
						temp_alph[k] = max(temp_alph[k], np.random.binomial(1, max(0, min(1, self.opt_alph[i][k]))))

					for k in range(len(temp_e)):
						temp_e[k] = max(temp_e[k], np.random.binomial(1, max(0, min(1, self.opt_E[i][k]))))

				if self.is_feasible(temp_alph, temp_e, i) and \
						np.dot(self.C[i], temp_alph)+sum(temp_e) <= 2*mt.log(self.data_per_cls[i])*self.classwise_obj[i]:
					
					self.rounded_alph.append(temp_alph)
					self.rounded_E.append(temp_e)
					self.rounded_obj[i] = np.dot(self.C[i], temp_alph)+sum(temp_e)

					for p in range(len(temp_alph)):
						if temp_alph[p] == 1:
							self.prototype.append((self.data[p], i, p))
							#pdb.set_trace()
					break


				if l == num_of_trial-1:
					print('No rounded solutions found!!!')
					pdb.set_trace()
		end = time.time()
		print('Time for rounding: {:.3f} sec'.format(end-start))
		#pdb.set_trace()
		return sum(self.rounded_obj)[0]

	def predict(self, instances, cover_error=False):
		'''Predicts the label for an array of instances using the framework learnt'''
		def distance(i, j):
			return (sum([(i[k]-j[k])**2 for k in range(len(i))]))**0.5

		target = []
		error = 0.0
		N = len(self.prototype)
		for i in instances:
			#target.append(sorted([(distance(i, self.prototype[j][0]), self.prototype[j][1]) for j in range(N)])[0][1])
			dist = [distance(i, self.prototype[j][0]) for j in range(N)]
			if cover_error and min(dist) > self.epsilon_:
				#target.append(-1)
				target.append(self.prototype[dist.index(min(dist))][1])
				error += 1
			else:
				if not dist:
					pdb.set_trace()
				target.append(self.prototype[dist.index(min(dist))][1])
		
		error = error/len(instances)
		#pdb.set_trace()
		return target if not cover_error else (target, error)


def cross_val(data, target, epsilon_, lambda_, k, verbose, cover_error=False):
	'''Implement a function which will perform k fold cross validation 
	for the given epsilon and lambda and returns the average test error and number of prototypes'''
	kf = KFold(n_splits=k, random_state = 42, shuffle=True)
	score = 0.0
	prots = 0.0
	obj_val = 0.0
	c_error = 0.0
	for train_index, test_index in kf.split(data):
		ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
		ps.train_lp(verbose)
		
		obj_val += ps.objective_value()
		#pdb.set_trace()
		if cover_error:
			pred, error = ps.predict(data[test_index], cover_error=True)
			c_error += error
			score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
		else:
			pred = ps.predict(data[test_index], cover_error=False)
			score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
		prots += len(ps.prototype)
		#pdb.set_trace()
		'''implement code to count the total number of prototypes learnt and store it in prots'''
	
	c_error /= k
	score /= k    
	prots /= k
	obj_val /= k
	

	return (score, prots, obj_val) if not cover_error else (score, prots, obj_val, c_error)

def visualize_iris():

	iris = load_iris()
	epsilon_ = 1
	lambda_ = 1/len(iris.data)
	ps = classifier(iris.data, iris.target, epsilon_, lambda_)
	ps.train_lp()
	ps.objective_value()

	x, y = [], []

	j,k = 2,3
	x.append([iris.data[i][j] for i in range(len(iris.target)) if iris.target[i] == 0])
	x.append([iris.data[i][j] for i in range(len(iris.target)) if iris.target[i] == 1])
	x.append([iris.data[i][j] for i in range(len(iris.target)) if iris.target[i] == 2])

	y.append([iris.data[i][k] for i in range(len(iris.target)) if iris.target[i] == 0])
	y.append([iris.data[i][k] for i in range(len(iris.target)) if iris.target[i] == 1])
	y.append([iris.data[i][k] for i in range(len(iris.target)) if iris.target[i] == 2])	

	plt.figure()
	plt.plot(x[0], y[0], 'b.', label='data type 0')
	plt.plot(x[1], y[1], 'r.', label='data type 1')
	plt.plot(x[2], y[2], 'g.', label='data type 2')

	for d in ps.prototype:
		#pdb.set_trace()
		if d[1] == 0:
			plt.plot([d[0][j]], [d[0][k]], 'b^')
		elif d[1] == 1:
			plt.plot([d[0][j]], [d[0][k]], 'r^')
		else:
			plt.plot([d[0][j]], [d[0][k]], 'g^')

	plt.legend()
	plt.grid()
	plt.show()

def breast_cancer():
	dataset = load_breast_cancer()
	cover_error = True
	verbose = False

	lambda_ = 1/len(dataset.data)	
	distance = euclidean_distances(dataset.data, dataset.data)

	pairwise_dist = []
	for i in range(len(distance)):
		for j in range(len(distance[0])):
			if dataset.target[i] != dataset.target[j]:
				#pdb.set_trace()
				pairwise_dist.append(np.linalg.norm(dataset.data[i]-dataset.data[j]))

	t_error, c_error = [],[]
	step = 5
	for i in range(2, 41, step):
		if i == 2:
			epsilon_ = 1
		else:
			epsilon_ = np.percentile(pairwise_dist, i)
		#pdb.set_trace()
		print('epsilon is: {:.4f}'.format(epsilon_))
		score, prots, obj_val, cvr_error = cross_val(dataset.data, dataset.target, epsilon_, lambda_, 4, verbose, cover_error)
		print('score: {:.4f}, cover error: {:.4f}, number of prototype: {:.4f}, objective value: {:.4f}'.\
			format(score, cvr_error, prots, obj_val))	

		t_error.append(1-score)
		c_error.append(cvr_error)

	plt.figure()
	plt.plot([i for i in range(2,41,step)], t_error, 'b-', label='test error')
	plt.plot([i for i in range(2,41,step)], c_error, 'g-', label='cover error')
	plt.ylabel('error rate')
	plt.xlabel('percentile in distance')
	plt.title("error vs epsilon")
	plt.grid()
	plt.legend()
	plt.show()

def digits():
	dataset = load_digits()
	cover_error = False
	verbose = False

	lambda_ = 1/len(dataset.data)	

	obj_list = []

	eps_list = [1, 35, 38, 40, 42, 45, 46, 47, 48]
	percentile = [0.04, 7, 10, 15, 20, 30, 35, 40, 48]
	
	# pairwise_dist = euclidean_distances(dataset.data, dataset.data)
	# distance = np.percentile(pairwise_dist, 10)
	# pdb.set_trace()

	for epsilon_ in eps_list:
		#pdb.set_trace()
		print('epsilon is: {:.4f}'.format(epsilon_))
		score, prots, obj_val  = cross_val(dataset.data, dataset.target, epsilon_, lambda_, 4, verbose, cover_error)
		print('score: {:.4f}, number of prototype: {:.4f}, objective value: {:.4f}'.\
			format(score, prots, obj_val))	

		obj_list.append(obj_val)

	plt.figure()
	plt.plot(percentile, obj_list, 'b-', label='objective value')
	plt.ylabel('integer objective value')
	plt.xlabel('percentile in distance')
	plt.title("objective value vs percentile in epsilon")
	plt.grid()
	plt.legend()
	plt.show()

if __name__ == '__main__':
	#breast_cancer()
	#pdb.set_trace()
	dataset = load_iris()
	#dataset = load_breast_cancer()
	#dataset = load_digits()

	cover_error = False
	verbose = False

	epsilon_ = 1
	lambda_ = 1/len(dataset.data)
	
	# ps = classifier(iris.data, iris.target, epsilon_, lambda_)
	# ps.train_lp()
	# ps.objective_value()
	start = time.time()
	if cover_error:
		score, prots, obj_val, c_error = cross_val(dataset.data, dataset.target, epsilon_, lambda_, 5, verbose, cover_error)
		print('score: {:.4f}, cover error: {:.4f}, number of prototype: {:.4f}, objective value: {:.4f}'.\
			format(score, c_error, prots, obj_val))		
	else:
		score, prots, obj_val = cross_val(dataset.data, dataset.target, epsilon_, lambda_, 4, verbose, cover_error)
		print('score: {:.4f}, number of prototype: {:.4f}, objective value: {:.4f}'.format(score, prots, obj_val))
	end = time.time()
	print('Time for training and testing: {:.3f} sec'.format(end-start))
	pdb.set_trace()