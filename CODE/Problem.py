# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:09:07 2015

@author: emmanuel
"""

import os
import numpy
import random
from math import sqrt
from Utils import create
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import os,sys,math,copy


class Problem():

        def __init__(self,bench,params):
		'''
		Constructor in charge of loading data
		'''
                self.bench=bench
		self.population = []
		self.params=params
                self.ind_function=None
                self.low=[self.bench.bounds[i][0] for i in range(self.bench.nvariables())]
                self.up=[self.bench.bounds[i][1] for i in range(self.bench.nvariables())]

	def __getitem__(self,name):
		'''
		Get access to the parameters dictionnary
		'''
		if self.params.has_key(name):
			return self.params[name]
		raise KeyError


	def __setitem__(self,name,value):
		'''
		Set the parameters dictionnary
		'''
		self.params[name]=value
 
        def init_ind(self):
        	'''
    	        Generate a single individual
        	'''
                assert(self.ind_function is not None)
                vect=[]
                for i in range(self.bench.nvariables()):
                    vect.append(numpy.random.uniform(self.bench.bounds[i][0],self.bench.bounds[i][1]))
            	return self.ind_function(vect)



        def init_population(self):
            '''
            Random initialization of the population
            '''
            for k in range(self.params["POP"]):
		self.population.append(self.init_ind())
            return self.population

           

        def evaluateOBJ(self,individuals):
            '''
            Evaluation of the objective function
            '''
            raise NotImplementedError("Abstract method") 

	def mate(self,ind1,ind2):
		'''
		Crossover operator
		'''
		return tools.crossover.cxSimulatedBinaryBounded(ind1, ind2, self.params["eta"], self.low, self.up)

	def mutate(self,individual,indpb):
		'''
		Mutation operator
		'''
		return tools.mutation.mutPolynomialBounded(individual,self.params["eta"],self.low,self.up,indpb) 


	def select(self,pop,size,tournsize):
		'''
		Tournament selection operator
		'''
                return tools.selTournament(pop,size,tournsize)

	def test_cstr(self,individual,indice):
		'''
		Verify if an individual respect the constraint with index "indice"
		Return 0 if ok else the distance to the constraint
		'''
                self.bench.unset_var()
                self.bench.set_var(individual)
                return self.bench.violation_constraint(indice)	

	def best(self,ind1,ind2):
		'''
		Return the best individual -- sense == minimization
		'''
		if ind1.fitness.values[0] <= ind2.fitness.values[0]:
			return ind1
		else:
			return ind2



	def findBestFeasible(self,population):
            '''
            Find the best feasible solution in a SINGLE population (basically a sub-population)
            '''
	    sorted_best=sorted(population,key=lambda ind: ind.fitness.values[0])
	    for i in range(len(sorted_best)):
		individual = sorted_best[i]
		if individual.feasible:
			return individual
	    return None

        def findWorstFeasible(self,population):
            '''
            Find the best feasible solution in a SINGLE population (basically a sub-population)
            '''
	    sorted_best=sorted(population,key=lambda ind: ind.fitness.values[0],reverse=True)
	    for i in range(len(sorted_best)):
		individual = sorted_best[i]
		if individual.feasible:
			return individual
	    return None



	def findAllFeasible(self,population):
            '''
            Find the best feasible solution in a SINGLE population (basically a sub-population)
            '''
	    feasible=[]
	    for i in range(len(population)):
		if population[i].feasible:
			feasible.append(population[i])
	    return feasible


