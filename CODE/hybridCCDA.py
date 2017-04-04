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
from benchmarks import *
from Problem import Problem
from SLSQP import * 
import os,sys,math,copy
import time
import copy
import gzip


class Fitness(object):
    '''
    Fitness function class
    Evaluate objecitve using a lexicographic order
    '''
    def __init__(self):

        self.values=[]

    def lower_or_equal(self,other_values):

        for i in range(len(self.values)):
            if self.values[i] < other_values.values[i]:
                return True
            elif self.values[i] > other_values.values[i]:
                return False
            else:
                continue

    def __leq__(self,other_values):
        return self.lower_or_equal(other_values)

    def __lt__(self,other_values):
        return self.lower_or_equal(other_values)

    def __gt__(self,other_values):
        return not self.__lt__(other_values)

    def __geq__(self,other_values):
        return not self.__leq__(other_values)


class Algorithm(Problem):
      '''
      Hybrid CCDA algorithm 
      Same mechanism as CCDA execpt that we generate an additional population 
      where individuals are evaluated using the SLSQP local search
      '''

      def __init__(self,path,params):
        Problem.__init__(self,path,params)
        self.toolbox = base.Toolbox()
        fitness_class = create("FitnessMin", Fitness)
        self.ind_function = create("Individual", list, fitness=fitness_class,feasible=False,repartition=[])
        self.selection={"best":self.selBest,"worst":self.selWorst,"random":self.selRandom,"tournamentBestK":self.tournamentBestK}
        self.targetPop={"BS":self.selectRouletteBS,"WS":self.selectRouletteWS,"link":self.selectRouletteLink,"random":self.targetRandom}


      def init_population(self):
         '''
         Generate initial population (1 cstr <=> 1 species)
         '''
         for i in range(self.bench.nconstraints()+1):
                 self.population.append([])
         for k in range(self.params["POP"]):
             ind = self.init_ind()
             self.population[k % (len(self.population))].append(ind)
         return self.population




      def tournament(self,species,N):
	'''
	Tournament for constraint handling problem
	'''
	vector=[]
	for i in range(N):
		ind1,ind2=tools.selRandom(species,2)
                if ind1.fitness < ind2.fitness:
                    vector.append(ind1)
                else:
                    vector.append(ind2)
	return vector


      def selBest(self,pop,size):
          '''
          Select the best "size" individuals in a sub-population
          '''
          s_pop=sorted(pop,key=lambda ind: ind.fitness)
          return s_pop[:size]


      def selWorst(self,pop,size):
          '''
          Select the worst "size" individuals in a sub-population
          '''
          s_pop=sorted(pop,key=lambda ind: ind.fitness,reverse=True)
          return s_pop[:size]

      def selRandom(self,pop,size):
          '''
          Select randomly size individuals in a sub-population
          '''
          return [pop[i] for i in numpy.random.randint(0,len(pop),size)]


      def tournamentBestK(self,pop,size):
          '''
          Select randomly size individuals in a sub-population
          using a tournament selection of the best K individuals
          '''
          selected = self.selBest(pop,int(self.params["K"]))
          tourn=[]
          for i in range(size):
            A,B = self.selRandom(pop,2)
            if A.fitness < B.fitness:
                  tourn.append(A)
            elif A.fitness > B.fitness:
                  tourn.append(B)
            else:
                  tourn.append(self.selRandom([A,B],1)[0])
          return tourn


      def selectRouletteWS(self,pop,index):
          '''
          Roulette wheel selection among the worst species (or sub-populations)
          '''
          res=[]
          i=index+1
          while (i%len(pop))!=index:
                sp = pop[(i%(len(pop)-1))]
                total_feasible=0
                for ind in sp:
                    val = ind.repartition[index]
                    if val == 0:
                        total_feasible += 0
                    else:
                        total_feasible += (1.0/val)
                res.append(total_feasible)
                i+=1
          if sum(res) == 0:
              return  numpy.random.randint(0,len(pop)-1)
          res = numpy.array(res)
          cumsum = numpy.cumsum(res)
          proba = cumsum/(sum(res)*1.0)
          u = random.random()
          for i in range(len(res)):
              if u <= proba[i]:
                  return (index+i+1)%(len(pop)-1)

      def selectRouletteBS(self,pop,index):
          '''
          Roulette wheel selection among the best species (or sub-populations)
          '''
          res=[]
          i=index+1
          while (i%len(pop))!=index:
                sp = pop[(i%(len(pop)-1))]
                total_feasible=0
                for ind in sp:
                    total_feasible += ind.repartition[index]
                res.append(total_feasible)
                i+=1
          if sum(res) == 0:
              return  numpy.random.randint(0,len(pop)-1)
          res = numpy.array(res)
          cumsum = numpy.cumsum(res)
          proba = cumsum/(sum(res)*1.0)
          u = random.random()
          for i in range(len(res)):
              if u <= proba[i]:
                  return (index+i+1)%(len(pop)-1)

      def targetRandom(self,pop,index):
          '''
          Select a target sub-population randomly
          '''
          other_index = [i for i in range(len(pop))]
          return random.choice(other_index)


      def selectRouletteLink(self,pop,index):
          '''   
          Target selection using the link rule 
          The selection mechanism implemented hereafter is roulette wheel  
          '''
          res=[]
          i=index+1
          while (i%len(pop))!=index:
                sp = pop[(i%(len(pop)-1))]
                res.append(self.bench.coupling_variables(index,(i%(len(pop)-1))))
                i+=1
          if sum(res) == 0:
              return  numpy.random.randint(0,len(pop)-1)
          res = numpy.array(res)
          cumsum = numpy.cumsum(res)
          proba = cumsum/(sum(res)*1.0)
          u = random.random()
          for i in range(len(res)):
              if u <= proba[i]:
                  return (index+i+1)%(len(pop)-1)

      def getFeasible(self,pop):
          '''
          Return the feasible individuals from a sub-population
          '''
          return [ind for ind in pop if ind.feasible is True]
    
      def compile_stats_feasible(self,pop,stats):
          '''
          Compile some statistics 
          '''
          subpop=self.getFeasible(pop)
          if len(subpop) == 0:
              return {"stdF":None,"minF":None,"maxF":None}
          recordF = stats.compile(subpop) if stats else {}
          # Avoid to have same key
          recordF["stdF"] = recordF.pop("std")
          recordF["minF"] = recordF.pop("min")
          recordF["maxF"] = recordF.pop("max")
          return recordF



      def solve(self,verbose=False):
          '''
          Main method to start the algorithm
          '''
          # Definitions
          start=time.time()
          total_evals=0
          stats = tools.Statistics(lambda ind: ind.fitness.values[-1])
	  stats.register("std", numpy.std)
	  stats.register("min", numpy.min)
	  stats.register("max", numpy.max)
	  statsF = tools.Statistics(lambda ind: int(ind.feasible))
 	  statsF.register("Feasible",sum)
	  self.logbook = tools.Logbook()
	  self.logbook.header = ['nevals'] + (stats.fields if stats else []) + ["stdF","minF","maxF"]+ (statsF.fields if statsF else [])

          #########################################
          # init all subpops and create self.population which is a list of subpopulation
          self.population = self.init_population()
          self.best=None
          # evaluate all individuals in each sub-populations
          for j in range(len(self.population)):
              for ind in self.population[j]:
                  ind.fitness.values,evals = self.evaluate(ind,j)
                  total_evals += evals
                  
          # put all individuals in a list (data) and compute the statistics which
          # are recorded in a logbook
          data=[ind for sp in self.population for ind in sp] 
          record = stats.compile(data) if stats else {}
          record.update(self.compile_stats_feasible(data,stats))
	  record.update(statsF.compile(data) if statsF else {})
	  self.logbook.record(nevals=total_evals, **record)
	  print self.logbook.stream
        
          # verbose mode allows to record all statistics for
          # each sub-populations independently
          if verbose is True:
              pop_record=[]
              self.logbooks=[]
              self.record_gen_pop(pop_record,self.population)

              stats1 = tools.Statistics(lambda ind: ind.fitness.values[-1])
              stats1.register("std", numpy.std)
              stats1.register("min", numpy.min)
              stats1.register("max", numpy.max)
              stats2 = tools.Statistics(lambda ind: int(ind.feasible))
              stats2.register("Feasible",sum)


              record={}
              for i in range(self.bench.nconstraints()):
                  log = tools.Logbook()
                  log.header = ['nevals']  + [field for field in stats1.fields]  + ["stdF","minF","maxF"]  +[field for field in stats2.fields]  
                  record.update(stats1.compile(self.population[i])) 
                  record.update(self.compile_stats_feasible(self.population[i],stats1))
                  record.update(stats2.compile(self.population[i]))
                  log.record(nevals=total_evals,**record)       
                  self.logbooks.append(log)
          k=0
          # Find the best individual
          self.best=sorted([ind for sp in self.population for ind in sp if ind.feasible],key=lambda ind:ind.fitness)[0]
          while total_evals <= self.params["EVALS"]:
                    # Random selection of the target species
                    # if "i" is the last sub-population, select a random target population
                    # else
                    # select a sub-population according to "targetRule"
                    i = k %(len(self.population))
		    if i == len(self.population)-1:
		       j = self.targetRandom(self.population,i)
		    else:
                    	j = self.targetPop[self.params["targetRule"]](self.population,i)
                    # Select the sub-population that we call species
                    species1 = self.population[i]
                    species2 = self.population[j]
                    # Apply the required selection of individuals according to "selection"
                    s1,s2=self.params["selection"].replace("\n","").split("-")
                    parent1= self.selection[s1](species1,1)[0]
                    parent2 = self.selection[s2](species2,1)[0]
                    # Create clones of parents
                    offspring1  = self.toolbox.clone(parent1)
                    offspring2  = self.toolbox.clone(parent2)
                    # Apply crossover
                    if random.random() < self.params["CXPB"]:
                        offspring1,offspring2 = self.mate(offspring1,offspring2)
                    # Apply mutation first individual
                    if random.random() < self.params["MUTPB"]:
                        offspring1, = self.mutate(offspring1,(1.0)/len(offspring1))
                    # Apply mutation first individual
                    if random.random() < self.params["MUTPB"]:
                        offspring2, = self.mutate(offspring2,(1.0)/len(offspring2))
                    # Evaluate the new individuals
                    offspring1.fitness.values,eval1 = self.evaluate(offspring1,i)
                    offspring2.fitness.values,eval2 = self.evaluate(offspring2,j)
                    total_evals += (eval1+eval2)
                    # replace only if better than parents
                    if offspring1.fitness < parent1.fitness:
                            self.replace(parent1,offspring1)
                    if offspring2.fitness < parent2.fitness:
                            self.replace(parent2,offspring2)
                    k+=1
                    # Compute statistics
                    if total_evals%self.params["POP"] ==0 :
                       data=[ind for sp in self.population for ind in sp] 
                       current = sorted([ind for sp in self.population for ind in sp if ind.feasible],key=lambda ind:ind.fitness)[0]
                       if current is not None and current.fitness.values[-1] < self.best.fitness.values[-1]:
                           self.best = current
                       record = stats.compile(data) if stats else {}
                       record.update(self.compile_stats_feasible(data,stats))
		       record.update(statsF.compile(data) if statsF else {})
		       self.logbook.record(nevals=total_evals, **record)
                       print self.logbook.stream
                       if verbose is True:
                         self.record_gen_pop(pop_record,self.population)
                         record={}
                         for i in range(self.bench.nconstraints()):
                             log = self.logbooks[i] 
                             record.update(stats1.compile(self.population[i])) 
                             record.update(self.compile_stats_feasible(self.population[i],stats1))
                             record.update(stats2.compile(self.population[i]))
                             log.record(nevals=total_evals,**record)       
	  self.time=time.time()-start
          if verbose is True:              
             return pop_record 


      def mate(self,ind1,ind2):
          '''
          Crossover procedure
          '''
          n1=self.toolbox.clone(ind1)
          n2=self.toolbox.clone(ind2)
          for i in range(len(ind1)):
              n1[i]=min(max(ind1[i]-0.35*(ind2[i]-ind1[i])+(1+2*0.35)*(ind2[i]-ind1[i])*random.random(),self.bench.bounds[i][0]),self.bench.bounds[i][1])
              n2[i]=min(max(ind1[i]-0.35*(ind2[i]-ind1[i])+(1+2*0.35)*(ind2[i]-ind1[i])*random.random(),self.bench.bounds[i][0]),self.bench.bounds[i][1])

          return n1,n2

      def mutate(self,ind,indpb):
          '''
          Mutation procedure
          '''
          new=self.toolbox.clone(ind)
          for i in range(len(ind)):
              if random.random() < indpb:
                  new[i] = min(max(new[i]+numpy.random.normal(0,1),self.bench.bounds[i][0]),self.bench.bounds[i][1])

          return new,
      def record_gen_pop(self,pop_record,pop):
          '''
          Record a generation (means that you record all sub-populations)
          '''
          map_data={}
          for i in range(len(pop)):
              map_data[i]=[]
              for ind in pop[i]:
                  map_data[i].append(ind)
          pop_record.append(map_data)   

      def bestIndividual(self):
          feasible=[ind for sp in self.population for ind in sp if ind.feasible]
          return sorted(feasible,key=lambda ind : ind.fitness.values[-1])[0] 

      def replace(self,parent,offspring):
          for i in range(len(parent)):
              parent[i] = offspring[i]
          parent.fitness.values = offspring.fitness.values
          parent.feasible = offspring.feasible


      def evaluate(self,individual,i):
          if i == len(self.population)-1:
              # Local search
              solver=SLSQP(self.bench)
              if self.best != None:
                solver.addConstraints(self.best.fitness.values[-1]-1e-4)
              res=solver.solve(individual)
              if res.status == 0:
                for l in range(len(res.x)):
                  individual[l]=res.x[l]
              i=0

          self.bench.unset_var()
          self.bench.set_var(individual)
          value = self.bench.violation_constraint(i)
          other=0
          repartition=[]
          nb=0
          for j in range(self.bench.nconstraints()):
                res = self.bench.violation_constraint(j)
                if res > 0:
                   repartition.append(0)
                   nb += 1
                else:
                   repartition.append(1)
                if i!=j:
                    other+=res
                
          if nb==0:
              individual.feasible=True
          else:
              individual.feasible=False
          individual.repartition = repartition
          return [value,other,self.bench.evaluate_objectives(0)],(1.0)
      
      def write_solution(self,path):
          '''
          Record the best solution
          '''
          fd = open(path,"a")
          self.best_ind = self.bestIndividual()
          fd.write(str(self.best_ind.fitness.values[-1])+","+str(self.time)+","+str(int(self.best_ind.feasible))+"\n")
          fd.close()


      def write_logbook(self,path,log):
          '''
          Record the logbook
          '''
	  evals,std,mini,maxi,stdf,minf,maxf,f = log.select("nevals", "std","min","max","stdF","minF","maxF","Feasible")
	  descriptor = open(path,"w")
	  for i in range(len(evals)):
		descriptor.write(str(evals[i])+","+str(std[i])+","+str(mini[i])+","+str(maxi[i])+","+str(stdf[i])+","+str(minf[i])+","+str(maxf[i])+","+str(f[i])+"\n")
          descriptor.close()

      def write_all_pop(self,path,record_pop):
          '''
          Record all the generations
          '''
          descriptor = gzip.open(path,"w")
          for i in range(len(record_pop)):
              for j in range(len(record_pop[i])):
                  for k in range(len(record_pop[i][j])):
                        descriptor.write(str(j)+","+",".join(map(str,record_pop[i][j][k]))+","+",".join(map(str,record_pop[i][j][k].fitness.values))+","+str(record_pop[i][j][k].feasible)+"\n")
              descriptor.write("\n")
          descriptor.close()    
     
if __name__== "__main__":
    all_params = {"folder":".","verbose":0,"POP":100, "CXPB":0.9, "MUTPB":1,"EVALS":50000,"rate":0.01,"eta":20,"selection":"best-worst","K":5,"targetRule":"random"}
    problem = sys.argv[1]
    run = sys.argv[2] 
    for i in range(3,len(sys.argv)):
        if ":" in sys.argv[i]:
            parameter = sys.argv[i]
            name,value=parameter.split(":")
            all_params[name] = value.replace("\n","")
    print(all_params)
    P=globals()[problem]()
    prob = Algorithm(P, all_params)
    pop_record = prob.solve(bool(int(all_params["verbose"])))
    folder=all_params["folder"]
    prob.write_solution(folder+"/EQ"+"_"+problem+".ga.res")
    prob.write_logbook(folder+"/EQ"+"_"+problem+"_"+str(run)+".ga.log",prob.logbook)
    if bool(int(all_params["verbose"])) is True:
        prob.write_all_pop(folder+"/EQ"+"_"+problem+"_"+str(run)+".ga.pop",pop_record)
        for i in range(len(prob.logbooks)):
            prob.write_logbook(folder+"/EQ"+"_"+problem+"_"+str(run)+".ga.log"+str(i),prob.logbooks[i])


