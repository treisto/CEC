
# -*- coding: utf-8 -*-
import fnmatch
import copy
import inspect
import re
import numpy
import math

INF=1e+8

SENSES={-1:"<=",0:"=",1:">="}

class Benchmark:

    def __init__(self,**args):
        self.best_know=None
        self.x=[]
        self.senses=[]
        self.rhs=[]
        self.bounds=[]
        self.objectives=[]

        self.constraints=[]
        methods_name = self.__class__.__dict__.keys()

        cstr_name = [m for m in methods_name if fnmatch.fnmatch(m,"g_*")]
        sort_name = sorted(cstr_name,key=lambda name: int(name.split("_")[1]))
        for i in range(len(sort_name)):
            self.constraints.append(self.__class__.__dict__[sort_name[i]])


        name_obj = [m for m in methods_name if fnmatch.fnmatch(m,"obj_*") ]
        sort_name = sorted(name_obj,key=lambda name: int(name.split("_")[1]))
        for i in range(len(sort_name)):
            self.objectives.append(self.__class__.__dict__[sort_name[i]])
        self.tolerance=1e-4

    def __str__(self):
        s="Problem "+self.__class__.__name__+"\n"
        for i,obj in enumerate(self.objectives):
            source = inspect.getsource(obj)
            match = re.search("[^a-zA-Z](return)[^a-zA-Z]",source)
            returnPos = match.start(1) +  6
            s+="obj "+str(i+1)+":" + inspect.getsource(obj)[returnPos:]+"\n"
        for i,cstr in enumerate(self.constraints):
            source = inspect.getsource(cstr)
            match = re.search("[^a-zA-Z](return)[^a-zA-Z]",source)
            returnPos = match.start(1) +  6
            s+="cstr "+str(i+1)+":" + inspect.getsource(cstr)[returnPos:].replace("\n","")+" "+SENSES[self.senses[i]]+" "+str(self.rhs[i])+"\n"
        return s


    def nconstraints(self):
        return len(self.constraints)

    def nobjectives(self):
        return len(self.objectives)

    def generate_interaction_table(self):
        self.interaction=[[0]*self.nvariables() for k in range(self.nconstraints())]
        for i in range(self.nconstraints()):
            for item in self.extract_variables_from_constraint(i):
                self.interaction[i][int(item)]  = 1



    def extract_variables_from_constraint(self,i):
        source=inspect.getsource(self.constraints[i])
        matches = re.finditer('x\[\d*\]',source) #iterator
        decision_var=[]
        for it in matches:
            txt_var = source[it.start():it.end()].translate(None,"x[]")
            decision_var.append(int(txt_var))
        decision_var = list(numpy.unique(decision_var))
        return sorted(decision_var)


    def coupling_variables(self,i,j):
        if not hasattr(self, 'interaction'):
            self.generate_interaction_table()
        return sum([1  for k in range(self.nvariables()) if ((self.interaction[i][k]==self.interaction[j][k])  and self.interaction[i][k]==1)])

    def nvariables(self):
        return len(self.x)

    def evaluate_cstr(self,i):
        assert(None not in self.x)
        if self.senses[i] == 0 :
            return self.constraints[i](self) == self.rhs[i]
        elif self.senses[i] == -1:
            return self.constraints[i](self) <= self.rhs[i]
        elif self.senses[i] == 1:
            return self.constraints[i](self) >= self.rhs[i]

    def evaluate_left_hand_side_constraint(self,i):
        assert(None not in self.x)
        return self.constraints[i](self)

    def violation_constraint(self,i):
        value = self.evaluate_left_hand_side_constraint(i)
        if self.senses[i] == -1:
            if value <= self.rhs[i]:
                return 0
            else:
                return value - self.rhs[i]
        if self.senses[i] == 0:
            if abs(value - self.rhs[i])<=self.tolerance:
                return 0
            else:
                return abs(value-self.rhs[i])
        if self.senses[i] == 1:
            if value >= self.rhs[i]:
                return 0
            else:
                return self.rhs[i] - value

    def is_constraint_satisfied(self,i):
        return self.violation_constraint(i) == 0

    def is_feasible(self,sol):
        self.unset_var()
        self.set_var(sol)
        return sum([self.violation_constraint(j) for j in range(self.nconstraints())]) == 0

    def evaluate_objectives(self,i):
        assert(None not in self.x)
        return self.objectives[i](self)


    def test_assert(self):
        assert(len(self.x) == len(self.bounds))
        assert(len(self.constraints) == len(self.senses))
        assert(len(self.senses) == len(self.rhs))
        assert(len(self.x) > 0)
        #assert(len(self.constraints) > 0)
        assert(len(self.objectives) > 0)

    def unset_var(self):
        for i in range(len(self.x)):
            self.x[i] = None

    def set_var(self,liste):
        assert(len(liste)==len(self.x))
        self.assert_inside_bounds(liste)
        self.x=copy.copy(liste)

    def is_unset(self):
        return None in self.x

    def assert_inside_bounds(self,liste):
            pass
        #for i in range(len(liste)):
        #    assert( liste[i] >= self.bounds[i][0] and liste[i] <= self.bounds[i][1])


