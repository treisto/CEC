# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__ = "Emmanuel Kieffer"
__copyright__ = "Copyright (C) 2016, Emmanuel Kieffer"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Emmanuel Kieffer"
__email__ = "emmanuel.kieffer@uni.lu"
__status__ = "dev"

from scipy.optimize import minimize
from benchmarks import *
from copy import copy
from numpy import *

class SLSQP():
    '''
    Local search algorithm

    '''

    def __init__(self,bench):
        self.bench=bench
        self.obj=bench.objectives[0]
        def wrapper(x):
            '''
            Just wrap the original constraints as new objective functions
            '''
            self.bench.unset_var()
            self.bench.set_var(x)
            total=0
            for i in range(self.bench.nconstraints()):
                total += self.bench.violation_constraint(i)
            return  total

        def wrapper_cstr(x,i):
            '''
            Just wrap the constraints
            In fact, no necessary in theory but mandatory for the numpy function
            '''
            self.bench.unset_var()
            self.bench.set_var(x)
            return (-1.0)*self.bench.constraints[i](self.bench)



        self.func = lambda x,sign=1.0: wrapper(x)
        self.cons=[]
        for i in range(self.bench.nconstraints()):
            dico={}
            if self.bench.senses[i] == 0:
                dico['type']='eq'
            else:
                dico['type']='ineq'
            dico['fun'] = lambda x,cstr=i :wrapper_cstr(x,cstr)
            self.cons.append(dico)
            
    def addConstraints(self,value):
        '''
        Add epsilon-constraint on objective
        '''
        def wrapper(x,v):
            self.bench.unset_var()
            self.bench.set_var(x)
            return (-1.0)*self.bench.evaluate_objectives(0) + value
        dico={}
        dico['type']='ineq'
        dico['fun'] = lambda x,v=value :wrapper(x,v)
        self.cons.append(dico)

    def solve(self,init):
        return minimize(self.func,init,args=(-1.0),tol=1e-20,constraints=self.cons,bounds=self.bench.bounds,method="SLSQP")



if __name__ == "__main__":

   ben=G06()
   s=SLSQP(ben)
   sol = s.solve([15,20])
   print(sol)
   ben.unset_var()
   ben.set_var(sol.x)
   print(ben.evaluate_objectives(0))

