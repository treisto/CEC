
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


class C00(Benchmark):
    def __init__(self, D=None, x=None, senses=None, bounds=None):
        # parent class constructor
        Benchmark.__init__(self)
        # extract problem number from class name
        problemno = int(self.__class__.__name__[1:])
        # load shift dataset
        self.o = numpy.loadtxt("./inputData/shift_data_" + str(problemno) + ".txt",dtype=numpy.float32)
        # default values
        self.D = D or 10
        self.D2 = int(self.D/2)
        self.x = x or [78, 33, 27, 27, 27, 0, 0, 0, 0, 0]
        self.set_var(self.x)
        self.senses = senses or [-1]
        self.rhs = [0]*len(self.senses)
        bounds = bounds or (-100,100)
        self.bounds = [bounds] * self.D
        # checks
        self.test_assert()

class C01(C00):

    def obj_1(self):
        z = self.x-self.o[:self.D]
        result=0
        for i in range (self.D):
            SumZ = 0.0
            for j in range (i):
                SumZ += z[j]
            result += SumZ**2
        return result

    def g_1(self):
        zz = self.x-self.o[:self.D]
        return sum([ z * z - 5000 * math.cos(0.1 * math.pi * z) - 4000 for z in zz])

class C02(C00):

    def __init__(self):
        C00.__init__(self, bounds=(-10,10))
        self.M = numpy.loadtxt("./inputData/M_2_D" + str(self.D) + ".txt", dtype=numpy.float32)

    def obj_1(self):
        z = self.x - self.o[:self.D]
        result = 0.0
        for i in range(self.D):
            SumZ = 0.0
            for j in range (i):
                SumZ += z[j]
            result = SumZ**2
        return result

    def g_1(self):
        z = self.x - self.o[:self.D]
        yy = numpy.dot(self.M, z)
        return sum([ y * y - 5000 * math.cos(0.1 * math.pi * y) - 4000 for y in yy ])

class C03(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0], bounds=(-10,10))


    def obj_1(self):
        z = self.x-self.o[:self.D]
        result = 0.0
        for i in range (0, self.D):
            SumZ = 0.0
            for j in range (0, i):
                SumZ += z[i]
            result += SumZ**2
        return result

    def g_1(self):
        zz = self.x-self.o[:self.D]
        return sum([ z * z - 5000.0 * math.cos(0.1 * math.pi * z) - 4000.0 for z in zz ])

    def g_2(self):
        zz = self.x-self.o[:self.D]
        return - sum([ z * math.sin(0.1 * math.pi * z) for z in zz ])

class C04(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, -1], bounds=(-10,10))

    def obj_1(self):
        zz = self.x - self.o[:self.D]
        return sum([ z * z - 10 * math.cos(2 * math.pi * z) + 10.0 for z in zz ])

    def g_1(self):
        zz = self.x - self.o[:self.D]
        return - sum([ z * math.sin(2 * z) for z in zz ])

    def g_2(self):
        zz = self.x - self.o[:self.D]
        return sum([ z * math.sin(z) for z in zz ])
		
class C05(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, -1], bounds=(-10,10))
        self.M1 = numpy.loadtxt("./inputData/M1_5_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.M2 = numpy.loadtxt("./inputData/M2_5_D" + str(self.D) + ".txt", dtype=numpy.float32)

    def obj_1(self):
        z = self.x - self.o[:self.D]
        result = 0.0
        for i in range (self.D - 1):
            result += 100.0 * (z[i] * z[i] - z[i+1]) ** 2 + (z[i] - 1) ** 2
        return result

    def g_1(self):
        z = self.x - self.o[:self.D]
        yy = numpy.dot(self.M1, z)
        return sum([ y * y - 50.0 * math.cos(2 * math.pi * y) - 40.0 for y in yy ])

    def g_2(self):
        z = self.x - self.o[:self.D]
        ww = numpy.dot(self.M2, z)
        return sum([ w * w - 50.0 * math.cos(2 * math.pi * w) - 40.0 for w in ww ])
		
class C06(C00):

    def __init__(self):
        C00.__init__(self, senses=[0]*6, bounds=(-20,20))

    def obj_1(self):
        zz = self.x - self.o[:self.D]
        return sum([ z * z - 10.0 * math.cos(2 * math.pi * z) + 10.0 for z in zz ])

    def g_1(self):
        zz = self.x - self.o[:self.D]
        return -sum([z * math.sin(z) for z in zz])

    def g_2(self):
        zz = self.x - self.o[:self.D]
        return sum([z * math.sin(math.pi * z) for z in zz])

    def g_3(self):
        zz = self.x - self.o[:self.D]
        return -sum([z * math.cos(z) for z in zz])

    def g_4(self):
        zz = self.x - self.o[:self.D]
        return sum([z * math.sin(math.pi * z) for z in zz])

    def g_5(self):
        zz = self.x - self.o[:self.D]
        return sum([z * math.sin(2.0 * math.sqrt(abs(z))) for z in zz])

    def g_6(self):
        zz = self.x - self.o[:self.D]
        return -sum([z * math.sin(2.0 * math.sqrt(abs(z))) for z in zz])
		
class C07(C00):

    def __init__(self):
        C00.__init__(self, senses=[0, 0], bounds=(-50,50))

    def obj_1(self):
        zz = self.x - self.o[:self.D]
        return sum([ z * math.sin(z) for z in zz ])

    def g_1(self):
        zz = self.x - self.o[:self.D]
        return sum([ z - 100.0 * math.cos(0.5 * z) + 100.0 for z in zz ])

    def g_2(self):
        zz = self.x - self.o[:self.D]
        return - sum([ z - 100.0 * math.cos(0.5 * z) + 100.0 for z in zz ])

class C08(C00):

    def __init__(self):
        C00.__init__(self, senses=[0, 0])

    def setup(self):
        self.z = self.x - self.o[:self.D]
        z = list(self.z)
        self.y = []
        self.w = []
        while z:
            self.w.insert(0, z.pop())
            self.y.insert(0, z.pop())

    def obj_1(self):
        z = self.x - self.o[:self.D]
        return max(z)

    def g_1(self):
        z = list(self.x - self.o[:self.D])
        y = []
        w = []
        while z:
            w.insert(0, z.pop())
            y.insert(0, z.pop())
        result = 0.0
        for i in range(self.D2):
            SumY = 0.0
            for j in range(i):
                SumY += y[j]
            result = SumY**2
        return result


    def g_2(self):
        z = list(self.x - self.o[:self.D])
        y = []
        w = []
        while z:
            w.insert(0, z.pop())
            y.insert(0, z.pop())
        result = 0.0
        for i in range(self.D2):
            SumW = 0.0
            for j in range(i):
                SumW += w[j]
            result = SumW**2
        return result

class C09(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0], bounds=(-10,10))


    def obj_1(self):
        z = self.x - self.o[:self.D]
        return max(z)

    def g_1(self):
        z = list(self.x - self.o[:self.D])
        y = []
        w = []
        while z:
            w.insert(0, z.pop())
            y.insert(0, z.pop())
        result = 1.0
        for i in range(self.D2):
            result = result * w[i]
        return result

    def g_2(self):
        z = list(self.x - self.o[:self.D])
        y = []
        w = []
        while z:
            w.insert(0, z.pop())
            y.insert(0, z.pop())
        result = 0.0
        for i in range(self.D2-1):
            result += (y[i] * y[i] - y[i+1]) ** 2
        return result

class C10(C00):

    def __init__(self):
        C00.__init__(self, senses=[0, 0])

    def obj_1(self):
        z = self.x - self.o[:self.D]
        return max(z)

    def g_1(self):
        z = self.x - self.o[:self.D]
        result = 0.0
        for i in range (self.D):
            SumZ = 0.0
            for j in range (i):
                SumZ += z[i]
            result += SumZ**2
        return result

    def g_2(self):
        z = self.x - self.o[:self.D]
        result = 0.0
        for i in range(self.D-1):
            result += (z[i] - z[i+1]) ** 2
        return result

class C11(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        z = self.x - self.o[:self.D]
        return sum(z)

    def g_1(self):
        z = self.x - self.o[:self.D]
        result = 1.0
        for i in range(self.D):
            result = result * z[i]
        return result

    def g_2(self):
        z = self.x - self.o[:self.D]
        'same as C10.h_2'
        result = 0.0
        for i in range(self.D-1):
            result += (z[i] - z[i+1]) ** 2
        return result

class C12(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y - 10.0 * math.cos(2 * math.pi * y) + 10.0 for y in yy ])

    def g_1(self):
        yy = self.x - self.o[:self.D]
        return 4 - sum([ abs(y) for y in yy ])

    def g_2(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y for y in yy ]) - 4

class C13(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, -1, -1])

    def obj_1(self):
        y = self.x - self.o[:self.D]
        result = 0.0
        for i in range(self.D-1):
            result += 100.0 * (y[i] * y[i] - y[i+1]) ** 2 + (y[i] - 1) ** 2
        return result

    def g_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y - 10.0 * math.cos(2 * math.pi * y) + 10.0 for y in yy ]) - 100.0

    def g_2(self):
        yy = self.x - self.o[:self.D]
        return sum([ y for y in yy ]) - 2 * self.D

    def g_3(self):
        yy = self.x - self.o[:self.D]
        return 5.0 - sum([ y for y in yy ])


class C14(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        return (-20.0 * math.exp(-0.2 * math.sqrt(1.0/self.D * sum([ y * y for y in yy ])))
            + 20.0 - math.exp( 1.0/self.D * sum([ math.cos(2 * math.pi * y) for y in yy ]) )
            + math.e)

    def g_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y for y in yy[1:] + 1 - abs(yy[0]) ])

    def g_2(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y for y in yy ]) - 4
		
class C15(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        return max([ abs(y) for y in yy ])

    def g_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ y*y for y in yy ]) - 100.0 * self.D

    def g_2(self):
        yy = self.x - self.o[:self.D]
        fx = max([ abs(y) for y in yy ])
        return math.cos(fx) + math.sin(fx)

class C16(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ abs(y) for y in yy ])

    def g_1(self):
        yy = self.x - self.o[:self.D]
        return sum([ y*y for y in yy ]) - 100.0 * self.D

    def g_2(self):
        yy = self.x - self.o[:self.D]
        fx = sum([ abs(y) for y in yy ])
        return (math.cos(fx) + math.sin(fx)) ** 2 - math.exp(math.cos(fx) + math.sin(fx)) - 1 + math.exp(1)

class C17(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        product = 1.0
        for i in range(self.D):
            product = product * (math.cos(yy[i] / math.sqrt(i+1)))
        return 1.0/4000.0 * sum([ y * y for y in yy ]) + 1.0 - product

    def g_1(self):
        y = self.x - self.o[:self.D]
        result = 0.0
        for i in range(self.D):
            SumY = 0.0
            for j in range(self.D):
                if not i == j:
                    SumY += y[j] * y[j]
            result += numpy.sign(abs(y[i]) - SumY - 1.0)
        return 1.0 - result

    def g_2(self):
        yy = self.x - self.o[:self.D]
        return sum([ y * y for y in yy ]) - 4 * self.D

class C18(C00):

    def __init__(self):
        C00.__init__(self, senses=[-1, -1, 0])

    def obj_1(self):
        yy = self.x - self.o[:self.D]
        zz = []
        for y in yy:
            if abs(y) < 0.5:
                zz.append(y)
            else:
                zz.append(0.5 * round(2.0 * y))
        return sum([ z * z - math.cos(2.0 * math.pi * z) + 10.0 for z in zz ])

    def g_1(self):
        yy = self.x - self.o[:self.D]
        zz = []
        for y in yy:
            if abs(y) < 0.5:
                zz.append(y)
            else:
                zz.append(0.5 * round(2.0 * y))
        return 1.0 - sum([ abs(y) for y in yy ])

    def g_2(self):
        yy = self.x - self.o[:self.D]
        zz = []
        for y in yy:
            if abs(y) < 0.5:
                zz.append(y)
            else:
                zz.append(0.5 * round(2.0 * y))
        return sum([ y * y for y in yy ]) - 100.0 * self.D

    def g_3(self):
        yy = self.x - self.o[:self.D]
        zz = []
        for y in yy:
            if abs(y) < 0.5:
                zz.append(y)
            else:
                zz.append(0.5 * round(2.0 * y))
        result = 1.0
        for i in range(self.D):
            result = result * (math.sin(yy[i] - 1) ** 2) * math.pi
        for i in range(self.D-1):
            result += 100.0 * (yy[i] * yy[i] - yy[i+1]) ** 2
        return result

class C19(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return sum([ math.sqrt(abs(y)) + 2 * math.sin(y * y * y) for y in self.y ])

    def g_1(self):
        result = 0.0
        for i in range(self.D-1):
            result += -10.0 * math.exp(-0.2 * math.sqrt(self.y[i] ** 2 + self.y[i+1] ** 2)) + (self.D - 1) * 10.0 / math.exp(-5.0)
        return result

    def g_2(self):
        return sum([ math.sin(2.0 * y) **2 for y in self.y ]) - 0.5 * self.D

class C20(C00):

    def g(self, y1, y2):
        return 0.5 + ( math.sin(math.sqrt(y1 * y1 + y2 * y2)) ** 2 + 0.5) / (1.0 + 0.001 * math.sqrt(y1 * y1 + y2 * y2)) ** 2

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        result = 0.0
        for i in range(self.D-1):
            result += self.g(self.y[i], self.y[i+1])
        return result + self.g(self.y[self.D-1], self.y[0])

    def g_1(self):
        s = sum([ y * y for y in self.y ])
        return math.cos(s) ** 2 - 0.25 * math.cos(s) - 0.125

    def g_2(self):
        s = sum([ y * y for y in self.y ])
        return math.exp(math.cos(s)) - math.exp(0.25)

class C21(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_21_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        return sum([ y * y - 10.0 * math.cos(2 * math.pi * y) + 10.0 for y in self.y ])

    def g_1(self):
        return 4.0 - sum([ abs(z) for z in self.z ])

    def g_2(self):
        return sum([ z * z for z in self.z ]) - 4.0

class C22(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_22_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        result = 0.0
        for i in range(self.D):
            result += 100 * (self.z[i] ** 2 - self.x[i] ** 2) ** 2 + (self.z[i] - 1.0) ** 2
        return result

    def g_1(self):
        return sum([ z * z - 10.0 * math.cos(2 * math.pi * z) + 10.0 for z in self.z ]) - 100.0

    def g_2(self):
        return sum(self.z) - 2.0 * self.D

    def g_3(self):
        return 5.0 - sum(self.z)

class C23(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_23_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        z2 = sum([ z * z for z in self.z ])
        cosz = sum([ math.cos(2 * math.pi * z) for z in self.z ])
        return -20.0 * math.exp(-0.2 * math.sqrt(1.0/self.D * z2)) + 20.0 - math.exp(1.0/self.D * cosz) + math.e

    def g_1(self):
        result = 0.0
        for i in range(1, self.D):
            result += self.z[i] ** 2
        return result + 1.0 - abs(self.z[0])

    def g_2(self):
        return sum([ z * z for z in self.z]) - 4.0

class C24(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_24_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        self.fz = max([ abs(z) for z in self.z ])
        return self.fz

    def g_1(self):
        return sum([ z * z for z in self.z]) - 100.0 * self.D

    def g_2(self):
        return math.cos(self.fz) + math.sin(self.fz)

class C25(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_25_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        self.fz = sum([ abs(z) for z in self.z ])
        return self.fz

    def g_1(self):
        return sum([ z * z for z in self.z]) - 100.0 * self.D

    def g_2(self):
        return (math.cos(self.fz) + math.sin(self.fz)) ** 2 - math.exp(math.cos(self.fz) + math.sin(self.fz)) - 1.0 + math.exp(1.0)

class C26(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_26_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        product = 1.0
        for i in range(self.D):
            product = product * math.cos(self.y[i] / math.sqrt(i+1))
        return 1.0/4000.0 * sum([ y * y for y in self.y ]) + 1.0 - product

    def g_1(self):
        result = 0.0
        for i in range(self.D):
            sumz = 0.0
            for j in range(self.D):
                if not i == j:
                    sumz += self.z[j] ** 2
            result += numpy.sign(abs(self.z[i]) - sumz - 1.0)
        return 1.0 - result

    def g_2(self):
        return sum([ z * z for z in self.z]) - 4.0 * self.D

class C27(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_27_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        return sum([ z * z - 10.0 * math.cos(2 * math.pi * z) + 10.0 for z in self.z ])

    def g_1(self):
        return 1.0 - sum([ abs(y) for y in self.y ])

    def g_2(self):
        return sum([ z * z for z in self.z]) - 100.0 * self.D

    def g_3(self):
        product = 1.0
        for i in range(self.D):
            product = product * math.sin(self.y[i] - 1.0) ** 2 * math.pi
        result = 0.0
        for i in range(self.D-1):
            result += 100.0 * (self.y[i] ** 2 - self.y[i+1]) ** 2
        return result + product

class C28(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.M = numpy.loadtxt("./inputData/M_28_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.z = numpy.dot(self.M, self.y)
        return sum([ math.sqrt(abs(z)) + 2.0 * math.sin(z * z * z) for z in self.z ])

    def g_1(self):
        result = 0.0
        for i in range(self.D-1):
            result += -10.0 * math.exp(-0.2 * math.sqrt(self.z[i] ** 2 + self.z[i+1] ** 2))
        return 1.0 - sum([ abs(y) for y in self.y ]) + (self.D - 1.0) * 10.0 / math.exp(-5)

    def g_2(self):
        return sum([ math.sin(2.0 * z) ** 2 for z in self.z]) - 0.5 * self.D

if  __name__ == "__main__":

    def test_problem(cls):
        g = cls()
        print("= " + cls.__name__ + " ===============")
        print("= Constraint violations:")
        for i in range(g.nconstraints()):
            print(g.violation_constraint(i))
        print("= Objectives:")
        print(g.evaluate_objectives(0))

    test_problem(C01)
    test_problem(C02)
    test_problem(C03)
    test_problem(C04)
    test_problem(C05)
    test_problem(C06)
    test_problem(C07)
    test_problem(C08)
    test_problem(C09)
    test_problem(C10)
    # test_problem(C11, senses=[-1, 0])
    # test_problem(C12, senses=[-1, 0])
    # test_problem(C13, senses=[-1, -1, -1])
    # test_problem(C14, senses=[-1, 0])
    # test_problem(C15, senses=[-1, 0])
    # test_problem(C16, senses=[-1, 0])
    # test_problem(C17, senses=[-1, 0])
    # test_problem(C18, senses=[-1, -1, 0])
    # test_problem(C19, senses=[-1, -1], bounds=(-50,50))
    # test_problem(C20, senses=[-1, -1])
    # test_problem(C21, senses=[-1, 0])
    # test_problem(C22, senses=[-1, -1, -1])
    # test_problem(C23, senses=[-1, 0])
    # test_problem(C24, senses=[-1, 0])
    # test_problem(C25, senses=[-1, 0])
    # test_problem(C26, senses=[-1, 0])
    # test_problem(C27, senses=[-1, -1, 0])
    # test_problem(C28, senses=[-1, -1], bounds=(-50,50))

class G04(Benchmark):

    def __init__(self,**args):
        Benchmark.__init__(self,**args)
        self.best_know=-30665.5387
        self.x=[None for i in range(5)]
        self.rhs=[0]*6
        self.senses = [-1]*6
        self.bounds=[(78,102),(33,45),(27,45),(27,45),(27,45)]
        self.test_assert()

    def obj_1(self):
        return 5.3578547*numpy.power(self.x[2],2) + 0.8356891*self.x[0]*self.x[4] + 37.293239*self.x[0] - 40792.141

    def g_1(self):
        return 85.334407 + 0.0056858*self.x[1]*self.x[4]+0.0006262*self.x[0]*self.x[3] - 0.0022053*self.x[2]*self.x[4] -92

    def g_2(self):
        return (-1.0)*self.g_1() - 92

    def g_3(self):
        return 80.51249+0.0071317*self.x[1]*self.x[4] + 0.0029955*self.x[0]*self.x[1] + 0.0021813*numpy.power(self.x[2],2)-110

    def g_4(self):
        return -(1.0)*self.g_3() - 20

    def g_5(self):
        return 9.300961+0.0047026*self.x[2]*self.x[4]+0.0012547*self.x[0]*self.x[2]+0.0019085*self.x[2]*self.x[3] - 25

    def g_6(self):
        return (-1.0)*self.g_5() - 5
