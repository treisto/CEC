
from benchmarks import *

class C00(Benchmark):
    def __init__(self, D=10, x=None, rhs=None, senses=None, bounds=None):
        # parent class constructor
        Benchmark.__init__(self)
        # extract problem number from class name
        problemno = int(self.__class__.__name__[1:])
        # load shift dataset
        self.o = numpy.loadtxt("./inputData/shift_data_" + str(problemno) + ".txt",dtype=numpy.float32)
        # default values
        self.D = D
        self.D2 = int(D/2)
        self.x = x or [None for i in range(self.D)]
        self.rhs = rhs or [0]
        self.senses = senses or [-1]
        bounds = bounds or (-100,100)
        self.bounds = [bounds] * self.D
        # checks
        self.test_assert()

class C01(C00):

    def obj_1(self):
        self.z=self.x-self.o[:self.D]
        result=0
        for i in range (self.D):
            SumZ = 0.0
            for j in range (i):
                SumZ += self.z[j]
            result += SumZ**2
        return result

    def g_1(self):
        return sum([ z * z - 5000 * math.cos(0.1 * math.pi * z) - 4000 for z in self.z])

class C02(C00):
    def __init__(self, **args):
        C00.__init__(self, **args)
        self.M = numpy.loadtxt("./inputData/M_2_D" + str(self.D) + ".txt", dtype=numpy.float32)

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        self.y = self.M * self.z
        result = 0.0
        for i in range(self.D):
            SumZ = 0.0
            for j in range (i):
                SumZ += self.z[j]
            result = SumZ**2
        return result

    def g_1(self):
        return sum([ z * z - 5000 * math.cos(0.1 * math.pi * z) - 4000 for z in self.z ])

class C03(C00):

    def obj_1(self):
        self.z=self.x-self.o[:self.D]
        result=0
        for i in range (0, self.D):
            SumZ=0
            for j in range (0, i):
                SumZ += self.z[i]
            result += SumZ**2
        return result

    def g_1(self):
        return sum([ z * z - 5000 * math.cos(0.1 * math.pi * z) - 4000 for z in self.z ])

    def g_2(self):
        return - sum([ z * math.sin(0.1 * math.pi * z) for z in self.z ])

class C04(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        return sum([ z * z - 10 * math.cos(2 * math.pi * z) + 10 for z in self.z ])

    def g_1(self):
        return - sum([ z * math.sin(2 * z) for z in self.z ])

    def g_2(self):
        return sum([ z * math.sin(z) for z in self.z ])

class C05(C00):
    def __init__(self, **args):
        C00.__init__(self, **args)
        self.M1 = numpy.loadtxt("./inputData/M1_5_D" + str(self.D) + ".txt", dtype=numpy.float32)
        self.M2 = numpy.loadtxt("./inputData/M2_5_D" + str(self.D) + ".txt", dtype=numpy.float32)

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        self.y = numpy.dot(self.M1, self.z)
        self.w = numpy.dot(self.M2, self.z)
        z = self.z
        result = 0.0
        for i in range (self.D - 1):
            result += 100 * (z[i] * z[i] - z[i+1]) ** 2 + (z[i] - 1) ** 2
        return result

    def g_1(self):
        return sum([ y * y - 50 * math.cos(2 * math.pi * y) - 40 for y in self.y ])

    def g_2(self):
        return sum([ w * w - 50 * math.cos(2 * math.pi * w) - 40 for w in self.w ])

class C06(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        return sum([ z * z - 10 * math.cos(2 * math.pi * z) + 10 for z in self.z ])

    def g_1(self):
        return -sum([z * math.sin(z) for z in self.z])

    def g_2(self):
        return sum([z * math.sin(math.pi * z) for z in self.z])

    def g_3(self):
        return -sum([z * math.cos(z) for z in self.z])

    def g_4(self):
        return sum([z * math.sin(math.pi * z) for z in self.z])

    def g_5(self):
        return sum([z * math.sin(2 * math.sqrt(math.fabs(z))) for z in self.z])

    def g_6(self):
        return -sum([z * math.sin(2 * math.sqrt(math.fabs(z))) for z in self.z])

class C07(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        return sum([ z * math.sin(z) for z in self.z ])

    def g_1(self):
        return sum([ z - 100 * math.cos(0.5 * z) + 100 for z in self.z ])

    def g_2(self):
        return - sum([ z - 100 * math.cos(0.5 * z) + 100 for z in self.z ])

class C08(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        self.y = []
        self.w = []
        z = list(self.z)
        while z:
            self.w.insert(0, z.pop())
            self.y.insert(0, z.pop())
        return max(self.z)

    def g_1(self):
        result = 0.0
        for i in range(self.D2):
            SumY = 0
            for j in range(i):
                SumY += self.y[j]
            result = SumY**2
        return result


    def g_2(self):
        result = 0.0
        for i in range(self.D2):
            SumW = 0
            for j in range(i):
                SumW += self.w[j]
            result = SumW**2
        return result

class C09(C00):

    def obj_1(self):
        'just like in C08'
        self.z = self.x - self.o[:self.D]
        self.y = []
        self.w = []
        z = list(self.z)
        while z:
            self.w.insert(0, z.pop())
            self.y.insert(0, z.pop())
        return max(self.z)

    def g_1(self):
        result = 1.0
        for i in range(self.D2):
            result = result * self.w[i]
        return result

    def g_2(self):
        result = 0.0
        for i in range(self.D2-1):
            result += (self.y[i] * self.y[i] - self.y[i+1]) ** 2
        return result

class C10(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        return max(self.z)

    def g_1(self):
        'same as in C03.obj_1'
        result=0
        for i in range (self.D):
            SumZ=0
            for j in range (i):
                SumZ += self.z[i]
            result += SumZ**2
        return result

    def g_2(self):
        result = 0.0
        for i in range(self.D-1):
            result += (self.z[i] - self.z[i+1]) ** 2
        return result

class C11(C00):

    def obj_1(self):
        self.z = self.x - self.o[:self.D]
        return sum(self.z)

    def g_1(self):
        result = 1.0
        for i in range(self.D):
            result = result * self.z[i]
        return result

    def g_2(self):
        'same as C10.g_2'
        result = 0.0
        for i in range(self.D-1):
            result += (self.z[i] - self.z[i+1]) ** 2
        return result

class C12(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return sum([ y * y - 10 * math.cos(2 * math.pi * y) + 10 for y in self.y ])

    def g_1(self):
        return 4 - sum([ abs(y) for y in self.y ])

    def g_2(self):
        return sum([ y * y for y in self.y ]) - 4

class C13(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        y = self.y
        result = 0.0
        for i in range(self.D-1):
            result += 100 * (y[i] * y[i] - y[i+1]) ** 2 + (y[i] - 1) ** 2
        return result

    def g_1(self):
        return sum([ y * y - 10 * math.cos(2 * math.pi * y) + 10 for y in self.y ]) - 100

    def g_2(self):
        return sum([ y for y in self.y ]) - 2 * self.D

    def g_3(self):
        return 5 - sum([ y for y in self.y ])

class C14(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return (-20 * math.exp(-0.2 * math.sqrt(1.0/self.D * sum([ y * y for y in self.y ])))
            + 20 - math.exp( 1.0/self.D * sum([ math.cos(2 * math.pi * y) for y in self.y ]) )
            + math.e)

    def g_1(self):
        return sum([ y * y for y in self.y[1:] + 1 - abs(self.y[0]) ])

    def g_2(self):
        return sum([ y * y for y in self.y ]) - 4

class C15(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return max([ abs(y) for y in self.y ])

    def g_1(self):
        return sum([ y*y for y in self.y ]) - 100 * self.D

    def g_2(self):
        fx = max([ abs(y) for y in self.y ])
        return math.cos(fx) + math.sin(fx)

class C16(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return sum([ abs(y) for y in self.y ])

    def g_1(self):
        return sum([ y*y for y in self.y ]) - 100 * self.D

    def g_2(self):
        fx = sum([ abs(y) for y in self.y ])
        return (math.cos(fx) + math.sin(fx)) ** 2 - math.exp(math.cos(fx) + math.sin(fx)) - 1 + math.exp(1)

class C17(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        product = 1.0
        for i in range(self.D):
            product = product * (math.cos(self.y[i] / math.sqrt(i+1)))
        return 1.0/4000 * sum([ y * y for y in self.y ]) + 1 - product

    def g_1(self):
        result = 0.0
        for i in range(self.D):
            SumY = 0.0
            for j in range(self.D):
                if not i == j:
                    SumY += self.y[j] * self.y[j]
            result += numpy.sign(abs(self.y[i]) - SumY - 1)
        return 1 - result

    def g_2(self):
        return sum([ y * y for y in self.y ]) - 4 * self.D

class C18(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        self.z = []
        for y in self.y:
            if abs(y) < 0.5:
                self.z.append(y)
            else:
                self.z.append(0.5 * round(2 * y))
        return sum([ z * z - math.cos(2 * math.pi * z) + 10 for z in self.z ])

    def g_1(self):
        return 1 - sum([ abs(y) for y in self.y ])

    def g_2(self):
        return sum([ y * y for y in self.y ]) - 100 * self.D

    def g_3(self):
        result = 1.0
        for i in range(self.D):
            result = result * (math.sin(self.y[i] - 1) ** 2) * math.pi
        for i in range(self.D-1):
            result += 100 * (self.y[i] * self.y[i] - self.y[i+1]) ** 2
        return result

class C19(C00):

    def obj_1(self):
        self.y = self.x - self.o[:self.D]
        return sum([ math.sqrt(abs(y)) + 2 * math.sin(y * y * y) for y in self.y ])

    def g_1(self):
        result = 0.0
        for i in range(self.D-1):
            result += -10 * math.exp(-0.2 * math.sqrt(self.y[i] ** 2 + self.y[i+1] ** 2)) + (self.D - 1) * 10 / math.exp(-5)
        return result

    def g_2(self):
        return sum([ math.sin(2 * y) **2 for y in self.y ]) - 0.5 * self.D

if  __name__ == "__main__":

    def test_problem(cls, **args):
        g = cls(**args)
        g.set_var([78, 33, 27,27,27, 0, 0, 0, 0, 0])
        print("= " + cls.__name__ + " ===============")
        print("= Objectives:")
        print(g.evaluate_objectives(0))
        print("= Constraint violations:")
        for i in range(g.nconstraints()):
            print(g.violation_constraint(i))

    test_problem(C01)
    test_problem(C02, bounds=(-10,10))
    test_problem(C03, rhs=[0, 0], senses=[-1, 0], bounds=(-10,10))
    test_problem(C04, rhs=[0, 0], senses=[-1, -1], bounds=(-10,10))
    test_problem(C05, rhs=[0, 0], senses=[-1, -1], bounds=(-10,10))
    test_problem(C06, rhs=[0]*6, senses=[0]*6, bounds=(-20,20))
    test_problem(C07, rhs=[0, 0], senses=[0, 0], bounds=(-50,50))
    test_problem(C08, rhs=[0, 0], senses=[0, 0])
    test_problem(C09, rhs=[0, 0], senses=[-1, 0], bounds=(-10,10))
    test_problem(C10, rhs=[0, 0], senses=[0, 0])
    test_problem(C11, rhs=[0, 0], senses=[-1, 0])
    test_problem(C12, rhs=[0, 0], senses=[-1, 0])
    test_problem(C13, rhs=[0, 0, 0], senses=[-1, -1, -1])
    test_problem(C14, rhs=[0, 0], senses=[-1, 0])
    test_problem(C15, rhs=[0, 0], senses=[-1, 0])
    test_problem(C16, rhs=[0, 0], senses=[-1, 0])
    test_problem(C17, rhs=[0, 0], senses=[-1, 0])
    test_problem(C18, rhs=[0, 0, 0], senses=[-1, -1, 0])
    test_problem(C19, rhs=[0, 0], senses=[-1, -1], bounds=(-50,50))
