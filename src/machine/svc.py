from sklearn.svm import SVC
import random


class Svc:

    @staticmethod
    def generate(numberFeatures, icls, use_col):
        listKernel = ["linear", "rbf", "poly", "sigmoid"]

        gen = [
            listKernel[random.randint(0, 3)],
            random.uniform(0.1, 100),
            random.uniform(0.1, 5),
            random.uniform(0.001, 5),
            random.uniform(0.01, 10),
        ]

        if use_col:
            for i in range(numberFeatures):
                    gen.append(random.randint(0, 1))

        return icls(gen)

    @staticmethod
    def mutation(individual):
        numberParamer = random.randint(0, len(individual) - 1)

        if numberParamer == 0:
            listKernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = listKernel[random.randint(0, 3)]

        elif numberParamer == 1:
            k = random.uniform(0.1, 100)
            individual[1] = k

        elif numberParamer == 2:
            individual[2] = random.uniform(0.1, 5)

        elif numberParamer == 3:
            gamma = random.uniform(0.01, 1)
            individual[3] = gamma

        elif numberParamer == 4:
            coeff = random.uniform(0.1, 1)
            individual[4] = coeff

        elif numberParamer == 5:
            tol = random.uniform(1e-6, 1e-1)
            individual[5] = tol

        elif numberParamer == 6:
            individual[6] = False if individual[6] else False

        elif numberParamer == 7:
            individual[7] = False if individual[7] else False

        elif numberParamer == 8:
            listDec = ["ovo", "ovr"]
            individual[8] = listDec[random.randint(0, 1)]

        else:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return SVC(
            kernel=individual[0], C=individual[1],
            degree=individual[2], gamma=individual[3],
            coef0=individual[4], random_state=101
    )
