from sklearn.neighbors import KNeighborsClassifier
import random


listKernel = ['auto', 'ball_tree', 'kd_tree', 'brute']
listWeight = ['uniform', 'distance']


class KNeighbors:
    @staticmethod
    def generate(numberFeatures, icls, use_col):
        gen = [
            random.randint(2, 8),
            listWeight[random.randint(0, 1)],
            listKernel[random.randint(0, 3)],
            random.randint(15, 45),
            random.randint(1, 2),
        ]

        if use_col:
            for i in range(numberFeatures):
                gen.append(random.randint(0, 1))

        return icls(gen)

    @staticmethod
    def mutation(individual):
        numberParamer = random.randint(0, len(individual) - 1)

        if numberParamer == 0:
            k = random.randint(2, 8)
            individual[numberParamer] = k

        if numberParamer == 1:
            k = listWeight[random.randint(0, 1)]
            individual[numberParamer] = k

        if numberParamer == 2:
            k = listKernel[random.randint(0, 3)]
            individual[numberParamer] = k

        if numberParamer == 3:
            k = random.randint(15, 45)
            individual[numberParamer] = k

        if numberParamer == 4:
            k = random.randint(1, 2)
            individual[numberParamer] = k

        if numberParamer > 4:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return KNeighborsClassifier(
            n_neighbors=individual[0], weights=individual[1],
            algorithm=individual[2], leaf_size=individual[3],
            n_jobs=individual[4]
        )
