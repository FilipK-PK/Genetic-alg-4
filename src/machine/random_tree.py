from sklearn.ensemble import RandomForestClassifier
import random


c_list = ['gini', 'entropy']


class RandomTree:

    @staticmethod
    def generate(numberFeatures, icls, use_col):

        gen = [
            c_list[random.randint(0, 1)],
            random.randint(50, 200),
            True if random.randint(0, 1) == 1 else False,
            random.randint(1, 5),
        ]

        if use_col:
            for i in range(numberFeatures):
                gen.append(random.randint(0, 1))

        return icls(gen)

    @staticmethod
    def mutation(individual):
        numberParamer = random.randint(0, len(individual) - 1)

        if numberParamer == 0:
            k = c_list[random.randint(0, 1)]
            individual[numberParamer] = k

        if numberParamer == 1:
            k = random.randint(50, 200)
            individual[numberParamer] = k

        if numberParamer == 2:
            k = True if random.randint(0, 1) == 1 else False
            individual[numberParamer] = k

        if numberParamer == 3:
            k = random.randint(1, 5)
            individual[numberParamer] = k

        if numberParamer > 3:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return RandomForestClassifier(
            criterion=individual[0], max_depth=individual[1],
            bootstrap=individual[2],
            min_samples_leaf=individual[3], random_state=101
        )
