from sklearn.tree import DecisionTreeClassifier
import random


c_list = ['gini', 'entropy']
s_list = ['best', 'random']


class Tree:

    @staticmethod
    def generate(numberFeatures, icls, use_col):
        gen = [
            c_list[random.randint(0, 1)],
            s_list[random.randint(0, 1)],
            random.randint(5, 25),
            random.randint(2, 5),
            random.randint(1, 4),
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
            k = s_list[random.randint(0, 1)]
            individual[numberParamer] = k

        if numberParamer == 2:
            k = random.randint(5, 25)
            individual[numberParamer] = k

        if numberParamer == 3:
            k = random.randint(2, 5)
            individual[numberParamer] = k

        if numberParamer == 4:
            k = random.randint(1, 4)
            individual[numberParamer] = k

        if numberParamer > 4:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return DecisionTreeClassifier(
            criterion=individual[0], splitter=individual[1],
            max_depth=individual[2], min_samples_split=individual[3],
            min_samples_leaf=individual[4], random_state=101
        )
