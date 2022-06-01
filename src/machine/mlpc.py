from sklearn.neural_network import MLPClassifier
import random


a_list = ['identity', 'logistic', 'tanh', 'relu']
s_list = ['lbfgs', 'sgd', 'adam']


class Mlpc:

    @staticmethod
    def generate(numberFeatures, icls, use_col):
        gen = [
            a_list[random.randint(0, 3)],
            s_list[random.randint(0, 2)],
            random.uniform(1e-5, 1e-2),
            random.randint(200, 400),
            True if random.randint(1, 4) == 1 else False
        ]

        if use_col:
            for i in range(numberFeatures):
                gen.append(random.randint(0, 1))

        return icls(gen)

    @staticmethod
    def mutation(individual):
        numberParamer = random.randint(0, len(individual) - 1)

        if numberParamer == 0:
            k = a_list[random.randint(0, 3)]
            individual[numberParamer] = k

        if numberParamer == 1:
            k = s_list[random.randint(0, 2)]
            individual[numberParamer] = k

        if numberParamer == 2:
            k = random.uniform(1e-5, 1e-3)
            individual[numberParamer] = k

        if numberParamer == 3:
            k = random.randint(200, 400)
            individual[numberParamer] = k

        if numberParamer == 4:
            k = True if random.randint(0, 1) == 1 else False
            individual[numberParamer] = k

        if numberParamer > 4:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return MLPClassifier(
            activation=individual[0], solver=individual[1],
            alpha=individual[2], max_iter=individual[3],
            shuffle=individual[4], random_state=101
        )
