from sklearn.gaussian_process import GaussianProcessClassifier
import random


o_list = ['fmin_l_bfgs_b', 'callable']
m_list = ['one_vs_rest', 'one_vs_one']


class Gauss:

    @staticmethod
    def generate(numberFeatures, icls, use_col):
        gen = [
            o_list[random.randint(0, 1)],
            random.randint(50, 150),
            True if random.randint(0, 1) == 1 else False,
            m_list[random.randint(0, 1)]
        ]

        if use_col:
            for i in range(numberFeatures):
                gen.append(random.randint(0, 1))

        return icls(gen)

    @staticmethod
    def mutation(individual):
        numberParamer = random.randint(0, len(individual) - 1)

        if numberParamer == 0:
            k = o_list[random.randint(0, 1)]
            individual[numberParamer] = k

        if numberParamer == 1:
            k = random.randint(50, 150)
            individual[numberParamer] = k

        if numberParamer == 2:
            k = True if random.randint(0, 1) == 1 else False
            individual[numberParamer] = k

        if numberParamer == 3:
            k = m_list[random.randint(0, 1)]
            individual[numberParamer] = k

        if numberParamer > 3:
            if individual[numberParamer] == 0:
                individual[numberParamer] = 1
            else:
                individual[numberParamer] = 0

    @staticmethod
    def fun_opt(individual):
        return GaussianProcessClassifier(
            optimizer=individual[0], max_iter_predict=individual[1],
            warm_start=individual[2], multi_class=individual[3],
            random_state=101
        )
