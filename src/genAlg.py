from deap import tools
from src import statistic
from src.toolbox import ToolBox

import pandas as pd
import random


class GenAlg:
    def __init__(self, list_opt):
        self.__list_opt = list_opt
        self.__statistic = statistic.Statistic()

    def run(self):
        self.__load_data()
        self.__set_toolbox()
        self.__find_el()

    def __find_el(self):

        pop = self.__toolbox.population(n=self.__list_opt['len_popu'])
        fitnesses = list(map(self.__toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for i in range(self.__list_opt['epoch']):
            print("epoka nr:", i+1, end=" ")

            """len_sel = int(self.__list_opt['popu'] * self.__list_opt['proc_sel'])
            offspring = self.__toolbox.select(pop, len_sel)
            offspring = [offspring[random.randint(0, len_sel-1)] for _ in range(self.__list_opt['popu'])]"""

            offspring = self.__toolbox.select(pop, self.__list_opt['len_popu'])
            offspring = list(map(self.__toolbox.clone, offspring))

            listElitism = []
            for x in range(self.__list_opt['elit']):
                listElitism.append(tools.selBest(pop, 0)[0])

            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < self.__list_opt['p_cross']:
                    self.__toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.__list_opt['p_mutate']:
                    self.__toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.__toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring + listElitism
            fits = [ind.fitness.values[0] for ind in pop]

            best = tools.selBest(pop, 1)[0]
            print("best:", best.fitness.values[0], best)
            self.__statistic.put_epoch(fits, pop)

        print(self.__statistic.get_best_param())
        self.__print_statistic()

    def __print_statistic(self):
        data = self.__statistic.get_statistic()
        print()
        print('max end', data['max'][-1])
        print('max global', max(data['max']))

        self.__statistic.draw_graphs()

    def __load_data(self):
        self.__df = pd.read_csv('data/dataR2.csv')
        self.__y = self.__df['Classification']
        self.__df.drop('Classification', axis=1, inplace=True)

    def __set_toolbox(self):
        toolbox = ToolBox(
            self.__list_opt, len(self.__df.columns),
            self.__y, self.__df
        )
        self.__toolbox = toolbox.get_opt()
