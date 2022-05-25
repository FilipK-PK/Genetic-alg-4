from deap import base, creator, tools
from src import algorithm

MATE = {
    '1-point': tools.cxOnePoint,
    '2-points': tools.cxTwoPoint,
    'jednolite': tools.cxUniform
}

SELECT = {
    'turniej': tools.selTournament,
    'ruletki': tools.selRoulette,
    'losowa': tools.selRandom,
    'najlepsi': tools.selBest,
    'najgorsi': tools.selWorst
}


class ToolBox:
    def __init__(self, set_opt, len_column, y, df):
        self.__param = set_opt
        self.__len_col = len_column
        self.__y = y
        self.__df = df
        self.__toolbox = base.Toolbox()

        self.__init_max()
        self.__set_ind()
        self.__set_pop()
        self.__set_eval()
        self.__set_select()
        self.__set_mutate()
        self.__set_cross()

    def get_opt(self):
        return self.__toolbox

    def __init_max(self):

        creator.create(
            "FitnessMax", base.Fitness, weights=(1.0,)
        )
        creator.create(
            "Individual", list, fitness=creator.FitnessMax
        )

    def __set_ind(self):
        self.__toolbox.register(
            'individual', algorithm.SVCParameters,
            self.__len_col, creator.Individual
        )

    def __set_pop(self):
        self.__toolbox.register(
            "population", tools.initRepeat,
            list, self.__toolbox.individual
        )

    def __set_eval(self):
        self.__toolbox.register(
            "evaluate",
            algorithm.SVCParametersFitness,
            self.__y, self.__df,
            self.__len_col
        )

    def __set_mutate(self):
        self.__toolbox.register(
            "mutate",
            algorithm.mutationSVC
        )

    def __set_cross(self):
        self.__toolbox.register(
            "mate", MATE[self.__param['cross']]
        )

    def __set_select(self):
        if self.__param['select'] == 'turniej':
            self.__toolbox.register(
                "select", tools.selTournament,
                tournsize=int(
                    self.__param['len_popu']
                    * self.__param['p_select']
                )
            )
        else:
            self.__toolbox.register(
                "select",
                SELECT[self.__param['select']]
            )
