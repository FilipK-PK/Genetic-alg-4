from deap import base, creator, tools
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection, metrics

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
    def __init__(self, set_opt, len_column, y, df, class_opt):
        self.__class_opt = class_opt
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
            'individual', self.__class_opt.generate,
            self.__len_col, creator.Individual,
            self.__param['use_col'] == 'tak'
        )

    def __set_pop(self):
        self.__toolbox.register(
            "population", tools.initRepeat,
            list, self.__toolbox.individual
        )

    def __set_eval(self):
        self.__toolbox.register(
            "evaluate",
            self.__ParametersFitness,
            self.__y, self.__df,
            self.__len_col
        )

    def __set_mutate(self):
        self.__toolbox.register(
            "mutate",
            self.__class_opt.mutation
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

    def __ParametersFitness(self, y, df, numberOfAtributtes, individual):
        split = 5
        cv = model_selection.StratifiedKFold(n_splits=split)

        listColumnsToDrop = []
        for i in range(numberOfAtributtes, len(individual)):

            if individual[i] == 0:
                listColumnsToDrop.append(i - numberOfAtributtes)

        dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

        mms = MinMaxScaler()
        df_norm = mms.fit_transform(dfSelectedFeatures)

        estimator = self.__class_opt.fun_opt(individual)

        resultSum = 0

        for train, test in cv.split(df_norm, y):
            estimator.fit(df_norm[train], y[train])
            predicted = estimator.predict(df_norm[test])
            expected = y[test]
            tn, fp, fn, tp = metrics.confusion_matrix(
                expected, predicted
            ).ravel()
            result = (tp + tn) / (tp + fp + tn + fn)

            resultSum = resultSum + result

        return resultSum / split,
