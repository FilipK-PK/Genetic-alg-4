from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection, metrics
from sklearn.svm import SVC
import random


def SVCParameters(numberFeatures, icls):
    listKernel = ["linear", "rbf", "poly", "sigmoid"]
    listDec = ["ovo", "ovr"]

    gen = [
            listKernel[random.randint(0, 3)],
            random.uniform(0.1, 100),
            random.uniform(0.1, 5),
            random.uniform(0.001, 5),
            random.uniform(0.01, 10),
            random.uniform(1e-6, 1e-1),
            True if random.randint(0, 1) == 1 else False,
            True if random.randint(0, 1) == 1 else False,
            listDec[random.randint(0, 1)],
            #random.randint(100, 400)

        ]

    for i in range(numberFeatures):
        gen.append(random.randint(0, 1))

    return icls(gen)


def SVCParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAtributtes, len(individual)):

        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAtributtes)

    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = SVC(
        kernel=individual[0], C=individual[1],
        degree=individual[2], gamma=individual[3],
        coef0=individual[4], tol=individual[5],
        shrinking=individual[6], probability=individual[7],
        decision_function_shape=individual[8], #cache_size=individual[9],
        random_state=101
    )

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


def mutationSVC(individual):
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

    #elif numberParamer == 9:
        #individual[9] = random.randint(100, 400)

    else:
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
