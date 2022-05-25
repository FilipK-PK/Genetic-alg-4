from src.cui import Cui
from src.genAlg import GenAlg


if __name__ == '__main__':
    #cui = Cui()
    #cui.run()

    #alg = GenAlg(cui.get_result())
    alg = GenAlg({'len_popu': 100, 'epoch': 10, 'select': 'najlepsi', 'p_select': 0.2,
                  'cross': '2-points', 'p_cross': 0.8, 'p_mutate': 0.2, 'elit': 0})
    alg.run()
