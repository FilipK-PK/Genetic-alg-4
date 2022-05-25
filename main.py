from src.cui import Cui
from src.genAlg import GenAlg


if __name__ == '__main__':
    cui = Cui()
    cui.run()

    alg = GenAlg(cui.get_result())
    alg.run()
