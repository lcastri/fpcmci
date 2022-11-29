from enum import Enum

SOURCE = 'source'
SCORE = 'score'
PVAL = 'pval'
LAG = 'lag'

DASH = '-' * 55
RESFOLDER_DEFAULT = "%Y-%m-%d_%H-%M-%S"
SEP = "/"
RES_FILENAME = "res.pkl"
DAG_FILENAME = "dag.png"
TSDAG_FILENAME = "ts_dag.pdf"
DEP_FILENAME = "dependency.pdf"
LOG_FILENAME = "log.txt"
RESULTS_FILENAME = "results"

class LabelType(Enum):
    Lag = "Lag"
    Score = "Score"
    NoLabels = "NoLabels"