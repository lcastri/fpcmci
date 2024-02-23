from enum import Enum

SOURCE = 'source'
SCORE = 'score'
PVAL = 'pval'
LAG = 'lag'

DASH = '-' * 55
SEP = "/"
RES_FILENAME = "res.pkl"
DAG_FILENAME = "dag"
TSDAG_FILENAME = "ts_dag"
LOG_FILENAME = "log.txt"


class LabelType(Enum):
    Lag = "Lag"
    Score = "Score"
    NoLabels = "NoLabels"
    
    
class ImageExt(Enum):
    PNG = ".png"
    PDF = ".pdf"
    JPG = ".jpg"