import os
from pathlib import Path
from fpcmci.basics.constants import *


def cls():
    """
    Clear terminal
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def get_selectorpath(resfolder):
    """
    Return log file path

    Args:
        resfolder (str): result folder

    Returns:
        (str, str): log file path, dependency image file path
    """
    Path(SEP.join([RESULTS_FILENAME, resfolder])).mkdir(parents=True, exist_ok=True)
    return SEP.join([RESULTS_FILENAME, resfolder, LOG_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, DEP_FILENAME])


def create_results_folder():
    """
    Creates results folder if doesn't exist
    """
    Path(RESULTS_FILENAME).mkdir(parents=True, exist_ok=True)


def get_validatorpaths(resfolder):
    """
    Creates resfolder if doesn't exist

    Args:
        resfolder (str): result folder name

    Returns:
        (str, str, str): result.pkl file path, dag file path, ts_dag file path
    """
    Path(SEP.join([RESULTS_FILENAME, resfolder])).mkdir(parents=True, exist_ok=True)
    return SEP.join([RESULTS_FILENAME, resfolder, RES_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, DAG_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, TSDAG_FILENAME])