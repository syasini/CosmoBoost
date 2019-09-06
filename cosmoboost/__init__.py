import os
import sys
COSMOBOOST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, COSMOBOOST_DIR)
from .cosmoboost import *
from lib.MatrixHandler import mL2indx

