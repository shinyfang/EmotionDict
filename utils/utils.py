import os
from scipy import spatial
import scipy.stats
import numpy as np
import math

def compute_distance(predict, label):
    # distance turn
    # 'chebyshev'	'clark'	'canberra'	'kldist'	'cosine'	'intersection'

    # 1. chebyshev
    chevb = spatial.distance.chebyshev(label, predict)

    # 2. clark
    clark = (predict - label) / (predict + label)
    clark = math.sqrt(np.sum(clark * clark))

    # 3. canberra
    canb = spatial.distance.canberra(label, predict)

    # 4. kldist
    kl = np.sum(label * np.log(label / predict)) 

    # 5.consine
    consine = 1- spatial.distance.cosine(label, predict)

    # 6.intersection
    inter = np.sum(np.minimum(label, predict))


    result = np.array([chevb, clark, canb, kl, consine, inter]).astype(float)
    
    return result
