import cv2 as cv
import numpy as np
import math

def gauss_matrix(size, sigma):
    
    a = size // 2
    b = size // 2
    
    multiplier = 1.0 / (2 * math.pi * sigma**2)
    ker = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            
            dx = i - a
            dy = j - b
            
            exp = -(dx*dx + dy*dy) / (2 * sigma**2)
            ker[i, j] = multiplier * math.exp(exp)
    return ker