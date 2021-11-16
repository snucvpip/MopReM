import numpy as np

def YUV2RGB(y, u, v):
    r = y -3.94e-5*u +1.14 *v
    g = y -0.394*u -0.581*v
    b = y +2.032*u -4.81e-4*v
    return r, g, b

def RGB2YUV(r, g, b):
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.147*r - 0.289*g + 0.436*b
    v = 0.615*r - 0.515*g - 0.100*b
    return y, u, v