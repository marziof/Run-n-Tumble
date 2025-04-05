import numpy as np

def update_tumble_linear(p, delta_loss, a=0.3): 
    """ p: previous prob; delta_loss: difference between new position and old position"""
    # a = parameter in (0,1)
    if delta_loss >= 0:
        proba = min(1, p + (1-p)*a)
    else:
        proba = max(0, p*(1-a))
    return proba


def update_tumble_tanh(p, delta_loss, a=2250): 
    """ p: previous prob; delta_loss: difference between new position and old position"""
    s = delta_loss*a
    proba = 1/2 + 0.4*np.tanh(s)
    return proba