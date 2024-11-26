# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:25:42 2022

@author: MrBinh
"""
import numpy as np

def getRateTransData(channel_banwidth, pr, distance, path_loss_exponent, sigmasquare):
    return (channel_banwidth * np.log2(
            1 + pr / np.power(distance,path_loss_exponent) / sigmasquare
        )
    ) 
