#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 /*************************************************************************
 *
 * EDGE'S FUND CONFIDENTIAL
 * __________________
 *
 *  [2020] - [2021] Edge's Fund LLC
 *  All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Edge's Fund LLC and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Edge's Fund LLC
 * and its suppliers and may be covered by U.S. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Edge's Fund LLC. Products derived from the technology
 * are subject to the same domain governing the software and content.
 **************************************************************************/
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~
            Authored by Julian Henry starting April 1, 2021.
                ->juliennedpotato@protonmail.com<-
                        ALL RIGHTS RESERVED
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~

"""
# import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from cooler import call_tenor
import pandas as pd
from typing import List

def gridder(start:float, n:int=10, step:int=10) -> List[float]:
    n=step=10
    '''
    Can be used fractally for parameter search
    converts a parameter to list of possible nearby values
    :param y:
    :param n:
    :return:
    '''
    return list(filter(lambda z:z>0,
           [start + ix*(start/step) - n//2 for ix in range(n)]))


def my_custom_loss_func(ground_truth, predictions):
    if len(ground_truth) != len(predictions):
        raise Exception("gt=ps")
    return np.sqrt(
        sum([(a - b) ** 2 for a, b in zip(ground_truth, predictions)])
        / len(ground_truth)
    )


class scibao:
    OPEN_INTEREST = 3e4
    verj = 0
    def __init__(self,filename=None):
        if not filename:filename = "csvs/VIX_April_16_2021_de_may_19_2021.csv"
        df = pd.read_csv(filename)
        market_strikes  = df["Strike"].values.tolist()
        mkt_vals = df["Last Price"].values.tolist()
        #TODO: limits to OPEN_interest e.g. 81 was the cutoff
        self.strike_vals = list(filter(lambda o: o < 81, market_strikes))
        # cutoff to equal market_strikes
        self.mkt_vals = mkt_vals[: len(self.strike_vals)]

    def score(self,**kwargs):
        ks = self.strike_vals
        mkt_vals = self.mkt_vals
        callpreds = []
        # Iterate over strike prices
        # appraise call tenor via bao
        for k in ks:
            kwargs.update({"K": k})
            callpreds += [call_tenor(**kwargs)]
        # normalize
        callpreds = list(map(lambda n: n * (mkt_vals[0] / callpreds[0]), callpreds))

        return my_custom_loss_func(callpreds, mkt_vals)


    @staticmethod
    def compare(strikes,market_values,call_predictions):
        plt = None
        plt.plot(strikes, call_predictions, label="model"+str(scibao.verj))
        plt.plot(strikes, market_values, label="market")
        plt.xlabel("strike"),
        plt.ylabel("call value"),
        plt.title("ráfaga"),
        plt.legend()
        plt.show()

VOL_ROLLING_30 = np.std(
        [
            28.57,
            24.66,
            25.47,
            24.03,
            22.56,
            21.91,
            20.69,
            20.03,
            19.79,
            19.23,
            21.58,
            20.95,
            18.88,
            20.30,
            21.20,
            19.81,
            18.86,
            20.74,
            19.61,
            19.40,
            17.33,
            17.91,
            18.12,
            17.16,
            16.95,
            16.69,
            16.91,
            16.65,
            16.99,
            16.57,
        ]
    )  # cbauto
import random
from heapq import heappop, heappush

def evalu(A=None,B=None,C=None,D=None,):
    return 3*A - 4*B + C/2 - D

if __name__ == "__main__":
    params = {chr(k): k for k in range(65,69)}
    best_choice = []
    param_grid = {k: gridder(v, n=3) for k, v in params.items()}

    for e in range(10):
        for k, guesses in param_grid.items():
            eval_params = params.copy()
            for g in guesses:
                eval_params[k]=g
                print(eval_params,loss:=abs(evalu(**eval_params)))
                heappush(best_choice, (loss,tuple(eval_params)))
    print(heappop(best_choice))
    random.seed(0)
    # raise Exception
    fixto = {
        "t": 0,
        "T": 30,
        "r_s": 0.0167, #FED'S RATE
        "kappa": 4.27, #FROM BAO
        "theta": 3.3, #FROM BAO
        "sigma": 2, #FROM BAO
        "K": 17,  # could be autogenerated (first ITM option april 19 was 17)
        "VIX_t": 16.57,  # cbauto
        "V_t": VOL_ROLLING_30,
    }
    hyper_params = {
        "rho": 0.2,  # what is this???
        "phi": 1,  # what is this???
        "kappa_v": 1.68,  # calibratable
        "theta_v": 1.11,  # calibratable
        "sigma_v": 2,  # calibratable
    }
    params = dict()
    params.update(**fixto);params.update(**hyper_params)
    param_grid = {
        k: v for k, v in zip(hyper_params, map(lambda q: gridder(q, n=7), hyper_params.values()))
    }
    [print(i,j)for i,j in param_grid.items()]
    start=dt.datetime.utcnow()
    og = fixto.copy()
    work = 32
    for i in range(work):
        # epoch
        print('>Epoch',i+1)
        # in each epoch, the loss minimizer determines which
        # parameter adjustment improves the most
        # of course we are grasping for machine learning here
        # but optimizing naively and settling to
        # local minimum
        loss_minimizer = int(1e10)
        best_param = (None,None)
        for key, v in list(param_grid.items()):
            # try k, v[len(v)//2+-
            print(key,'@',end='')
            for guess in v[(a:=len(v)//2)-1:a+2]:
                paramz = params.copy()
                paramz.update(dict(k=guess))
                print(guess,end=',')
                # BAO = scibao()
                # if loss := BAO.score(**paramz) < loss_minimizer:
                #     loss_minimizer = loss
                #     best_param = (key, guess)
                best_param = (key,guess)
        print()
        print(best_param, 'yielded most accuracy. Recalibrating...')
        key,guess = best_param
        # params.update(dict(key=guess));param_grid.update(dict(key))

        '''
                        # compute rmse
                        BAO = scibao()
                        mse = BAO.score(best_yet= min_paramwise[
                            "minscore"
                        ],**fixto) #prints on new best yet
                        if (mse < min_paramwise[
                            "minscore"
                        ]):
                            print('*'*32)
                            print('new low-score!',round(mse,4),)
                            print(fixto)
                            min_paramwise.update({"minscore": mse})
                            min_paramwise.update({"winning_params": fixto})
                            
                        '''
    # clf = GridSearchCV(scibao,param_grid=_param_grid)
    # clf.fit(None,None)







# callpreds= list(map(lambda o: callpreds[0]*(o-callpreds[0] )/o, callpreds))
# callpreds = list(
#     map(
#         lambda n: mkt_vals[0]
#         * (
#             (
#                 callpreds[0]
#                 * (1 - callpreds[0] / ( mkt_vals[0]))
#             )
#             ** -1
#         )
#         * callpreds[0]
#         * (1 - callpreds[0] / (n * mkt_vals[0] / callpreds[0])),
#         callpreds,
#     )
# )