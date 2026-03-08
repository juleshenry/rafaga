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
 * from Edge's Fund LLC
 **************************************************************************/
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~
            Authored by Julian Henry starting April 1, 2021.
                ->juliennedpotato@protonmail.com<-
                        ALL RIGHTS RESERVED
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~

"""
import numpy as np
from math import *
from scipy.stats import norm
import time
import datetime as dt
from abc import ABC, abstractmethod


class Zen(ABC):
    @abstractmethod
    def confess(self):
        pass


def scipy_calc(func):
    def g(*args, **krgs):
        print(f"{func.__name__} computed in scipy_calc")
        return func(*args, **krgs)

    return g


# @scipy_calc
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


# Black Scholes Function
def black_scholes_vanilla(s, x, tau, sigma, r_t, q_t, call=1):
    if call:
        return
    else:
        return


def time_t_bs(S_t, X, tau, sigma, r_t, q_t) -> float:
    """
    Calculates the time_t black_scholes of a put
    :param S_t: Asset value at t
    :param X: Strike price
    :param tau: tenor of option given by T-t
    :param sigma: implied vol.
    :param r_t: the risk-free rate
    :param q_t: time-t continuously compounded cash flow yielded by asset
    :return: value of Call option
    """
    # ans = S_t * (np.e ** (-q_t * tau)) * phi(
    #     (np.log(S_t / X) + (r_t - q_t + (sigma ** 2)/2) * tau)
    #     / (sigma * np.sqrt(tau))
    # ) - X * (np.e ** (-r_t * tau)) * phi(
    #     (np.log(S_t / X) + (r_t - q_t - (sigma ** 2)/2) * tau)
    #     / (sigma * np.sqrt(tau))
    # )
    asset_value = S_t * pow(np.e,-q_t*tau)
    strike_value = - X * pow(np.e,-r_t*tau)
    d = (np.log(S_t/X) + (r_t - q_t)*tau)/(sigma*np.sqrt(tau))
    vol_factor = (sigma * np.sqrt(tau))/2
    d1 = d + vol_factor
    d2 = d - vol_factor
    return asset_value * phi(d1) + strike_value * phi(d2)

import random
def iterative_implied_volatility(value, S_t, X, tau, r_t, q_t,epsilon = 10e-5):
    upper_bound, lower_bound = 1000, 0
    est_implied_vol = .6
    while(abs((est_value:=time_t_bs(S_t,X,tau,est_implied_vol,r_t,q_t)) - value) > epsilon):
        print(est_implied_vol,est_value,value)
        print(upper_bound,lower_bound)
        if est_value > value:
            upper_bound = est_implied_vol
            est_implied_vol -= random.uniform(lower_bound,upper_bound)
        else:
            lower_bound = est_implied_vol
            est_implied_vol += random.uniform(lower_bound,upper_bound)
        print(est_implied_vol, est_value, value)
        print(upper_bound, lower_bound)
        print('*'*55)
    print('final_error',time_t_bs(S_t, X, tau, est_implied_vol, r_t, q_t) - value)
    return est_implied_vol

def historical_volatility(values):return np.std(values)


def odd_mod(o, x):
    # up and down modulus
    return (x - o if o % (2 * x) > x else o) % x


def progreso(index, string, live=0):
    i = odd_mod(index, len(string))
    try:
        string = string[0:i] + ordem(string[i]) + string[i + 1 : len(string)]
        world = chr(127757 + (i // (len(string) // 3 + 1)) % 3)
        print(
            "\t" + f"{(world)}{string}{world}",
            end="\r",
        )
    except:
        pass
    if not live:
        print(f"O tempo é {dt.datetime.utcnow()} UTC, cara." + " " * 24)
        print()


def ordem(c: chr):
    return (
        c
        if ord(c) < 66
        else chr(ord(c) + 32)
        if ord(c) < 97
        else (
            (chr(ord(c) + 32) if ord(c) > 192 + 32 else chr(ord(c) - 32))
            if ord(c) > 191
            else chr(ord(c) - 32)
        )
    )


def intro(scale):
    tit = "Bem Vindo à Ráfaga : Onde Sonhos Se Realizam"
    for i in range(len(tit) * (scale)):
        progreso(i, tit, live=i + 1 - len(tit) * scale)
        time.sleep(0.08 * 2 ** (-i // len(tit)))

def interpolated_volatility(exercise_price):
    """
    A metric for market value of options
    :param exercise_price:
    :return:
    """
    sigma__t_X_tau = .1
    return sigma__t_X_tau

def call_valuation_function(S_t, X, tau, r_t, q_t, exercise_price):
    # sigma_t_X_tau is the interpolated implied volatility corresponding to the given exercise price
    time_t_bs(S_t, X, tau, interpolated_volatility(exercise_price), r_t, q_t)



from matplotlib import pyplot as plt


class Picasso(Zen):
    def confess(self):
        print()
        xs = np.array([z / 10 for z in range(-25, 25, 1)]).tolist()
        ys = [phi(x) for x in xs]
        plt.plot(xs, ys)
        plt.show()


class Phase1:
    # Converting CBOE prices to risk-free volatility
    # OR
    # Processing SP500 data to csv to compute Vainilha Precio da Opção
    pass


class InterpolotedSmile:
    pass


class CallValuationFunction:
    pass


class CumDistFunc:
    pass


class ProbDensFunc:
    pass


class Phase2:
    InterpolotedSmile()
    pass


class Phase3:
    def __init__(julian_henry, *args, **kw):
        julian_henry.headers = (
            "X/S   |X|Volatility|Call value|Delta|Vega|Lower bound|Upper bound|PI_t(x)"
        )
        ranges = [
            1 + np.sign(a - 4) * 2 ** (abs(a - 3) if a < 5 else abs(a - 5)) * 0.025
            for a in range(9)
        ]
        # print(ranges)
        julian_henry.knowledge = [
            [k if not i else "☁" for i, j in enumerate(julian_henry.headers.split("|"))]
            for k in ranges
        ]
        julian_henry.confesar()

    def confesar(julian_henry):
        header = "\t".join(julian_henry.headers.split("|"))
        start = "⠈" * len(header.replace("\t", "______"))
        print(start)
        print("⠈  ", header, "  ⠈")

        for i in julian_henry.knowledge:
            print("⠈  ", end="")
            print(
                "\t\t".join(
                    map(
                        lambda s: s if type(s) == str else "{0:.3f}".format(float(s)), i
                    )
                )
            )
            print("⠈  ", end="\n")
        print(start)

    # Gets all `Phase` gathered in 1-index dict


def get_phases():

    return [Phase1, Phase2, Phase3]


if __name__ == "__main__":
    SPEED = 16
    intro(SPEED)
    phases = [
        "interpolating and extrapolating volatility smile data using clamped cublic spline",
        "applying the call_valuation_function and taking bs-vol. to get option value",
        "differencing the call valuation viz. exercise price to approximatee risk-neutral cdf",
    ]

    # args = [83.11, 80, 1 / 365, .6, 0.25, 0]
    # print(args,'\n',time_t_bs(*args),"~3.37")
    # args = [83.11, 80, 1 / 365, 0.3, 0.25, 0]
    # print(args,'\n',time_t_bs(*args),"~3.14")
    # args = [83.11, 80, 1 / 365, 0.541, 0.25, 0]
    # print(args, '\n', time_t_bs(*args), "~3.23")
    # print(iterative_implied_volatility(3.23,83.11, 80, 1 / 365,0.25, 0))
    # x = time_t_bs(83.11, 80, 1/365, 0.541, 0.025, 0)
    # print(x)
    # x = time_t_bs(83.11, 80, 1/365, 0.541, 0.25, 0)
    # print(x)
    for n, phase in enumerate(get_phases()):
        print(f"{n+1}. {phase.__name__} {phases[n]}")
        print(phi(1+2/3))
        phase()

    # T = 100
    # t = 90
    # r_t = 0.01
    # tau = T - t # tenor
    # X = 10 # call struck at X
    # payoff_maturity(t,X,r_t, tau)
