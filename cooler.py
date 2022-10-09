#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

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
import scipy
from scipy import integrate
from scipy.special import hyp1f1 as M, hyperu as U
from scipy.stats import norm
import datetime as dt
from cool import risk_neutral_opportunity_cost, TradingCal
import matplotlib.pyplot as plt
import pytz
from math import log10 as l10, floor as flr
import warnings
warnings.simplefilter(action='ignore',category=scipy.ComplexWarning)
warnings.simplefilter(action='ignore',category=integrate.IntegrationWarning)
COMP = _comp = True

_i = complex(0, 1)
_e = np.e
_inf = float('inf')


def A_bao(s, **kwargs):
    """
    TODO:
    TODO: FIX THIS AND THE MONEY FLOWS.
    TODO:
    AKA A(t;s)
    :param kwargs:
    :return:
    """
    kappa = kwargs["kappa"]
    theta_v = kwargs["theta_v"]
    sigma_v = kwargs["sigma_v"]
    T = kwargs["T"]
    t = kwargs["t"]
    rho = kwargs["rho"]
    kappa_v = kwargs["kappa_v"]
    # remember, eventually, s is going to be a free variable and the mother function
    # will be a lambda s: me(s) s.t. integrate.quad(me,0,_inf) calculable
    # calculate
    step1 = C_bao(**kwargs)
    """
    TODO: this was done. why?
    CAVEAT: this should be theta_h
    not theta_v, but I am unsure
    how to calculate instanteneous theta backwards.
    Example flow once theta_h is defined.
    def theta_h(x):return x
    motion = lambda h: 4 * theta_h(h)
    from scipy import integrate
    ans,_ = integrate.quad(motion,0,9)
    """

    theta_h = theta_v
    motion = (
        lambda h:  theta_h * _e ** (-kappa * (T - h))
    )  # honestly, what is this term representing...?
    step2_integration, step2_err = integrate.quad(motion, t, T)
    step2 = _i * s * kappa *  step2_integration
    step3 = (
        (_i * s * rho * kappa_v * theta_v)
        / (sigma_v * kappa)
        * (1 - _e ** (-kappa * (T - t)))
    )

    # _inf = int(1e7)
    mag = lambda a:a
    if _comp:print('step1',step1,'step2',mag(step2),'step3',mag(step3))
    # C(t;s)+i*s*κ*Integral (...tau e ^(−kappa(T−h))theta_h dh − i *s* ρ* kappa_v* theta_v* [1−e^(−kappa(T−t))]
    return step1 + step2 - step3
from typing import Union

def B_bao(s, **kwargs):
    tau = kwargs["T"] - kwargs["t"]
    kappa = kwargs["kappa"]
    sigma_v = kwargs["sigma_v"]
    rho = kwargs["rho"]

    return D(**kwargs) - (_i * s * rho * _e ** (-kappa * (tau))) / sigma_v


def C_bao(**kwargs):
    tau = kwargs["T"] - kwargs["t"]
    kappa = kwargs["kappa"]
    kappa_v = kwargs["kappa_v"]
    theta_v = kwargs["theta_v"]
    sigma_v = kwargs["sigma_v"]
    phi = kwargs["phi"]
    rho = kwargs["rho"]

    # Preliminary calculations (redundant in D)
    b = 1 - kappa_v / kappa
    a = b * (np.sqrt(1 - rho ** 2) - rho) / (2 * np.sqrt(1 - rho ** 2))
    phi_ = lambda x: norm.pdf(x, 0, 1)
    z0 = phi_(sigma_v * np.sqrt(1 - rho ** 2)) / kappa
    z = _e ** (-kappa * tau) * z0
    g = (a * U(a + 1, b + 1, z0) + 0.5 * U(a, b, z0)) / (
        (a / b) * M(a + 1, b + 1, z0) - 0.5 * M(a, b, z0)
    )

    # Herein the two cases based on kappa
    if kappa_v // kappa != kappa_v / kappa:
        step1 = -(2 * kappa_v * theta_v) / (sigma_v ** 2)
        step2 = (phi_(sigma_v * np.sqrt(1 - rho ** 2)) * (1 - _e ** (-kappa * tau))) / (
            2 * kappa
        )
        step3 = np.log((g * M(a, b, z) + U(a, b, z)) / (g * M(a, b, z0) + U(a, b, z0)))
    else:  # Frankly rarer to occur
        raise NotImplementedError
    return step1 * (step2 + step3)


def D(**kwargs):
    tau = kwargs["T"] - kwargs["t"]
    kappa = kwargs["kappa"]
    kappa_v = kwargs["kappa_v"]
    sigma_v = kwargs["sigma_v"]
    phi = kwargs["phi"]
    rho = kwargs["rho"]
    _phi = lambda x: norm.pdf(x)
    # Preliminary calculations (redundant in D)
    b = 1 - kappa_v / kappa
    a = b * (np.sqrt(1 - rho ** 2) - rho) / (2 * np.sqrt(1 - rho ** 2))
    z0 = _phi(sigma_v * np.sqrt(1 - rho ** 2)) / kappa
    z = _e ** (-kappa * tau) * z0
    g = (a * U(a + 1, b + 1, z0) + 0.5 * U(a, b, z0)) / (
        (a / b) * M(a + 1, b + 1, z0) - 0.5 * M(a, b, z0)
    )

    # Equations
    # Herein the two cases based on kappa
    if kappa_v // kappa != kappa_v / kappa:
        step1 = (phi * np.sqrt(1 - rho ** 2) * _e ** (-kappa * tau)) / sigma_v
        step2 = 1 - (
            (g * 2 * a / b) * M(a + 1, b + 1, z) - 2 * a * U(a + 1, b + 1, z)
        ) / (g * M(a, b, z) + U(a, b, z))
        return step1 * step2
    else:
        raise NotImplementedError("Unknown")


def Pi_1(**kwargs):
    K = kwargs["K"]
    t = kwargs["t"]
    T = kwargs["T"]
    kappa = kwargs["kappa"]
    VIX_t = kwargs["VIX_t"]
    V_t = kwargs["V_t"]

    # Psi(t;s-_i)
    top = lambda s: _e ** (
        A_bao(s - _i, **kwargs)
        + B_bao(s - _i, **kwargs) * V_t
        + _i * s * _e ** (-kappa * (T - t)) * np.log(VIX_t)
    )
    # Psi(t;-_i)
    bot = _e ** (
        A_bao(-_i, **kwargs)
        + B_bao(-_i, **kwargs) * V_t
        - _e ** (-kappa * (T - t)) * np.log(VIX_t)
    )

    # Psi_1(t;s) = Psi(t;s-_i) / Psi(t;-_i)
    # psi(t;s)=exp{A(t;s)+B(t;s)Vt +ise^−κ(T−t)lnVIXt}
    eqn = lambda s: top(s) / bot * _e ** (_i * -s * np.log(K)) / (_i * s)
    #TODO: miscreant integration
    pi_calc, pi_err = integrate.quad(eqn, 0, _inf)
    pi1 = 0.5 + (1 / np.pi) * pi_calc.real
    if COMP:print('COMPONENT: pi 1', pi1)
    return pi1


def Pi_2(**kwargs):
    t = kwargs["t"]
    T = kwargs["T"]
    kappa = kwargs["kappa"]
    VIX_t = kwargs["VIX_t"]
    V_t = kwargs["V_t"]

    # Psi_2(t;s)
    # Psi(t;s)=exp{A(t;s)+B(t;s)Vt +ise^−κ(T−t)lnVIXt}

    psi_j = lambda s: _e ** (
        A_bao(s, **kwargs)
        + B_bao(s, **kwargs) * V_t
        + _i * s * _e ** (-kappa * (T - t)) * np.log(VIX_t)
    )
    #TODO: miscreant integration
    pi_calc, pi_err = integrate.quad(psi_j, 0, _inf,limit=69)
    pi2= 0.5 + (1 / np.pi) * pi_calc.real
    if COMP:print('COMPONENT: pi2',pi2)
    # raise Exception
    return pi2


def F_tenor(**kwargs):
    """
    VIX Future solved as psi(t,-i)
    :param kwargs:
    :return:
    """
    t = kwargs["t"]
    T = kwargs["T"]
    kappa = kwargs["kappa"]
    VIX_t = kwargs["VIX_t"]
    V_t = kwargs["V_t"]
    rnoc = VIX_t ** (_e ** (-kappa * (T - t)))
    f1_t = rnoc * _e ** (A_bao(-_i, **kwargs) + B_bao(-_i, **kwargs) * V_t)
    if COMP:print('COMPONENT: F_tenor', f1_t)
    return f1_t

def call_tenor(**kwargs):
    """
    Implements 5.31; call value calculation as function of tenor.
    """
    t = kwargs["t"]
    T = kwargs["T"]
    r_s = kwargs["r_s"]
    K = kwargs["K"]
    rnoc = risk_neutral_opportunity_cost(t, T, r_s)
    return rnoc * (F_tenor(**kwargs).real * Pi_1(**kwargs) - K * Pi_2(**kwargs))


def scaleflop(functor, param, seeds, nward, **kw):
    # eg. call_value, [1e-3,1e-2,1e-1],10
    seed_n = dict()
    for seed in seeds:
        for n in range(1, nward + 1):
            kw.update({param: seed * n})
            seed_n.update({seed: seed_n.get(seed, []) + [functor(**kw)]})
    for a, b in seed_n.items():
        print(a, b[:10])
        x = list([i * a for i in range(1, nward + 1)])
        print(a, "setting")
        fig, ax = plt.subplots()
        ax.plot(x, b)

        # ax.set(xlabel="time (s)", ylabel="voltage (mV)", title=f"{a} Setting ")
        ax.grid()
        plt.show()



def scaler(x):
    return 10 ** flr(l10(x))


def gridder(y, n=10):
    return [
        scaler(y),
        list(filter(lambda z: z > 1e-10, [y -  scaler(y) * n / 2 +  scaler(y) * o for o in range(n)])),
    ][1]


if __name__ == "__main__":
    filename = "csvs/VIX_April_16_2021_de_may_19_2021.csv"
    tz = pytz.timezone("Africa/Abidjan")
    now = dt.datetime.utcnow() + tz.utcoffset(dt.datetime.utcnow())
    hoy = dt.date(now.year, now.month, now.day)
    # SHALL: format is `de_$mt_$dy_$yr.csv`
    month, day, year = filename.split("de_")[1].split("_")
    # format
    month = dt.datetime.strptime(month, "%b").month
    year = year.split(".csv")[0]
    print(int(month), int(day), int(year))
    # datebuild
    expiry_date = dt.date(
        int(year),
        int(month),
        int(day),
    )  # could be autogenerated
    tenor = TradingCal.get_tenor(hoy, expiry_date)
    V_t = np.std(
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
    fixto = {
        "t": 0,
        "T": tenor,
        "r_s": 1.67,
        "kappa": 4.27,
        "theta": 3.3,
        "sigma": 2,
        "K": 17,  # could be autogenerated (first ITM option april 19 was 17)
        "VIX_t": 16.57,  # derived from 30-day average of vix
        "V_t": V_t,
    }
    params = {
        "rho": 0.2,  # calibratable
        "phi": 1,  # calibratable
        "kappa_v": 1.68,  # calibratable
        "theta_v": 1.11,  # calibratable
        "sigma_v": 2,  # calibratable
    }
    _param_grid = {
        k: v for k, v in zip(params, map(lambda q: gridder(q), params.values()))
    }

    print("Walltime=", dt.datetime.utcnow() - now)
