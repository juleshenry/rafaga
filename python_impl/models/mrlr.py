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
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
from scipy import integrate
from scipy.stats import norm
import datetime as dt


def vix_future(vix_t, kappa, theta_s, sigma_s, T, t, verbose=False):
    tenor_drift = vix_t ** (np.e ** (-kappa * (T - t)))
    drift = lambda s: kappa * theta_s * np.e ** (-kappa * (T - s))
    calced_drift, drift_err = integrate.quad(drift, t, T)
    if verbose:
        print("drift_err", drift_err)
    volatility = lambda s: (sigma_s ** 2) * np.e ** (-2 * kappa * (T - s))
    calced_volatility, volatility_err = integrate.quad(volatility, t, T)
    if verbose:
        print("volatility_err", volatility_err)
    risk_neutral_dynamics = calced_drift + 0.5 * calced_volatility
    return tenor_drift * np.e ** risk_neutral_dynamics


def risk_neutral_opportunity_cost(t, T, r_s):
    """
    1st Component of Eqn 5.31
    :param t:
    :param T:
    :param r_s:
    :return:
    """
    opp_cost, err_o_c = integrate.quad(lambda s: r_s, t, T)
    return np.e ** (-opp_cost)


def calc_Pi(F_T_t, K, sigma_s, kappa, t, T, d=0, verbose=False):
    # returns cdf of distribution under MRLR assumption
    if d not in {1, 2}:
        raise NotImplementedError  # only two choices
    sigma_drift = lambda s: (sigma_s ** 2) * np.e ** (-2 * kappa * (T - s))
    calced_sigma_drift, s_d_err = integrate.quad(sigma_drift, t, T)
    if verbose:
        print("s_d_err", s_d_err)
    d_factor = (
        -1 if d == 2 else 1
    ) * 0.5  # there are exactly two forms for the MLRL model distribution formulas
    return (np.log(F_T_t / K) + d_factor * calced_sigma_drift) / np.sqrt(
        calced_sigma_drift
    )


def call_value(t, T, K, r_s, vix_t, kappa, theta_s, sigma_s):

    rnoc = risk_neutral_opportunity_cost(t, T, r_s)
    F_T_t = vix_future(vix_t, kappa, theta_s, sigma_s, T, t)
    Pi_1 = calc_Pi(F_T_t, K, sigma_s, kappa, t, T, d=1)
    Pi_2 = calc_Pi(F_T_t, K, sigma_s, kappa, t, T, d=2)
    return rnoc * (F_T_t * Pi_1 - K * Pi_2)


class TradingCal:
    CAL = USFederalHolidayCalendar()
    CAL.rules.pop(5)
    CAL.rules.pop(5)

    @staticmethod
    def get_tenor(dt1, dt2):
        if dt1 == dt2:
            return 0
        return (
            len(
                pd.date_range(
                    start=dt1, end=dt2, freq=CustomBusinessDay(calendar=TradingCal.CAL)
                )
            )
            - 1
        )


if __name__ == "__main__":

    def calibrate():
        K = 17  # Strike price of in-the-money call
        t = 0
        today = dt.date(2021, 4, 9)
        expiry_date = dt.date(2021, 5, 19)
        T = TradingCal.get_tenor(today, expiry_date)
        r_s = 0.0167  # Fed's lending rate
        vix_t = 16.69  # April 9,2021
        true_call_value = 4.3
        min_err = np.inf
        # Calibration
        range_view = range(5, 100)
        k_t_s = []
        for sigma in range_view:
            for theta in range_view:
                for k in range_view:
                    kappa, sigma_s, theta_s = k / 10, sigma / 10, theta / 10
                    try:
                        MRLR = call_value(t, T, K, r_s, vix_t, kappa, theta_s, sigma_s)
                        # TODO: halt at a specific efficacy?
                        # TODO: cache incalculable values. skip incalculable
                        # TODO: rolling printout of error
                        if abs(MRLR - true_call_value) < min_err:
                            print(min_err)
                            min_err = abs(MRLR - true_call_value)

                            k_t_s += [(kappa, theta_s, sigma_s)]
                            if min_err / true_call_value < 0.025:
                                return k_t_s
                    except:
                        # Incalculable
                        pass

    k_t_s = calibrate()
    t = 0
    today = dt.date(2021, 4, 9)
    expiry_date = dt.date(2021, 5, 19)
    T = TradingCal.get_tenor(today, expiry_date)
    r_s = 0.0167  # Fed's lending rate
    vix_t = 16.69  # April 9,2021
    print(
        "PASS 0",
        {param: val for param, val in zip(["kappa", "theta_s", "sigma_s"], k_t_s[-1])},
    )
    global_min_err = np.inf
    best = None
    real_calls = [
        (18, 3.7),
        (19, 3.3),
        (20, 2.95),
        (21, 2.6),
        (22, 2.27),
        (23, 2.07),
        (24, 1.9),
    ]
    for _k_t_s in k_t_s[int(len(k_t_s) * 0.25) :]:
        print("^" * 90)
        print(">>>", _k_t_s)
        kappa, theta_s, sigma_s = _k_t_s
        epsilon = 0
        # try:
        for K_trueval in real_calls:
            K, trueval = K_trueval
            epsilon += abs(
                call_value(t, T, K, r_s, vix_t, kappa, theta_s, sigma_s) - trueval
            )
        print(epsilon, "from", _k_t_s)
        if epsilon < global_min_err:
            best = _k_t_s
            global_min_err = epsilon
        # except:
        #     print(_k_t_s,"cannot work")
    print("BEST", best, global_min_err / len(real_calls))
    # range_view = range(1,11)
    # for k in range_view:
    #     for s in range_view:
    #         for t in range_view:
    #             kappa,sigma_s,theta_s = k_t_s[0]*k,k_t_s[1]*s,k_t_s[2]*t
    #             # print(kappa,sigma_s,theta_s)
    #             MRLR = call_value(t,T,K,r_s, vix_t, kappa, theta_s, sigma_s)
    #             if abs(MRLR - true_call_value) < min_err:
    #                 min_err = abs(MRLR - 4.30)
    #                 k_t_s = (kappa,theta_s,sigma_s)

    # print('PASS 1',k_t_s,min_err)
    # range_view = range(1, 11)
    # for k in range_view:
    #     for s in range_view:
    #         for t in range_view:
    #             kappa, sigma_s, theta_s = k_t_s[0] + k_t_s[0]/10 * k, k_t_s[1] + k_t_s[1]/10 * s, k_t_s[2] + k_t_s[2]/10 * t
    #             # print(kappa, sigma_s, theta_s)
    #             # try:
    #             MRLR = call_value(t, T, K, r_s, vix_t, kappa, theta_s, sigma_s)
    #             if abs(MRLR - true_call_value) < min_err:
    #                 min_err = abs(MRLR - 4.30)
    #                 k_t_s = (kappa, theta_s, sigma_s)
    #             # except:
    #                 # Incalculable
    #                 # pass
    # print('PASS 2', k_t_s, min_err)
