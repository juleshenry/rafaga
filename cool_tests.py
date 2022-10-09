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
 * the property of Edge's Fund Incorporated and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Edge's Fund LLC
 * and its suppliers and may be covered by U.S. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Edge's Fund LLC.
 **************************************************************************/
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~
            Authored by Julian Henry starting April 1, 2021.
                ->juliennedpotato@protonmail.com<-
                        ALL RIGHTS RESERVED
 ~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~!!!!~~~~
"""
import unittest
import datetime as dt
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np


class TradingCal:
    CAL = USFederalHolidayCalendar()
    CAL.rules.pop(5)
    CAL.rules.pop(5)


def get_tenor(dt1, dt2) -> 1:
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


class TestTenorMethods(unittest.TestCase):
    def test_simple_bus_day(self):
        # monday to tuesday
        self.assertEqual(get_tenor(dt.date(2021, 4, 5), dt.date(2021, 4, 6)), 1)

    def test_simple_bus_week(self):
        # monday to monday
        self.assertEqual(get_tenor(dt.date(2021, 4, 5), dt.date(2021, 4, 12)), 5)

    def test_simple_weekend(self):
        # sunday to monday
        self.assertEqual(get_tenor(dt.date(2021, 4, 9), dt.date(2021, 4, 12)), 1)

    def test_simple_holiday(self):
        # pre xmas to post xmas
        self.assertEqual(get_tenor(dt.date(2021, 12, 23), dt.date(2021, 12, 27)), 1)

    def test_whole_shebang(self):
        # pre xmas to post xmas
        self.assertEqual(get_tenor(dt.date(2021, 4, 9), dt.date(2022, 4, 10)), 252)
from typing import List,Callable

def emphasize_loss(xs:List[List[float]], ys:List[List[float]]
                   , verdades:List[float],loss:Callable=lambda x,y: np.sqrt((x-y) ** 2)):
    """
    xs = [[i for i in range(10)]for j in range(10)]
    ys = [[(k+1)*o**2 for o in range(10)]for k in range(10)]
    ascending_gradient(xs,ys,ys[0])
    :param xs:
    :param ys:
    :param verdades:
    :param loss:
    :return:
    """
    # compute min loss
    import sys

    mini, minindex = sys.maxsize, None
    for i, y in enumerate(ys):
        lost = sum(map(lambda a:loss(a[0],a[1]),  zip(y, verdades)))
        # print(lost,mini)
        if lost < mini:
            minindex = i
            mini = lost
    # Plotting both the curves simultaneously
    for i, xy in enumerate(zip(xs, ys)):
        print("@")
        alpha = min(1, abs(i - minindex) / len(xs) + 1 / len(xs))
        print(alpha)
        # alpha = i/len(xs)
        # print(alpha)
        x, y = xy
        plt.plot(x, y, color="r", label=f"Test_{i}", alpha=alpha)
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Angle")
    plt.ylabel("Magnitude")
    plt.title("Sine and Cosine functions")
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()


def ascending_gradient(xs, ys):
    # Plotting both the curves simultaneously
    for i, xy in enumerate(zip(xs, ys)):
        x, y = xy
        plt.plot(x, y, color="g", label=f"Test_{i}", alpha=i / len(xs))
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Angle")
    plt.ylabel("Magnitude")
    plt.title("Sine and Cosine functions")
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()


def test_calendar():
    suite = unittest.TestSuite()
    for test in [x for x in dir(TestTenorMethods) if x[:4] == "test"]:
        suite.addTest(TestTenorMethods(test))
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)


# from coolest import bcool

if __name__ == "__main__":
    # test_calendar()
    xs=[[i for i in range(10)]for j in range(10)]
    ys = [[(k+1)*o**2 for o in range(10)]for k in range(10)]
    vs = ys[0]
    emphasize_loss(xs,ys,vs)