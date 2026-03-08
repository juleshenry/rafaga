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
# driver = webdriver.Firefox()
# driver.get("http://www.python.org")
# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()
from selenium import webdriver,common
import datetime as dt
import os


class scritor:
    @classmethod
    def src(s, agua: str):
        agua += "\n"
        if not type(agua) == str:
            raise ValueError
        megusta = sorted([h for h in os.listdir() if h.startswith("pra_")])
        if not len(megusta):
            with open("pra_0.txt", "a") as a:
                a.write(agua)
        else:
            if os.path.getsize(megusta[-1]) > 2 ** 16:
                print(os.path.getsize(megusta[-1]), "too big")
                pranext = (
                    int("".join(filter(lambda x: x.isnumeric(), megusta[-1][:-4]))) + 1
                )
                with open(f"pra_{pranext}.txt", "a") as ss:
                    ss.write(agua)
            else:
                with open(megusta[-1], "a") as a:
                    a.write(agua)


DRIVER = webdriver.Firefox(executable_path="./geckodriver")
from selenium.webdriver.support.ui import Select
import time

class RunFFTests:
    def testMethod(self):
        # Initiate the driver instance
        DRIVER.get("https://finance.yahoo.com/quote/%5EVIX/options?p=%5EVIX")
        scritor.src(f"charging juice...{dt.datetime.utcnow()}")
        # Get VIX
        element = DRIVER.find_element_by_xpath(
            "/ html / body / div[1] / div / div / div[1] / div / div[2] / div / div / div[4] / div / div / div / div[3] / \
          div[1] / div / span[1]"
        )
        print("vix is ", element.text)
        vix_t = float(element.text)
        element = DRIVER.find_element_by_xpath('/html/body/div[1]/div/div/div[1]/div/'
                                               'div[3]/div[1]/div/div[2]/div/div/section/div/div[1]/select')
        print('Select which maturity to inspect, by number')

        print('\n'.join(list(map(lambda a:str(a[0])+'. '+str(a[1]),enumerate(element.text.split('\n'))))))
        decision = input()
        date_chosen = element.text.split('\n')[int(decision)]
        select = Select(element)
        # Now we have many different alternatives to select an option.
        select.select_by_index(int(decision))
        print('Getting data...\r')
        time.sleep(4) # wait to load new options data
        # Get Current Calls ITM
        calls = [
            ['Contract Name', 'Last Trade Date', 'Strike', 'Last Price', 'Bid', 'Ask', 'Change', '% Change', 'Volume',
             'Open Interest', 'Implied Volatility']]
        for i in range(1, 51):
            try:
                element = DRIVER.find_element_by_xpath(
                    # f'/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div/section/section[1]/div[2]/div/table/tbody/tr[{i}]')
                    f'/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/section/section[1]/div[2]/div/table/tbody/tr[{i}]')
                    # f'/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/section/section[1]/div[2]/div/table/tbody/tr[8]'
                table = element.find_elements_by_xpath(".//td")
                if float(table[2].text) > vix_t:  # itm
                    calls += [list(map(lambda a: a.text.replace(',',''), table))]
            except common.exceptions.NoSuchElementException as nsee:
                pass
        for c in calls:print(c)
        hoy = dt.datetime.today()
        with open(f"csvs/VIX_{vix_t}_{hoy.strftime('%B_%d_%Y').lower()}_de_{date_chosen.lower().replace(' ','_').replace(',','')}.csv","a") as e:
            for c in calls:
                e.write(','.join(c)+'\n')
        DRIVER.close()
        # print("1st itm call is ", element.text)
        #     scritor.src(element.text)
        # scritor.src(element.tag_name)
        # element.screenshot("jkj.png")


ff = RunFFTests()
ff.testMethod()

