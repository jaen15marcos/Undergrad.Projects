
import requests
import subprocess
import os
from bs4 import BeautifulSoup
import lxml
import numpy as np
import pandas as pd
from sympy import symbols, Sum, nsolve, exp
import math
import datetime 
import calendar
from IPython.display import display
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


url1 = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
html_text = requests.get(url1).text
soup = BeautifulSoup(html_text, "lxml")
tags = []
for x in soup.body.find_all('td', class_ ="table__td text-right"):
    tags.append(x.text.strip())
url2 = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2C3%2C4%2C16&coupon=&currency=184&rating=&country=19"
html_text = requests.get(url2).text
soup = BeautifulSoup(html_text, "lxml")
for x in soup.body.find_all('td', class_ ="table__td text-right"):
    tags.append(x.text.strip())
url3 = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=longterm&yield=&bondtype=2%2C3%2C4%2C16&coupon=&currency=184&rating=&country=19"
html_text = requests.get(url2).text
soup = BeautifulSoup(html_text, "lxml")
for x in soup.body.find_all('td', class_ ="table__td text-right"):
    tags.append(x.text.strip())
    
currency = [tags[i] for i in range(0,len(tags),7)]
cupon_rate = [tags[i] for i in range(1,len(tags),7)]
yield_ = [tags[i] for i in range(2,len(tags),7)]
rating = [tags[i] for i in range(3,len(tags),7)]
maturity_date = [tags[i] for i in range(4,len(tags),7)]
bid = [tags[i] for i in range(5,len(tags),7)]
ask = [tags[i] for i in range(6,len(tags),7)]

"""Now we create helping functions to make the information scrapped from the 
wb workable, and to check user's inputs. 
"""

for j in range(len(maturity_date)):
    maturity_date[j] = datetime.datetime.strptime(maturity_date[j], '%m/%d/%Y')
    cupon_rate[j] = float(cupon_rate[j].strip('%'))
    if type(ask[j]) is str and ask[j] != "-":
        ask[j] = float(ask[j])




def year_fraction(start_date, end_date):
    """Returns fraction in years between start_date and end_date, using Actual/Actual convention"""
    if start_date == end_date:
        return 0.0
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    
    start_year = start_date.year
    end_year = end_date.year
    year_1_diff = 366 if calendar.isleap(start_year) else 365
    year_2_diff = 366 if calendar.isleap(end_year) else 365

    total_sum = end_year - start_year - 1
    diff_first = datetime.datetime(start_year + 1, 1, 1) - start_date
    total_sum += diff_first.days / year_1_diff
    diff_second = end_date - datetime.datetime(end_year, 1, 1)
    total_sum += diff_second.days / year_2_diff

    return total_sum

for l in range(len(maturity_date)):
    maturity_date[l] = year_fraction(datetime.datetime.now(), maturity_date[l])


def check_cupon_rate_user(cupon_rate_user):
    while True:
        try: 
            cupon_rate_user = float(cupon_rate_user)
            while (cupon_rate_user <= 0 or cupon_rate_user >= 100):
                print("Cupon rate must be between 0 and a 100")
                cupon_rate_user = input('Please enter cupon rate of bond as an int or float:')
                cupon_rate_user = check_cupon_rate_user(cupon_rate_user) 
            return cupon_rate_user
        except ValueError as e:
          print("Please enter integers or floats")
        except Exception as e:
          print(e)


def check_price(user_price):
    try:
        user_price = float(user_price)
        while 0 >= user_price:
            print("Invalid number. The number must be gretaer than 0")
            user_price = (input('Please enter price of bond:'))
            user_price = check_price(user_price)
        return user_price
    except ValueError as e:
         print("Invalid number. The number must be gretaer than 0")
    except Exception as e:
          print(e)

def check_year(maturity_date_user1):
    maturity_date_user1 = str(maturity_date_user1)
    while True:
        try:
            maturity_date_user1.strip()
            if len(maturity_date_user1) == 2:
                maturity_date_user1 = "20" + maturity_date_user1[-2:]
            if (len(maturity_date_user1) != 2 and len(maturity_date_user1) !=4):
                print("Invalid Year was inputed, please revise year")
                while (len(maturity_date_user1) != 2 and len(maturity_date_user1) !=4):
                     maturity_date_user1 = (input('Please enter the year of maturity, ex 2023: '))
                     maturity_date_user1 = check_year(maturity_date_user1)
            if int(maturity_date_user1) - 2022 > 30.5:
                print("Year of maturity must be within 30 years")
                while (int(maturity_date_user1) - 2022 > 30.5):
                     maturity_date_user1 = (input('Please enter the year of maturity, ex 2023: '))
                     maturity_date_user1 = check_year(maturity_date_user1)
            return maturity_date_user1
        except ValueError as e:
            print("Please enter integers or str for the dates, date should be within 10y")
        except Exception as e:
            print(e)

def check_month(maturity_date_user2):
    maturity_date_user2 = str(maturity_date_user2)
    while True:
        try:
            maturity_date_user2.strip()
            if (0 >= int(maturity_date_user2) ) or  (int(maturity_date_user2) > 12):
                print("Month must be between 1 and 12")
                while ((0 >= int(maturity_date_user2) ) or ( int(maturity_date_user2) > 12)):
                     maturity_date_user2 = (input('Please enter the month of maturity, ex 8: '))
                     maturity_date_user2 = check_month(maturity_date_user2)
            return maturity_date_user2
        except ValueError as e:
            print("Please enter integers or str for the dates, date should be within 10y")
        except Exception as e:
            print(e)

def check_day(maturity_date_user3):
    maturity_date_user3 = str(maturity_date_user3)
    while True:
        try:
            maturity_date_user3.strip()
            if (0>= int(maturity_date_user3)) or (int(maturity_date_user3) > 31):
                print("Day of the month should be between 1 and 31")
                while ((0>= int(maturity_date_user3)) or (int(maturity_date_user3) > 31)):
                     maturity_date_user3 = (input('Please enter the day of maturity, ex 15: '))
                     maturity_date_user3 = check_day(maturity_date_user3)
            return maturity_date_user3
        except ValueError as e:
            print("Please enter integers or str for the dates, date should be within 10y")
        except Exception as e:
            print(e)

def working_maturity_date(maturity_date_user1, maturity_date_user2, maturity_date_user3):
            txt = "{}/{}/{}".format(maturity_date_user2,maturity_date_user3,maturity_date_user1)
            maturity_date_user = datetime.datetime.strptime(txt.strip(), '%m/%d/%Y')
            maturity_date_user = year_fraction(datetime.datetime.today(), maturity_date_user)
            return maturity_date_user



def main():
    cupon_rate_user = input('Please enter cupon rate of bond as an int or float: ')
    cupon_rate_user = check_cupon_rate_user(cupon_rate_user)
    maturity_date_user1 = (input('Please enter the year of maturity, ex 2023: '))
    maturity_date_user1 = check_year(maturity_date_user1)
    maturity_date_user2 = (input('Please enter the month of maturity, ex 8: '))
    maturity_date_user2 = check_month(maturity_date_user2)
    maturity_date_user3 = (input('Please enter the day of maturity, ex 15: '))
    maturity_date_user3 = check_day(maturity_date_user3)
    maturity_date_user = working_maturity_date(maturity_date_user1, maturity_date_user2, maturity_date_user3)
    user_price = (input('Please enter price of bond as an int or float: ')) 
    ytm = get_ytm(maturity_date_user,cupon_rate_user,user_price)*100
    print("The YTM of the bond you have inputed is: {}%".format(round(ytm,4)))
    spot_rate = get_spot_rate(maturity_date_user,user_price)*100
    print("The spot rate of the bond you have inputed is: {}%".format(round(spot_rate,4)))
    
          
def dis():
    trading_bonds = {}
    trading_bonds["Currency"] = currency[:]
    trading_bonds["Cupon Rate"] = cupon_rate[:]
    trading_bonds["Rating"] =  rating[:]
    trading_bonds["Maturity Date"] = maturity_date[:]
    trading_bonds["Bid"] = bid[:]
    trading_bonds["Ask"] = ask[:]
    ytm_bonds = []
    spot_rate = []
    x = []
    y = []
    for i in range(len(currency)):
        ytm_bonds.append(get_ytm(maturity_date[i],cupon_rate[i],ask[i]))
        spot_rate.append(get_spot_rate(maturity_date[i],ask[i]))
        maturity_date[i] = round(maturity_date[i],3)
        if type(ytm_bonds[i]) is not str:
            ytm_bonds[i] = round(ytm_bonds[i],4)*100
            y.append(ytm_bonds[i])
            x.append(maturity_date[i])
            spot_rate[i] = round(spot_rate[i],4)*100
    trading_bonds["YTM %"] = ytm_bonds[:]
    trading_bonds["Spot R %"] = spot_rate[:]
    df = pd.DataFrame(trading_bonds)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=4)
    plt.ylim(0,10)
    plt.xlim(0,10)
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.title('Yield Curve')
    plt.show()
    print("Check Plots in Console to see Yield Curve!  \n")
    return display(df)
    
dict = {} #easy way to find similar bonds by ordering them by cupon rate
for i in range(len(cupon_rate)):
    if str(cupon_rate[i]) not in dict.keys():
        dict[str(cupon_rate[i])] = [i]
    else:
        dict[str(cupon_rate[i])].append(i)
        


def get_similar_bonds(cr_user, md_user):
    """Finding indexes of similar bonds to interpolate in the future"""
    if str(cr_user) in dict.keys():
        for key in dict:
            if key == str(cr_user):
                s = dict.get(key)
                break
        for index in s:
            if maturity_date[index] < md_user:
                s.remove(index)
        return s
    else:
         return []

     
def get_spot_rate(md_user, price_user):
    """Calculates spot rate of currently trading bonds
    or the the rate of interest between today and maturity day"""
    if price_user == "-":
        return "N/A"
    price_user = float(price_user)
    return (100/price_user)**(1/md_user) - 1
    

def get_ytm(md_user,cr_user,price_user):
    if price_user == "-":
        return "N/A"
    price_user = float(price_user)
    if md_user < 1/2:
         return -math.log(price_user/(cr_user+100))/md_user
    amount_of_days = []
    while md_user > 0:
        amount_of_days.append(md_user)
        md_user -= 0.5
    amount_of_days.sort()
    md_user = amount_of_days[-1]
    x, i= symbols("x i")
    if len(amount_of_days)% 2 == 0:
        lhs = Sum(cr_user*exp(-x*(md_user-i)), (i, 0.5, 0.5*len(amount_of_days)-0.5)).doit()
        rhs = Sum(cr_user*exp(-x*(md_user-i)), (i, 1, 0.5*len(amount_of_days) -1)).doit()
        lrhs = lhs + rhs + (cr_user+100)*exp(-x*md_user) - price_user
        return(nsolve(lrhs, x, 0))
    else: 
        lhs = Sum(cr_user*exp(-x*(md_user-i)), (i, 0.5,  0.5*(len(amount_of_days)) -1)).doit()
        rhs = Sum(cr_user*(exp(-x*(md_user-i))), (i, 1, 0.5*(len(amount_of_days)) -0.5)).doit()
        lrhs = lhs + rhs + (cr_user+100)*exp(-x*md_user) - price_user
        return(nsolve(lrhs, x, 0))
       


if __name__ == "__main__" :
    while True:
        command = input("Please enter a comand ")
        
        if command == "h" or command == "help":
            print("Help:")
            print(" h: print this help menu")
            print(" c: calculates ytm of bond given price, cupon rate and maturity date")
            print(" d: displays canadian government bonds trading currently")
            print(" x: exits the ytm calculator")
        elif command == "x":
            break
        elif command == "c":
            try:
                main()
            except KeyboardInterrupt:
                    exit()
        elif command == "d":
            try:
                dis()
            except KeyboardInterrupt:
                    exit()
        else:
            print(command, "not recognize")
            print("try help or h")
    print("Good bye!")
            

         
         
         
  
        
