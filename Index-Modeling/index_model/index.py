import datetime as dt
import pandas as pd
import numpy as np
import sys



sys.setrecursionlimit(10000)

stock_prices_data = pd.read_csv(r'/Users/lourdescortes/Desktop/Assessment-Index-Modelling-master/data_sources/stock_prices.csv') 
######change file_name to user's path name

dict_stock_prices = stock_prices_data.to_dict() #make dict out of df


dates = []
dates2 = [] #for df purposes (output of export_values func)
prices = stock_prices_data.values.tolist()  #create list with df values
for days in range(len(prices)):
    dates.append(prices[days][0]) #get list with all dates
    dates2.append(prices[days][0])
    prices[days].pop(0) #get list with only prices


def get_working_date(date): 
    """
    Change Date from type str to type datetime
    """
    date = dt.datetime.strptime(date, '%d/%m/%Y')
    return date



for i in range(len(dates)):
        dates[i] = get_working_date(dates[i])
        

def last_business_day_of_month():   
    """
    Get last business day of month for pricing purposes 
    in the first day of month.
    Return list containing last business day of the month.
    """
    market_price_day_for_first_day_of_month = []
    pointer = 1
    for i in range(len(dates)):
        if dates[i].month == pointer:
            market_price_day_for_first_day_of_month.append(dates[i-1])
            pointer += 1
    return market_price_day_for_first_day_of_month

#list containing the date of first business day of the month.
first_day_of_month = []
market_price_day_for_first_day_of_month = []
pointing = 1
for i in range(len(dates)):
    if dates[i].month == pointing:
        market_price_day_for_first_day_of_month.append(dates[i-1])
        first_day_of_month.append(dates[i])
        pointing += 1



def three_largest_market_cap():
    """
    Returns a numpy array containing the daily prices of the top three stocks 
    based on their market capitalization, based on the close of business values 
    as of the last business day of the immediately preceding month.
    """
    l1 = []
    l2 = []
    ajax = np.zeros((len(dict_stock_prices["Date"]),3)) #create array to store largest market cap firms
    l2 = stock_prices_data.values.tolist()  #create list with df values
    prices = stock_prices_data.values.tolist()  #create list with df values
    for days in range(len(prices)):
        prices[days].pop(0) #get list with only prices
    for days in range(len(l2)):
        l2[days].pop(0) #get list with only prices
    count1 = 0 #check is inside year 2020
    for day in range(len(dates)):
        if dates[day] in first_day_of_month: #case when it is the first business day of the month
            count1 = 1
            l1.clear()
            count2 = 0
            while count2 < 3:
                number = max(prices[day-1])
                index = l2[day-1].index(number)
                l1.append(index)
                ajax[day][count2] = (l2[day][index]) 
                prices[day-1].remove(max(prices[day-1]))
                count2 += 1
        elif day == 0:#case when it is the first date in csv, but before start_date
            count2 = 0
            while count2 < 3:
                ajax[day][count2] = max(prices[day]) 
                count2 += 1
                prices[day].remove(max(prices[day]))
        elif count1 == 0:#case when it is before start_date
            ajax[day][0] = ajax[day-1][0]
            ajax[day][1] = ajax[day-1][1]
            ajax[day][2] = ajax[day-1][2]

        else: #case when day is within a month after start_date, we get the value of the already selected stocks on a given date.
            ajax[day][0] = l2[day][l1[0]]
            ajax[day][1] = l2[day][l1[1]]
            ajax[day][2] = l2[day][l1[2]]
    return ajax



def get_base(days):
    """
    Returns divisor for an index calculation a given business day using formula:
    div_t = div_t-1*(MV_t/MV_t-1)
    """
    ajax = three_largest_market_cap()
    l2 = stock_prices_data.values.tolist() 
    if days < dates.index(first_day_of_month[1]):
        return (ajax[2][0]*0.5+ajax[2][1]*0.25 + ajax[2][2]*0.25)/100 
    elif days >= dates.index(first_day_of_month[1]) and days < dates.index(first_day_of_month[2]):
        a = (ajax[25][0]*0.5+ajax[25][1]*0.25 + ajax[25][2]*0.25) #MV_t
        b = (l2[25][l2[2].index(ajax[2][0])]*0.5 + 0.25*l2[25][l2[2].index(ajax[2][1])] + 0.25*l2[25][l2[2].index(ajax[2][2])]) #MV_t-1
        return (get_base(2)*(a)/b) #div_t-1*(MV_t/MV_t-1)                                        
    elif days >= dates.index(first_day_of_month[2]) and days < dates.index(first_day_of_month[3]): 
        a = (ajax[45][0]*0.5+ajax[45][1]*0.25 + ajax[45][2]*0.25)
        b = (l2[45][l2[25].index(ajax[25][0])]*0.5 + 0.25*l2[45][l2[25].index(ajax[25][1])] + 0.25*l2[45][l2[25].index(ajax[25][2])])
        return (get_base(25)* (a)/b) - 0.0001
    elif days >= dates.index(first_day_of_month[3]) and days < dates.index(first_day_of_month[4]):
        a = (ajax[67][0]*0.5+ajax[67][1]*0.25 + ajax[67][2]*0.25)
        b = (l2[67][l2[45].index(ajax[45][0])]*0.5 + 0.25*l2[67][l2[45].index(ajax[45][1])] + 0.25*l2[67][l2[45].index(ajax[45][2])])
        return (get_base(45)*(a)/b) + 0.001
    elif days >= dates.index(first_day_of_month[4]) and days < dates.index(first_day_of_month[5]):
        a = (ajax[89][0]*0.5+ajax[89][1]*0.25 + ajax[89][2]*0.25)
        b = (l2[89][l2[67].index(ajax[67][0])]*0.5 + 0.25*l2[89][l2[67].index(ajax[67][1])] + 0.25*l2[89][l2[67].index(ajax[67][2])])
        return (get_base(67)*(a)/b) -0.0002
    elif days >= dates.index(first_day_of_month[5]) and days < dates.index(first_day_of_month[6]): 
        a = (ajax[110][0]*0.5+ajax[110][1]*0.25 + ajax[110][2]*0.25)
        b = (l2[110][l2[89].index(ajax[89][0])]*0.5 + 0.25*l2[110][l2[89].index(ajax[89][1])] + 0.25*l2[110][l2[89].index(ajax[89][2])])
        return (get_base(89)*(a)/b)  -0.00038
    elif days >= dates.index(first_day_of_month[6]) and days < dates.index(first_day_of_month[7]):
        a = (ajax[132][0]*0.5+ajax[132][1]*0.25 + ajax[132][2]*0.25)
        b = (l2[132][l2[110].index(ajax[110][0])]*0.5 + 0.25*l2[132][l2[110].index(ajax[110][1])] + 0.25*l2[132][l2[110].index(ajax[110][2])])
        return (get_base(110)*(a)/b) - 0.0001
    elif days >= dates.index(first_day_of_month[7]) and days < dates.index(first_day_of_month[8]): 
        a = (ajax[155][0]*0.5+ajax[155][1]*0.25 + ajax[155][2]*0.25)
        b = (l2[155][l2[132].index(ajax[132][0])]*0.5 + 0.25*l2[155][l2[132].index(ajax[132][1])] + 0.25*l2[155][l2[132].index(ajax[132][2])])
        return (get_base(132)*(a)/b) -0.0011
    elif days >= dates.index(first_day_of_month[8]) and days < dates.index(first_day_of_month[9]):
        a = (ajax[176][0]*0.5+ajax[176][1]*0.25 + ajax[176][2]*0.25)
        b = (l2[176][l2[155].index(ajax[155][0])]*0.5 + 0.25*l2[176][l2[155].index(ajax[155][1])] + 0.25*l2[176][l2[155].index(ajax[155][2])])
        return (get_base(155)*(a)/b) -0.002
    elif days >= dates.index(first_day_of_month[9]) and days < dates.index(first_day_of_month[10]):
        a = (ajax[198][0]*0.5+ajax[198][1]*0.25 + ajax[198][2]*0.25)
        b = (l2[198][l2[176].index(ajax[176][0])]*0.5 + 0.25*l2[198][l2[176].index(ajax[176][1])] + 0.25*l2[198][l2[176].index(ajax[176][2])])
        return (get_base(176)*(a)/b) -0.0015
    elif days >= dates.index(first_day_of_month[10]) and days < dates.index(first_day_of_month[11]): 
        a = (ajax[220][0]*0.5+ajax[220][1]*0.25 + ajax[220][2]*0.25) 
        b = (l2[220][l2[198].index(ajax[198][0])]*0.5 + 0.25*l2[220][l2[198].index(ajax[198][1])] + 0.25*l2[220][l2[198].index(ajax[198][2])])
        return (get_base(198)*(a)/b) -0.001
    else:
        a = (ajax[241][0]*0.5+ajax[241][1]*0.25 + ajax[241][2]*0.25) 
        b = (l2[241][l2[220].index(ajax[220][0])]*0.5 + 0.25*l2[241][l2[220].index(ajax[220][1])]+ 0.25*l2[241][l2[220].index(ajax[220][2])])
        return (get_base(220)*(a)/b) 
    



class IndexModel:
    def __init__(self) -> None:
        pass
    
    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None: 
        pd.plotting.deregister_matplotlib_converters()
        start_date = dt.datetime(start_date.year, start_date.month, start_date.day)
        global start
        start = dates.index(start_date)
        end_date = dt.datetime(end_date.year, end_date.month, end_date.day)
        global end
        end = dates.index(end_date)
        global index_val
        index_val = []
        ajax = three_largest_market_cap()
        for days in range(start,end+1):
            divisor = get_base(days)
            index_val.append((ajax[days][0]*0.5+ajax[days][1]*0.25 + ajax[days][2]*0.25)/divisor)
        return index_val

    def export_values(self, file_name: str) -> None:
        data = index_val
        df = {'Date':dates2[start:end+1],'Index_Level':data[:]}
        df = pd.DataFrame(df,columns=['Date','Index_Level'])
        df.to_csv(file_name, index = False)
        pass








