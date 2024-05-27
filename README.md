# Undergrad Projects

1. Fixed_Income.py is a program in python which enable users to get information of currently trading
Canadian Government Bonds via web scraping. It is also a calculator of yield curves, spot curves and forward curves 
via interpolation techniques. 

2. Merton_.py is a program in python that values a company and calculates the probability of default of such company using the Black-Scholes-Merton (BSM) model over time. The program also has a CreditMetrics-like VaR calculator that with the input of some credit ratings, exposure and recovery rate to access credit risk among other variables is able to yield the Expexted Shortfall and VaR of the Portafolio. 

3. Index-Modeling is a folder that contains the program idex.py that calculates a total return index consisting of stocks from the index universe ("Stock_A" to including "Stock_J") from file stock_prices.csv. Rules are that every first business day of a month the index selects from the universe the top three stocks based on their market capitalization, based on the close of business values as of the last business day of the immediately preceding month.The selected stock with the highest market capitalization gets assigned a 50% weight, while the second and third each get assigned 25%. The selection becomes effective close of business on the first business date of each month. Results are in export.csv
