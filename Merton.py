from io import StringIO
import requests
import time, boto3
import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm
from __future__ import division
import lxml
import datetime 
import calendar
import scipy.optimize as sco
from numpy.linalg import cholesky



t = 1 # one-year transition matrix
Nsim = 5500 # number of simulations for CVaR
rho = 0.40 # for correlation matrix

"""
np.matrix shows a smoothed 1-year transition probability matrix based on a
long-term (20+ years) historical Moody’s estimate, as used in the popular CreditMetrics
model. Source: RiskMetrics Group and Lehman Brothers. 
"""
#AAA, AA, A, BBB, BB, B, CCC, Default
transition_matrix = np.matrix([[0.8812, 0.1029, 0.0102, 0.0050, 0.0003, 0.0002, 0.0001, 0.0001],
                              [0.0108, 0.8870, 0.0955, 0.0034, 0.0015, 0.0010, 0.0004, 0.0003],
                              [0.0006, 0.0288, 0.9011, 0.0592, 0.0074, 0.0016, 0.0006, 0.0008],
                              [0.0005, 0.0034, 0.0707, 0.8504, 0.0605, 0.0101, 0.0028, 0.0016],
                              [0.0003, 0.0008, 0.0056, 0.0568, 0.7957, 0.0808, 0.0454, 0.0146],
                              [0.0001, 0.0004, 0.0017, 0.0065, 0.0659, 0.8270, 0.0276, 0.0706],
                              [0.0001, 0.0002, 0.0064, 0.0105, 0.0305, 0.0611, 0.6296, 0.2616],
                              [0, 0, 0, 0, 0, 0, 0, 1 ]])

def check_rf(rf):
    while True:
        try: 
            rf = float(rf)
            while (rf < 0 or rf >= 100):
                print("Risk free rate must be between 0 and a 100")
                rf = input('Please enter risk free rate as an int or float:')
                rf = check_rf(rf) 
            return rf
        except ValueError:
          print("Please enter integers or floats")
  
      
#1590899,1007000,188000
#[3, 1, 5]
#AAA, AA, A, BBB, BB, B, CCC, Default
        
def check_num_assets(num_of_assets):
    while True:
        try: 
            num_of_assets = int(num_of_assets)
            while (num_of_assets < 0):
                print("Number of Assets must be a positive integer")
                num_of_assets = input('Please enter number of assets as a positive int:')
                num_of_assets = check_num_assets(num_of_assets) 
            return num_of_assets
        except ValueError:
          print("Please enter positive integers")

def check_exposure_assets(val_assets_exposure):
    while True:
        try: 
            val_assets_exposure = float(val_assets_exposure)
            while (val_assets_exposure < 0):
                print("Exposure of Assets must be a positive integer or Float")
                val_assets_exposure = input('Please enter number of assests as a positive int:')
                val_assets_exposure = check_exposure_assets(val_assets_exposure) 
            return val_assets_exposure
        except ValueError:
          print("Please enter positive integers or floats")
          
def check_credit_rating(credit_rating):
    while True:
        try:
            if type(credit_rating) != str:
                print("Credit Rating must be a str")
                credit_rating = input('Credit Rating must be one of the following str: AAA, AA, A, BBB, BB, B, CCC, Default')
                credit_rating = check_credit_rating(credit_rating) 
            else:
                credit_rating = credit_rating.upper()
                while (credit_rating != "AAA" and credit_rating != "AA" and credit_rating != "A" and credit_rating != "BBB" and credit_rating != "BB" and credit_rating != "B" and credit_rating != "CCC" and credit_rating != "Default"):
                    print("Credit Rating must be one of the following str: AAA, AA, A, BBB, BB, B, CCC, Default")
                    credit_rating = input('Please enter a credit rating as a str:')
                    credit_rating = check_credit_rating(credit_rating) 
            return credit_rating
        except TypeError:
          print("Please enter a str")

def check_recovery_rate(recovery_rate):
    while True:
        try: 
            recovery_rate = float(recovery_rate)
            while (recovery_rate < 0 or recovery_rate > 100):
                print("Recovery Rate must be within 0-100")
                recovery_rate = input('Please enter the recovery rate as an percentage, ex 40%: ')
                recovery_rate = check_recovery_rate(recovery_rate) 
            return recovery_rate
        except TypeError:
          print("Please enter ints or floats")

def get_corr_matrix():
    sigma = rho*np.ones((count,count))
    sigma = sigma -np.diag(np.diag(sigma)) + np.eye(count)
    return sigma 

def credit_risk_calc():
    LGD = 1 - recovery_rate
    # compute the cut off for each credit rating
    Z=np.cumsum(np.flipud(transition_matrix.T),0)
    Z[Z>=1] = 1-1/1e12;
    Z[Z<=0] = 0+1/1e12;

    CutOffs=norm.ppf(Z,0,1) # compute cut offes by inverting normal distribution

    # credit spread implied by transmat
    PD_t = transition_matrix[:,-1] # default probability at t
    credit_spread = -np.log(1-LGD*PD_t)/1

    # simulate jointly normals with sigma as vcov matrix
    # use cholesky decomposition

    c = cholesky(sigma)
    # cut off matrix for each bond based on their ratings
    cut = np.matrix(CutOffs[:,credit_ratings]).T
    # reference value 
    EV = np.multiply(val_assets_exposure, np.exp(-(rf+credit_spread[credit_ratings])*t))

    # bond state variable for security Value
    cp = np.tile(credit_spread.T,[count,1])
    state = np.multiply(val_assets_exposure,np.exp(-(rf+cp)*t))
    state = np.append(state,np.multiply(val_assets_exposure,recovery_rate),axis=1) #last column is default case
    states = np.fliplr(state) # keep in same order as credit cutoff
    return c, cut, states, EV
    

def monte_carlo_sim(c, cut, states, EV):
    Loss=np.zeros((count,Nsim)) # initialization of value array for MC
    # Monte Carlo Simulation Nsim times
    for i in range(0,Nsim):
        YY = np.matrix(np.random.normal(size=3))
        rr = c*YY.T
        rating = rr<cut
        rate_idx = rating.shape[1]-np.sum(rating,1) # index of the rating
        row_idx = range(0,count)
        col_idx = np.squeeze(np.asarray(rate_idx))
        V_t = states[row_idx,col_idx] # retrieve the corresponding state value of the exposure
        Loss_t = V_t-EV.T
        Loss[:,i] = Loss_t
    Portfolio_MC_Loss = np.sum(Loss,0)
    Port_Var = -1*np.percentile(Portfolio_MC_Loss,1)
    ES = -1*np.mean(Portfolio_MC_Loss[Portfolio_MC_Loss<-1*Port_Var])
    return Port_Var, ES


def main():
    global rf
    global sigma
    global recovery_rate
    global credit_ratings
    global val_assets_exposure
    rf = input('Please enter risk free rate an int or float: ')
    rf = check_rf(rf)
    num_of_assets = (input('Please enter the number of assets, ex 4: '))
    num_of_assets = check_num_assets(num_of_assets)
    global count
    count = 0
    list_of_exp = []
    credit_ratings = []
    while count < num_of_assets:
        val_assets_exposure = (input('Please enter the exposure of the asset, ex 4000000: '))
        val_assets_exposure = check_exposure_assets(val_assets_exposure)
        list_of_exp.append(val_assets_exposure)
        credit_rating = (input('Please enter the credit rating of the asset, ex "AAA": '))
        credit_rating = check_credit_rating(credit_rating)
        if "AAA" == credit_rating:
            credit_ratings.append(0)
        elif credit_rating == "AA":
            credit_ratings.append(1)
        elif credit_rating == "A":
            credit_ratings.append(2)
        elif "BBB" == credit_rating:
             credit_ratings.append(3)
        elif credit_rating == "BB":
            credit_ratings.append(4)
        elif credit_rating == "B":
            credit_ratings.append(5)
        elif "CCC" == credit_rating:
             credit_ratings.append(6)
        elif "Default" == credit_rating:
             credit_ratings.append(7)
        count += 1
    val_assets_exposure = np.matrix(list_of_exp).T
    recovery_rate = (input('Please enter the recovery rate as an int or float, ex 40%: '))
    recovery_rate = check_recovery_rate(recovery_rate)
    recovery_rate = recovery_rate/100
    sigma = get_corr_matrix()
    c, cut, states, EV = credit_risk_calc()
    Port_Var, ES = monte_carlo_sim(c, cut, states, EV)
    print("The 1% Expected Shortfall is of: {}".format(round(ES,10)))
    print("The 1% Var of the Portafolio is of: {}".format(round(Port_Var,10)))
    total = 0 
    for i in val_assets_exposure:
        total += int(i)
    print("Total Exposure is of: {}".format(val_assets_exposure))
    
    

if __name__ == "__main__" :
    while True:
        command = input("Please enter a comand ")
        
        if command == "h" or command == "help":
            print("Help:")
            print(" h: print this help menu")
            print(" c: CreditMetrics Var Calculation")
            print(" x: exits CreditMetrics Var Calculation")
        elif command == "x":
            break
        elif command == "c":
            try:
                main()
            except KeyboardInterrupt:
                    exit()
        else:
            print(command, "not recognize")
            print("try help or h")
    print("Good bye!")


def time_frame(x: np.ndarray, window: int) -> np.ndarray:
    '''
    Create a view into the array with the given shape and strides.
    Returns np.ndarray of rolling vals. 
    '''
    if not isinstance(x, np.ndarray):
        raise TypeError
    return np.lib.stride_tricks.as_strided(x, x.shape[:-1] + (x.shape[-1] - window + 1, window), x.strides + (x.strides[-1],))

def company_csv(key: str, bucket='p-def'):
    '''
    Loads CSV file from firm frim the file key inputed. 
    Returns dict with [col name: np.ndarray.index()] and np.ndarray of data
    '''
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO.StringIO(obj['Body'].read()), index_col=0, parse_dates=True)

def company_csv_data(key: str, bucket='p-def'):
    '''
    Loads CSV file from firm frim the file key inputed. 
    Returns dict with [col name: np.ndarray.index()] and np.ndarray of data
    '''
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read().splitlines()

    #Clean data
    header_dict = {key:value-1 for value,key in enumerate(data[0].split(','))}
    date = [row.split(',')[0] for row in data[1:]]
    data = [",".join(row.split(',')[1:]) for row in data[1:]]
    #np.arraw of data
    data = np.genfromtxt(data, delimiter=',')
    return header_dict, date, data

def data_sucss(data, key: str, bucket='p-def') -> bool:
    '''
    Returns true if data is transcribed to S3
    '''
    
    return boto3.resource('s3').Object(bucket, key).put(Body=data)

def data_to_csv(header_dict: dict, date: list, comp: np.ndarray, results: np.ndarray):
    '''
    Gets all args to single file, returns csv arguments back into a single CSV file.
    Args:
        header_dict: [col name: np.ndarray.index()],  
        date: a list of with the date of each obs, 
        comp: a np.ndarray of company data 
        results: a np.array with len(results) == len(comp) containing asset vals. 
    '''
    csv_file = StringIO.StringIO()

    # Convert header map to map of index to name
    index_to_name = {value+1:key for key, value in header_dict.items()}

    # Add column names for new asset value columns
    index_start = max(index_to_name.keys()) + 1
    for i in range(results.shape[1]):
        index_to_name[index_start + i] = 'Va_{:d}'.format(i+1)

    # Create header row
    csv_file.write(','.join([index_to_name[i] for i in range(len(index_to_name))]) + '\n')

    # Combine company data and results and write to CSV
    np.savetxt(csv_file, np.hstack((np.array((date)).reshape((-1,1)), comp, results)), fmt='%s', delimiter=',')
    csv_file.seek(0)

    return csv_file

def Black_Scholes_Merton(comp: np.ndarray, header_dict: dict, time_horizon: list, min_hist_vals=252):
    '''
    Returns np.array of ts of firm value.
    '''

    def b_s_m(s, debug=False):
        # s = bookvalue
        # r_f = Riskfree
        # T = optionYears
        # vol = Volatility
        # x = optionStrike or face_value_debt
        # sqrtT = np.sqrt(T)
        
        d1 = (np.log(s/x) + (r_f + 0.5*vol*vol)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        callResult = v_e - (s*norm.cdf(d1) - np.exp(-r_f*T)*x*norm.cdf(d2))
        #putResult = x * np.exp(-r_f*T) * (1.0 - norm.cdf(d2)) - s * (1.0 - norm.cdf(d1))
        if debug:
            print("d1 = {:.6f}".format(d1))
            print("d2 = {:.6f}".format(d2))
            print("Error = {:.6f}".format(callResult))

        return callResult

    width = 252.0

    # Start time 
    start_time = min_hist_vals
    timesteps = range(min_hist_vals, len(comp))

    # Get Volatility
    ret_col = header_dict['RET']
    sigma_e = np.zeros((comp.shape[0]))
    sigma_e[:width-1] = np.nan
    sigma_e[width-1:] = np.std(time_frame(np.log(comp[:,ret_col] + 1), width), axis=-1)

    assert type(time_horizon) in [list, tuple],"time_horizon must be a list"

    # Create np.array to store results
    results = np.empty((comp.shape[0],len(time_horizon)))

    for i, years in enumerate(time_horizon):
        T = 252*years
        results[:,i] = comp[:,header_dict['mkt_val']]

        # Run through time series
        for i_t, t in enumerate(timesteps):
            # Check leverage
            if comp[t,header_dict['face_value_debt']] > 1e-11:
                
                v_a_per = results[t-252:t,i]
                v_a_ret = np.log(v_a_per/np.roll(v_a_per,1))
                v_a_ret[0] = np.nan
                vol = np.nanstd(v_a_ret)

                if i_t == 0:
                    subset_timesteps = range(t-252, t+1)
                else:
                    subset_timesteps = [t]

                # Iterate on previous values of S
                n_its = 0
                while n_its < 10:
                    n_its += 1
                    # Loop over date, calc S and vol 
                    for t_sub in subset_timesteps:
                        r_f = (1 + comp[t_sub,header_dict['DGS1']])**(1.0/365) - 1
                        v_e = comp[t_sub,header_dict['mkt_val']]
                        x = comp[t_sub,header_dict['face_value_debt']]
                        sol = sco.root(b_s_m, results[t_sub,i])
                        results[t_sub,i] = sol['x'][0]

                    # Update vol based on new values of S
                    last_vol = vol
                    v_a_per = results[t-252:t,i]
                    v_a_ret = np.log(v_a_per/np.roll(v_a_per,1))
                    v_a_ret[0] = np.nan
                    vol = np.nanstd(v_a_ret)

                    if abs(last_vol - vol) < 1e-2:
                        #comp.loc[t_sub, 'vol'] = vol
                        break
            else:
                # comp is unlevered
                pass

    return results

def run_model(key:str, time_horizon=[1,2,3,4,5]):
    '''
    Apply B-S-M model to calc firms value as a function of time option pricing model to calculate inferred firm asset values as a
    function of time. 
    Args:
        time_horizon: List of time horizons (In Years) to calculate model over
    '''
    start = time.time()

    # Get data from S3
    h_map, date, data = company_csv_data(key)

    if len(date) > 252:
        # Run the simulation
        results = Black_Scholes_Merton(data, h_map, time_horizon=time_horizon)

        # Merge data back into CSV
        csv_file = data_to_csv(h_map, date, data, results)

        # Save results to S3
        result_key = key.replace('merged-corp-data', 'merton-results')
        response = data_sucss(csv_file, result_key)
    else:
        response = False

    end = time.time()

    return start, end, response
