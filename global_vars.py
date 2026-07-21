import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime, timedelta

time_format = "%Y-%m-%d"
nlev = 58

data_folder = '/glade/derecho/scratch/sarahryu/leap2026/'

startdate_dyamond1 = datetime.strptime('2016-08-01', time_format)
enddate_dyamond1 = startdate_dyamond1 + timedelta(days=40)

startdate_dyamond2 = datetime.strptime('2020-01-20', time_format)
enddate_dyamond1 = startdate_dyamond2 + timedelta(days=40)


lev_input_var_names = ['Tin', 'qin', 'Uin', 'vinMinusSH']
surf_input_var_names = ['usurf', 'LANDFRAC', 'ICEFRAC', 'PHIS', 'SOLIN']


### All input variables for the first version of the Machine Learning Model
input_vars_ver1 = [f'{var}_lev{i}' for var in lev_input_var_names for i in range(58)]
input_vars_ver1 += surf_input_var_names

output_vars_ver1_dtcond = [f'DTCOND_lev{i}' for i in range(58)]
output_vars_ver1_dcq = [f'DCQ_lev{i}' for i in range(58)]