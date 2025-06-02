from parameters import Paras, TN
import numpy as np
import pandas as pd

I_shift_apps = np.zeros((3, 96), dtype=int) # Shift appliances action variable, 3 apps, 96 time intervals, default 0

def generateSchedule(TN_arr):
  return TN(TN_arr[0], TN_arr[1], TN_arr[2], TN_arr[3])

def multivariate_bernoulli(p, size=1):
  p = np.array(p)
  if isinstance(size, int):
    size = (size, len(p))
  return np.random.binomial(n=1, p=p, size=size)

def multivariate_normal_independent(mean, stddev, size=1): 
  # Here we assume variables(apps) probability is independaent, use standard deviation as input, instead of covariant matrix
  return np.random.normal(loc=mean, scale=stddev, size=(size, len(mean)))

# Solar, outside temperature, electricity price time series
columns_to_read = ['Solar', 'T_out', 'RTP']
df = pd.read_csv('hourly(1).csv', usecols=columns_to_read,  nrows=62*24)
solar = df['Solar'].to_numpy() / 1000 # convert (W / m2) to (kW / m2)
PV_t = solar * Paras['ConversionEfficiency'] * Paras['Area']
temp_out = df['T_out'].round(1).to_numpy()
real_time_price = df['RTP'].to_numpy()

p_max_hvac = Paras['AC'][0]
p_max_ewh = Paras['EWH'][0]
p_max_ev = Paras['EV'][0]
p_max_ess = Paras['ESS'][0]

# a made up parameter in EWH
Wt = generateSchedule(Paras['EWH_hwf'])
ewh_tank_vol = 200 # EWH water tank volume V in (7b) not given, I choose 200 liter (0.08 m3)

def generateAllSchedule():
    t_ini_dishwasher = int(generateSchedule(Paras['DishWasher'][1]))
    t_end_dishwasher = int(generateSchedule(Paras['DishWasher'][2]))
    t_ini_washmachine = int(generateSchedule(Paras['WashMachine'][1]))
    t_end_washmachine = int(generateSchedule(Paras['WashMachine'][2]))
    t_ini_clothesdryer = t_end_washmachine
    t_end_clothesdryer = t_ini_clothesdryer + 8

    _Ini_shift = np.array( [t_ini_dishwasher, t_ini_washmachine, t_ini_clothesdryer], dtype=np.int8 )
    _End_shift = np.array( [t_end_dishwasher, t_end_washmachine, t_end_clothesdryer], dtype=np.int8 )

    _ini_ev = int(generateSchedule(Paras['EV'][1]))
    _end_ev = int(generateSchedule(Paras['EV'][2]))

    t_ini_tv = int(generateSchedule(Paras['TV'][1]))
    t_dur_tv = int(generateSchedule(Paras['TV'][2]))
    t_end_tv = t_ini_tv + t_dur_tv
    t_ini_refri = Paras['refrige'][1]
    t_dur_refri = Paras['refrige'][2]
    t_end_refri = t_ini_refri + t_dur_refri
    t_ini_light = int(generateSchedule(Paras['light'][1]))
    t_dur_light = int(generateSchedule(Paras['light'][2]))
    t_end_light = t_ini_light + t_dur_light
    t_ini_vacuum = int(generateSchedule(Paras['vacuum'][1]))
    t_dur_vacuum = int(generateSchedule(Paras['vacuum'][2]))
    t_end_vacuum = t_ini_vacuum + t_dur_vacuum
    t_ini_hairdryer = int(generateSchedule(Paras['hairdryer'][1]))
    t_dur_hairdryer = Paras['hairdryer'][2]
    t_end_hairdryer = t_ini_hairdryer + t_dur_hairdryer

    _Ini_nocntl = np.array( [t_ini_tv, t_ini_refri, t_ini_light, t_ini_vacuum, t_ini_hairdryer], dtype=np.int8)
    _End_nocntl = np.array( [t_end_tv, t_end_refri, t_end_light, t_end_vacuum, t_end_hairdryer], dtype=np.int8)

    return _Ini_shift, _End_shift, _ini_ev, _end_ev, _Ini_nocntl, _End_nocntl

temp_out_min, temp_out_max = np.min(temp_out), np.max(temp_out)
def tempPriceSolar():
    real_time_price_min, real_time_price_max = np.min(real_time_price), np.max(real_time_price)
    PV_t_min, PV_t_max = np.min(PV_t), np.max(PV_t)
    solar_min, solar_max = np.min(solar), np.max(solar)
    
    # 2 months 62 days : December + January original value and normalized value
    temp_out_96 = np.zeros(Paras['T_end'] * 62)
    real_time_price_96 = np.zeros(Paras['T_end'] * 62)
    PV_t_96 = np.zeros(Paras['T_end'] * 62)
    solar_96 = np.zeros(Paras['T_end'] * 62)
    temp_out_norm_96 = np.zeros(Paras['T_end'] * 62)   
    real_time_price_norm_96 = np.zeros(Paras['T_end'] * 62)
    PV_t_norm_96 = np.zeros(Paras['T_end'] * 62)
    solar_norm_96 = np.zeros(Paras['T_end'] * 62)
    
    for day in range(62):
        for i in range(24):
            norm_temp = (temp_out[day*24 + i] - temp_out_min) / (temp_out_max - temp_out_min)
            norm_price = (real_time_price[day*24 + i] - real_time_price_min) / (real_time_price_max - real_time_price_min)
            norm_PV = (PV_t[day*24 + i] - PV_t_min) / (PV_t_max - PV_t_min)
            norm_solar = (solar[day*24 + i] - solar_min) / (solar_max - solar_min)
            for j in range(4):
                temp_out_96[day*96 + i*4 + j] = temp_out[day*24 + i]
                temp_out_norm_96[day*96 + i*4 + j] = norm_temp
                real_time_price_96[day*96 + i*4 + j] = real_time_price[day*24 + i]
                real_time_price_norm_96[day*96 + i*4 + j] = norm_price
                PV_t_96[day*96 + i*4 + j] = PV_t[day*24 + i]
                PV_t_norm_96[day*96 + i*4 + j] = norm_PV
                solar_96[day*96 + i*4 + j] = solar[day*24 + i]
                solar_norm_96[day*96 + i*4 + j] = norm_solar
                
    return temp_out_96, temp_out_norm_96, real_time_price_96, real_time_price_norm_96, PV_t_96, PV_t_norm_96, solar_96, solar_norm_96
