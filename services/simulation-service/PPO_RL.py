import gymnasium as gym

from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
import numpy as np
import math, os, sys

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from Plots import RewardPlotCallback, plotsAll
from parameters import Paras
from DataSample import (temp_out_min, temp_out_max, p_max_hvac, p_max_ewh, p_max_ev, p_max_ess, generateAllSchedule, tempPriceSolar)
from util import FlattenActionWrapper, MBD, MND

T_ini_shift, T_end_shift, t_ini_ev, t_end_ev, T_ini_nocntl, T_end_nocntl = generateAllSchedule()

K_shift = np.array( [Paras['K_DishWasher'], Paras['K_WashMachine'], Paras['K_ClothesDryer']], dtype=np.uint8)
Power_shift = np.array( [Paras['DishWasher'][0], Paras['WashMachine'][0], Paras['ClothesDryer']] )
Power_nonctrl = np.array( [Paras['TV'][0], Paras['refrige'][0], Paras['light'][0], Paras['vacuum'][0], Paras['hairdryer'][0]] )

interval = 96*8#Paras['N'] * Paras['proc_size']
total_steps = Paras['Mep'] * interval

temp_out_96, temp_out_norm_96, real_time_price_96, real_time_price_norm_96, PV_t_96, PV_t_norm_96, solar_96, solar_norm_96 = tempPriceSolar()

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.T_ini_shift = T_ini_shift
        self.T_end_shift = T_end_shift
        self.t_ini_ev = t_ini_ev
        self.t_end_ev = t_end_ev
        self.T_ini_nocntl = T_ini_nocntl
        self.T_end_nocntl = T_end_nocntl

        # Define observation space
        self.observation_space = spaces.Dict({
            'shift_prog': spaces.MultiDiscrete(1+K_shift),  # 4

            'time_to_start_shift': spaces.MultiDiscrete(Paras['T_end'] * np.ones(self.T_ini_shift.shape, dtype=np.int32), start=self.T_ini_shift-Paras['T_end']+1),   #10

            'home_temp': spaces.Box(low = -100.0, high = 100.0, shape = ()), #2

            'water_temp': spaces.Box(low = -100.0, high = 100.0, shape = ()),   #11

            'soc_ess': spaces.Box(low = 0.0, high = 1.0, shape = ()),  #5

            'soc_ev': spaces.Box(low = 0.0, high = 1.0, shape = ()),   #6

            'time_to_start_noctrl': spaces.MultiDiscrete(Paras['T_end'] * np.ones(self.T_ini_nocntl.shape, dtype=np.int32), start=-self.T_ini_nocntl),  #9

            'pv_power': spaces.Box(low = 0.0, high = 1.0, shape=()), #3

            'grid_price': spaces.Box(low = np.zeros((Paras['T_end'],)), high = np.ones((Paras['T_end'],))),  #1

            'solar': spaces.Box(low = np.zeros((Paras['T_end'],)), high = np.ones((Paras['T_end'],))),  #7

            'temp_out': spaces.Box(low = np.zeros((Paras['T_end'],)), high = np.ones((Paras['T_end'],))),   #8
        })

        # Define action space (action space)
        self.action_space = spaces.Dict({
             'ess_pow': spaces.Box(low=-1.0, high=1.0, shape=()),   #1
             'ess_pow_var': spaces.Box(low=-1.0, high=1.0, shape=()),   #2

             'ev_pow': spaces.Box(low=-1.0, high=1.0, shape=()),   #3
             'ev_pow_var': spaces.Box(low=-1.0, high=1.0, shape=()),   #4

             'ewh_pow': spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float16),  #5
             'ewh_pow_var': spaces.Box(low=-1.0, high=1.0, shape=()),   #6

             'hvac_pow': spaces.Box(low=-1.0, high=1.0, shape=()), #7
             'hvac_pow_var': spaces.Box(low=-1.0, high=1.0, shape=()),  #8

             'shift_on': spaces.Box(low=-np.array([1, 1, 1]), high=np.array([1, 1, 1]), shape=(3,)),  #9
        })

        self.n_calls = 0
        self.cur_steps = 0
        self.update_dryer = False
        self.dryer_bound_shifted = 0
        self.state = None
        self.day = 1

    def update_multidiscrete_bounds(self, space_name, new_starts):
        """
        Update bounds for a specific MultiDiscrete space
        """
        if space_name not in self.observation_space.spaces:
            raise KeyError(f"Space {space_name} not found")

        old_space = self.observation_space.spaces[space_name]
        if not isinstance(old_space, spaces.MultiDiscrete):
            raise TypeError(f"Space {space_name} is not MultiDiscrete")

        # Update the specific space
        self.observation_space.spaces[space_name] = spaces.MultiDiscrete(
            old_space.nvec,
            start=new_starts
        )

    def reset(self, seed=None, options=None):
        # Initial state
        prev_home_temp = None
        prev_water_temp = None
        prev_soc_ev = None
        prev_soc_ess = None

        # If this isn't the first reset, store previous values
        if hasattr(self, 'state') and self.state is not None:
            prev_home_temp = self.state['home_temp']
            prev_water_temp = self.state['water_temp']
            prev_soc_ev = self.state['soc_ev']
            prev_soc_ess = self.state['soc_ess']

        self.n_calls = 0

        if self.day == 60:
            self.day = 1
        else:
            self.day += 1

        # Generate new schedules for the day
        T_ini_shift, T_end_shift, t_ini_ev, t_end_ev, T_ini_nocntl, T_end_nocntl = generateAllSchedule()
        self.T_ini_shift = T_ini_shift
        self.T_end_shift = T_end_shift
        self.t_ini_ev = t_ini_ev
        self.t_end_ev = t_end_ev
        self.T_ini_nocntl = T_ini_nocntl
        self.T_end_nocntl = T_end_nocntl

        # Update MultiDiscrete spaces based on new bounds
        self.update_multidiscrete_bounds('time_to_start_shift', self.T_ini_shift-Paras['T_end']+1)
        self.update_multidiscrete_bounds('time_to_start_noctrl', -self.T_ini_nocntl)
        
        self.state = {
            # These should reset every day
            'shift_prog': np.array([0, 0, 0]),
            'time_to_start_shift': np.array([0, 0, 0]),
            'time_to_start_noctrl': np.array([0, 0, 0, 0, 0]),

            # These should persist between days if we have previous values
            'home_temp': prev_home_temp if prev_home_temp is not None else np.array(4.0), #np.array(np.random.randint(-2, 2)),
            'water_temp': prev_water_temp if prev_water_temp is not None else np.array(-5.0), #np.array(np.random.randint(-3, 3)),
            'soc_ev': prev_soc_ev if prev_soc_ev is not None else np.array(0.0),
            'soc_ess': prev_soc_ess if prev_soc_ess is not None else np.array(0.0),

            # These are day-specific and should update
            'pv_power': PV_t_norm_96[self.day * 96 + 32],
            'grid_price': real_time_price_norm_96[(self.day-1) * 96 + 1 + 32 : self.day*96+32+1],
            'solar': solar_norm_96[(self.day-1) * 96 + 1 + 32 : self.day*96+32+1],
            'temp_out': temp_out_norm_96[(self.day-1) * 96 + 1 + 32 : self.day*96+32+1],
        }

        info = {}
        self.update_dryer = False
        self.dryer_bound_shifted = 0

        return self.state, info

    def calShiftPower(self, action):
        shift_prog = self.state['shift_prog'] 
        action_shift = MBD(( 1 + action['shift_on']) / 2)
        time_to_start_shift = self.state['time_to_start_shift']
        p_shift = np.zeros(Power_shift.shape) # Default no penalty

        if shift_prog[1] == K_shift[1] and self.n_calls < self.T_ini_shift[2] and (not self.update_dryer):
            self.dryer_bound_shifted = self.n_calls - self.T_ini_shift[2]  # always <= 0 because Dryer starts after washer when initialized, so when washer finished, we need to update its new higher bound (smaller values of T_ini_shift[2], and T_end_shift[2], the dyer can start earlier)
            self.update_dryer = True
            assert self.dryer_bound_shifted <= 0.0

        for i in range(shift_prog.size):
            power_on = action_shift[i]
            progress = shift_prog[i]
            dryer_shifted = np.array(i == 2).astype(int) * self.dryer_bound_shifted

            if time_to_start_shift[i] == 0 or progress == K_shift[i]:
                p_shift[i] = 0 #power_on * Power_shift[i]
            else:
                if progress > 0 and progress < K_shift[i]:
                    p_shift[i] = Power_shift[i] #* (2 - power_on)
                    self.state['shift_prog'][i] += 1
                    
                elif progress == 0 and self.T_end_shift[i] + dryer_shifted - self.n_calls == K_shift[i]:
                    p_shift[i] = Power_shift[i] #* (2 - power_on) + (1 - power_on) * (K_shift[i] - progress) * Power_shift[i]
                    self.state['shift_prog'][i] += 1

                else:
                    p_shift[i] = power_on * Power_shift[i]
                    self.state['shift_prog'][i] = min(progress + power_on, K_shift[i])
           
            if self.n_calls + 1 < self.T_ini_shift[i] + dryer_shifted or \
                self.n_calls + 1 >= self.T_end_shift[i] + dryer_shifted:
                self.state['shift_prog'][i] = 0
                self.state['time_to_start_shift'][i] = 0
            else:
                self.state['time_to_start_shift'][i] = self.T_ini_shift[i] + dryer_shifted - (self.n_calls + 1)

        return p_shift

    def calControlPower(self, action):
        # 4 lines below get: 'T_in - T_inset', 'T_water - T_waterset', 'SoC_EV', 'SoC_ESS'
        hvac_dif = self.state['home_temp']   # = -Delta_T in (5a)
        ewh_dif = self.state['water_temp']
        soc_ev = self.state['soc_ev']
        soc_ess = self.state['soc_ess'] 

        hvac_action = action['hvac_pow']
        hvac_power = (1 + hvac_action) * p_max_hvac / 2
        hvac_power_var = (1 + action['hvac_pow_var']) * pow(p_max_hvac/2, 2) / 2

        ewh_action = action['ewh_pow']
        ewh_power = (1 + ewh_action) * p_max_ewh / 2
        ewh_power_var = (1 + action['ewh_pow_var']) * pow(p_max_ewh/2, 2) / 2

        ev_action = action['ev_pow']
        ev_power = (1 + ev_action) * p_max_ev / 2
        ev_power_var = (1 + action['ev_pow_var']) * pow(p_max_ev/2, 2) / 2

        ess_power = action['ess_pow'] * p_max_ess
        ess_power_var = (1 + action['ess_pow_var']) * pow(p_max_ess/2, 2) / 2

        p_mu = np.array([hvac_power, ewh_power, ev_power, ess_power])
        p_var = np.array([hvac_power_var, ewh_power_var, ev_power_var, ess_power_var])
        p_ctrl = MND(p_mu, p_var)
        for i in range(p_ctrl.size-1):
            if p_ctrl[i] < 0:
                p_ctrl[i] = 0

        # update HVAC
        need_power = min(p_ctrl[0], p_max_hvac)
        p_ctrl[0] = need_power
        T_out = self.state['temp_out'][-1] * temp_out_max + (1 - self.state['temp_out'][-1]) * temp_out_min
        temp_now_hvac = Paras['HVAC_set'] + hvac_dif
        if (hvac_dif > 0):
            need_power *= -1

        cost_hvac = Paras['HVAC_conv_in'] * \
                    pow(hvac_dif / Paras['HVAC_max_dev'], 2)  # Cost of HAVC current status

        temp_next_hvac = Paras['HVAC_iner'] * temp_now_hvac + \
                        (1 - Paras['HVAC_iner']) * \
                        (T_out + Paras['HVAC_conv'] * need_power * Paras['DeltaT'] / Paras['HVAC_cond'] )

        self.state['home_temp'] = np.array(temp_next_hvac - Paras['HVAC_set'], dtype=np.float16)

        # update EWH
        from DataSample import Wt, ewh_tank_vol
        need_power = min(p_max_ewh, p_ctrl[1])
        p_ctrl[1] = need_power
        temp_now_ewh = Paras['EWH_set'] + ewh_dif
        Q = ( Paras['EWH_cond'] * temp_now_hvac + \
              Paras['EWH_den'] * Paras['EWH_cap'] * Wt * Paras['EWH_cold'] + \
              need_power * 1000) / \
            ( Paras['EWH_cond'] + \
              Paras['EWH_den'] * Paras['EWH_cap'] * Wt ) # Here 'ewh_power * 1000' means convert 'kW' to 'W'  #need_power * 1000 ) / \

        Tau = Paras['EWH_den'] * Paras['EWH_cap'] * ewh_tank_vol / \
              ( Paras['EWH_cond'] + Paras['EWH_den'] * Paras['EWH_cap'] * Wt )

        temp_next_ewh = temp_now_ewh * math.exp(-Paras['DeltaT'] * 3600 / Tau) + \
                       Q * (1 - math.exp(-Paras['DeltaT'] * 3600 / Tau))   # 'Paras['T_end'] in hour, converted to seconds by *3600'

        cost_ewh = Paras['EWH_conv'] * pow(ewh_dif / Paras['EWH_max_dev'], 2)

        self.state['water_temp'] = np.array(temp_next_ewh - Paras['EWH_set'], dtype=np.float16)

        # update EV
        cost_ev = 0.0
        EV_SoC_max = 1.0  # not specified in paper (9b), assume 100%

        if (self.n_calls == self.t_end_ev):
            cost_ev = Paras['EV_anxiety'] * pow((soc_ev - EV_SoC_max) * Paras['EV_cap'], 2)

        power_up_ev = (EV_SoC_max - soc_ev) * Paras['EV_cap'] / \
                          ( Paras['DeltaT'] * Paras['EV_effi'] )

        EV_max_new = min(p_max_ev, power_up_ev)
        need_power = min(p_ctrl[2], EV_max_new)
        p_ctrl[2] = need_power

        soc_ev_next = soc_ev + \
                          Paras['EV_effi'] * need_power * Paras['DeltaT'] / \
                          Paras['EV_cap']

        if (self.n_calls+1 < self.t_ini_ev or self.n_calls+1 > self.t_end_ev):
            self.state['soc_ev'] = 0.0
        else:
            self.state['soc_ev'] = np.array(soc_ev_next, dtype=np.float16)
            if self.n_calls+1 == self.t_ini_ev:
                self.state['soc_ev'] = 0.35

        # update ESS
        cost_ess = 0
        ESS_soc_max = 1.0  #  not specified in paper (11)
        ESS_soc_min = 0.1  #  not specified in paper (12a)
        kappa, psi = 1.59e-5, 1.41e-6# given in reference [26]
        p_ch_ess = 0.0
        p_dis_ess = 0.0
        f_dod, delta_dod = 0.0, 0.0

        if (p_ctrl[3] >= 0.0):
            p_ch_ess = p_ctrl[3]
        else:
            p_dis_ess = p_ctrl[3]

            # delta_dod = -p_ctrl[3] * Paras['DeltaT'] / \
            #             (Paras['ESS_effi_dis'] * Paras['ESS_cap'])

            # f_dod = (1.06 * pow(delta_dod, 4) - \
            #          2.8 * pow(delta_dod, 3) + \
            #          2.66 * pow(delta_dod, 2) - \
            #          1.07 * delta_dod + 
            #          0.17) * 1e+5

        if (p_ch_ess > 0.0 or p_dis_ess == 0.0):
            p_ch_ess_max = min(p_max_ess, \
                           (ESS_soc_max - soc_ess) * Paras['ESS_cap'] / \
                           (Paras['DeltaT'] * Paras['ESS_effi_ch']))

            assert p_ch_ess_max >= 0

            if p_ch_ess > p_ch_ess_max:
                p_ch_ess = p_ch_ess_max
                p_ctrl[3] = p_ch_ess_max

        else:
            p_dis_ess_min = max(-p_max_ess, \
                           (ESS_soc_min - soc_ess) * Paras['ESS_cap'] * Paras['ESS_effi_dis'] / \
                           (Paras['DeltaT']))
            #assert p_dis_ess_min < 0

            if p_dis_ess_min > 0:
                p_dis_ess = 0
                p_ctrl[3] = 0
            elif p_dis_ess < p_dis_ess_min:
                p_dis_ess = p_dis_ess_min  # limit its saving power < 0
                p_ctrl[3] = p_dis_ess_min

            delta_dod = -p_dis_ess * Paras['DeltaT'] / \
                        (Paras['ESS_effi_dis'] * Paras['ESS_cap'])

            f_dod = (1.06 * pow(delta_dod, 4) - \
                     2.8 * pow(delta_dod, 3) + \
                     2.66 * pow(delta_dod, 2) - \
                     1.07 * delta_dod + 
                     0.17) * 1e+5

        soc_ess_next = soc_ess + \
                       Paras['ESS_effi_ch'] * p_ch_ess * Paras['DeltaT'] / Paras['ESS_cap'] + \
                       1 / Paras['ESS_effi_dis'] * p_dis_ess * Paras['DeltaT'] / Paras['ESS_cap']

        assert soc_ess_next <= ESS_soc_max

        self.state['soc_ess'] = np.array(soc_ess_next, dtype=np.float16)
        cost_soc = Paras['ESS_cost'] * (kappa * soc_ess - psi) / (Paras['ESS_Fmax'] * 15 * 365 * 24) * Paras['DeltaT'] / 60
        cost_ess = cost_soc
        if(f_dod != 0.0):
            cost_ess += Paras['ESS_cost'] * delta_dod / f_dod

        return [p_ctrl, [cost_hvac, cost_ewh, cost_ev, cost_ess]]

    def calNonControlPower(self, action):
        p_nctrl = np.zeros(self.T_ini_nocntl.shape, dtype=np.float16)
        time_to_start_noctrl = self.state['time_to_start_noctrl']

        for i in range(self.T_ini_nocntl.size):
            if time_to_start_noctrl[i] != 0:
                p_nctrl[i] = Power_nonctrl[i]

            if self.n_calls + 1 > self.T_ini_nocntl[i] and self.n_calls + 1 < self.T_end_nocntl[i]:
                self.state['time_to_start_noctrl'][i] = self.n_calls + 1 - self.T_ini_nocntl[i]
            else:
                self.state['time_to_start_noctrl'][i] = 0

        return p_nctrl

    def calGridPrice(self, p_shift, p_ctrl, p_nonc):
        cost_grid = 0.0

        unit_price = real_time_price_96[self.day*96 + self.n_calls + 32]
        solar_power = PV_t_96[self.day*96 + self.n_calls + 32]
        grid_power = np.sum(p_shift) + np.sum(p_ctrl) + np.sum(p_nonc) - solar_power # Kw watts 

        if (grid_power >=0 and grid_power <= Paras['PowerGridMax']):
            cost_grid = unit_price
        elif (grid_power > Paras['PowerGridMax']):
            cost_grid = Paras['IBR'] * unit_price
        else:
            cost_grid = 0.9 * unit_price

        return grid_power * cost_grid * Paras['DeltaT']

    def step(self, action):
        if self.n_calls == Paras['T_end']:
            self.reset()

        Energy_shift = self.calShiftPower(action)  # Only energy consumption, no cost
        Energy_cntl = self.calControlPower(action)  # [[powers], [costs]]
        Energy_noncntl = self.calNonControlPower(action) # Only energy consumption

        grid_cost = self.calGridPrice(Energy_shift, Energy_cntl[0], Energy_noncntl)

        reward = - np.sum(Energy_cntl[1]) - grid_cost # Reward function

        if self.day*96 + self.n_calls + 1 + 32 < 62 * Paras['T_end']:
            self.state['pv_power'] = PV_t_norm_96[self.day*96 + self.n_calls + 1 + 32]

        self.state['grid_price'] = real_time_price_norm_96[(self.day - 1)*96 + self.n_calls + 2 + 32:self.day*96+self.n_calls+32+2]
        self.state['solar'] = solar_norm_96[(self.day - 1)*96 + self.n_calls + 2 + 32:self.day*96+self.n_calls+32+2]
        self.state['temp_out'] = temp_out_norm_96[(self.day - 1)*96 + self.n_calls + 2 + 32:self.day*96+self.n_calls+32+2]

        terminated = (self.n_calls == Paras['T_end'] - 1)
        truncated = False
        info = {}
        self.n_calls += 1
        self.cur_steps += 1

        return self.state, reward, terminated, truncated, info

if __name__ == "__main__":
    reward_callback = RewardPlotCallback(plot_interval=Paras['T_end'],
                                         episodes=Paras['Mep'],
                                         steps_per_episode=interval
                                         )
    policy_kwargs = dict(net_arch = [128, 128, 128]) 

    env = CustomEnv()
    env = FlattenObservation(env)
    env = FlattenActionWrapper(env)
    check_env(env)

    eval_episodes = 1
    shift_apps = np.zeros((eval_episodes, K_shift.size, Paras['T_end']))
    temp_apps = np.zeros((eval_episodes, 2, Paras['T_end']))
    soc_power_ev = np.zeros((eval_episodes, 2, Paras['T_end']))
    soc_power_ess = np.zeros((eval_episodes, 2, Paras['T_end']))

    if len(sys.argv) == 2:
        zip_file = sys.argv[1]
        if not os.path.isfile(zip_file):
            print(f"Error: The file '{zip_file}' does not exist.")
            sys.exit(1)

        model = PPO.load(zip_file)
        total_reward = np.array([])
        for i in range(eval_episodes):
            obs, info = env.reset()
            shift_prog_pos_index = np.array([0, 4, 11])
            ess_prev_soc = 0.0
            episode_reward = 0.0
            done = False
            _X = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                temp_Y = np.array([obs[96]+Paras['HVAC_set'], obs[-1]+Paras['EWH_set']])  # [hvac_temp, ewh_temp]

                ev_temp_pow = MND( np.array([ ( 1 + action[2] ) * p_max_ev / 2 ]), np.array([ (1 + action[3]) * pow(p_max_ev/2, 2) / 2 ]) )
                ev_SocPower_Y = np.array([obs[96+2+17+1], ev_temp_pow[0] ])  #[soc_ev, power_ev]

                ess_temp_pow = MND( np.array( [ action[0]*p_max_ess ] ), np.array( [ (1 + action[1]) * pow(p_max_ess/2, 2) / 2 ]) )
                ess_SocPower_Y = np.array([obs[96+2+17], ess_temp_pow[0] ])  #[soc_ess, power_ess] 
                if abs(ess_prev_soc - obs[96+2+17]) < 0.009:
                    ess_SocPower_Y[1] = 0.0
                ess_prev_soc = obs[96+2+17]

                temp_apps[i,:, _X] = temp_Y
                soc_power_ev[i, :, _X] = ev_SocPower_Y
                soc_power_ess[i, :, _X] = ess_SocPower_Y
                obs, reward, done, _, _ = env.step(action)

                new_pos_index = np.where(obs[98:115] == 1)[0]
                shift_Y = (new_pos_index > shift_prog_pos_index).astype(int)
                shift_apps[i, :, _X] = shift_Y
                shift_prog_pos_index = new_pos_index

                episode_reward += reward
                _X += 1
               
            total_reward = np.append(total_reward, episode_reward)  #if above 'model.predict(deterministic=True)' all the values in the array are the same
        plotsAll(shift_apps, Power_shift, K_shift, T_ini_shift, T_end_shift, real_time_price_96, temp_apps, temp_out_96, soc_power_ev, t_ini_ev, t_end_ev, soc_power_ess, PV_t_96)
    else:
        model = PPO("MlpPolicy", env, verbose=1, device='cpu', policy_kwargs=policy_kwargs, \
                    n_steps=interval, vf_coef=0.128, learning_rate=1.4e-4,\
                    batch_size=Paras['T_end'], ent_coef=0.01, gamma=0.995, \
                    clip_range=0.2, gae_lambda=0.97, n_epochs=15)

        model.learn(total_timesteps=total_steps, callback=reward_callback)
        model.save("smartHomePPO")
        del model

    env.close()
