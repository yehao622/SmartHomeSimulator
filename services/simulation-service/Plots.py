from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

from parameters import Paras

class RewardPlotCallback(BaseCallback):
  def __init__(self, plot_interval=96, episodes=2500, steps_per_episode=288, rank=-1, verbose=1):
    super(RewardPlotCallback, self).__init__(verbose)
    self.plot_interval = plot_interval
    self.steps_per_episode = steps_per_episode
    self.total_episodes = episodes
    self.rewards_per_chunk = []  # Store rewards for each 96-step chunk
    self.episode_rewards = []    # Store average rewards for each episode
    self.current_chunk_rewards = []
    self.current_episode = 0
    self.step_in_episode = 0
    self.rank = rank

  def _on_step(self) -> bool:
    # Collect reward for current step
    self.current_chunk_rewards.append(self.locals["rewards"][0])
    self.step_in_episode += 1

    # If we've collected 96 steps worth of rewards
    if len(self.current_chunk_rewards) == self.plot_interval:
        chunk_avg = np.sum(self.current_chunk_rewards)# * self.plot_interval
        self.rewards_per_chunk.append(chunk_avg)
        self.current_chunk_rewards = []

    # If we've completed an episode
    if self.step_in_episode >= self.steps_per_episode:
        self.episode_rewards.append(np.mean(self.rewards_per_chunk))
        self.plot_rewards()
        # Reset for next episode
        self.rewards_per_chunk = []
        self.step_in_episode = 0
        self.current_episode += 1

    return True
    
  def plot_rewards(self):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(self.episode_rewards)), self.episode_rewards, label="Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean rewards per 96-step chunk ($)")
    plt.xticks([0, 500, 1000, 1500, 2000, 2500])
    plt.yticks([-300, -250, -200.0, -150, -100, -50, 0])
    plt.title(f"Reward Plot at Episode {self.current_episode}")
    if self.rank == -1:
        plt.savefig("reward_plot.png", dpi=300)
    else:
        plt.savefig(f"reward_plot_{self.rank}.png", dpi=300)
    plt.close()

def concateXY(x_arr, y_arr, pos=16):
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    tmp_x = np.concatenate((x_arr[-pos:], x_arr[:-pos]))
    new_x_arr = np.array([str(v) for v in tmp_x])
    new_y_arr = np.concatenate((y_arr[-pos:], y_arr[:-pos]))

    return new_x_arr, new_y_arr

def updateDryerBound(Y_wash, _ini_dryer, _end_dryer, start, end, prog_wash):
    prog = 0
    for t in range(start, end):
        if Y_wash[t] > 0:
            prog += 1
        else:
            prog = 0
        
        if prog == prog_wash:
            return t+1, t+9
    
    return _ini_dryer, _end_dryer
    
def plotsAll(shift_apps, Power_shift, K_shift, T_ini_shift, T_end_shift, real_time_price_96, temp_apps, temp_out_96, soc_power_ev, t_ini_ev, t_end_ev,\
             soc_power_ess, PV_t_96):
    shift_labels = np.array(["DishWasher", "WashingMachine", "ClothesDryer"])
    fig, axes = plt.subplots(K_shift.size+1, 1, figsize=(18, 10), sharex=True)
    X = np.array([str(x) for x in range(Paras['T_end'])])
    xticks = [i for i in X if int(i) % 4 == 0]
    for app_ind, ax in zip(range(K_shift.size+1), axes):
        if app_ind < K_shift.size:
            min_x = str(T_ini_shift[app_ind])
            max_x = str(T_end_shift[app_ind])

            #for _iter in range(shift_apps.shape[0]):
            Y = (shift_apps[:, app_ind, :].mean(axis=0) >= 0.5).astype(int) * Power_shift[app_ind]
            if app_ind == 1:
                T_ini_shift[2], T_end_shift[2] = updateDryerBound(Y, T_ini_shift[2], T_end_shift[2], T_ini_shift[1], T_end_shift[1], K_shift[1])
            ax.step(X, Y, where='post', linewidth=1, label=f"{app_ind}-th app")

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            #ax.set_xlabel("Time")
            ax.set_ylabel("Power (kW)")
            ax.axvline(x=min_x, color="red", linestyle="--", label="_nolegend_")
            ax.axvline(x=max_x, color="red", linestyle="--", label="_nolegend_")
            ax.axvspan(min_x, max_x, color="red", alpha=0.2, label="_nolegend_")
            ax.set_title(shift_labels[app_ind])
            #ax.legend()
        else:
            Y = real_time_price_96[96+32:96*2+32]
            ax.set_xlabel("Time")
            ax.set_ylabel("Price ($/kWh)")
            ax.step(X, Y, where='post', linewidth=1, label="Price")

    plt.savefig(f"shift_apps_power.png", dpi=300)
    
    control_apps = np.array(["HAVC", "EWH", "EV_SoC", "EV_Price", "ESS_SoC", "ESS_Price"])
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, ax in zip(range(2), axes):
        #for _iter in range(temp_apps.shape[0]):
        Y = temp_apps[:, i, :].mean(axis=0) #[_iter][i]
        ax.plot(X, Y, linewidth=1, label=f"{i}-th app") 

        if i == 0:
            ax.axhline(y=Paras['HVAC_set'], color="orange", linestyle="--", label="HVAC set Temperature") 
            ax2 = ax.twinx()
            ax2.plot(X, temp_out_96[96+32:96*2+32], linewidth=1, color="gold", label="Outdoor temperature")
            ax.set_ylabel("Indoor Temperature(\u00B0C)")
            ax2.set_ylabel("Outdoor Temperature(\u00B0C)")
            handle1, label1 = ax.get_legend_handles_labels()
            handle2, label2 = ax2.get_legend_handles_labels()
            handles = handle1 + handle2
            labels = label1 + label2
            ax.set_title("HVAC")
            #ax.legend(handles, labels, loc='upper right')
        else:
            ax.plot(X, Y, linewidth=1, label="Water temperature") 
            ax.set_ylabel("Water Temperature(\u00B0C)")
            ax.axhline(y=Paras['EWH_set'], color="orange", linestyle="--", label="Set Temperature")
            handle1, label1 = ax.get_legend_handles_labels()
            #ax.legend(handle1, label1, loc='upper right')
            ax.set_title("EWH")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)

    plt.savefig(f"Temperature.png", dpi=300)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, ax in zip(range(2), axes):
        #for _iter in range(soc_power_ev.shape[0]):
        Y = soc_power_ev[:, i, :].mean(axis=0) #[_iter][i]
        if i == 0:
            #ax.plot(X, Y, linewidth=1, label="EV SoC")
            ax.step(X, Y, where='post', linewidth=1, label="EV SoC")
        else:
            ax.bar(X, Y, alpha=0.7, label="EV Power")        
            
        if i == 0:
            ax.set_ylabel("SoC (%)")
            handle1, label1 = ax.get_legend_handles_labels()
            #ax.legend(handle1, label1, loc='upper right')
            ax.set_title("EV SoC")
        else:
            ax2 = ax.twinx()
            ax2.plot(X, real_time_price_96[96+32:96*2+32], linewidth=1, color="blue", label="Price")
            ax.set_ylabel("Power (kW)")
            ax2.set_ylabel("Price ($/kWh)")
            handle1, label1 = ax.get_legend_handles_labels()
            handle2, label2 = ax2.get_legend_handles_labels()
            handles = handle1 + handle2
            labels = label1 + label2
            #ax.legend(handles, labels, loc='upper right')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            ax.set_title("EV Power")

        ax.axvline(x=str(t_ini_ev), color="red", linestyle="--")
        ax.axvline(x=str(t_end_ev), color="red", linestyle="--")
        ax.axvspan(str(t_ini_ev), str(t_end_ev), color="red", alpha=0.2)
    
    plt.savefig(f"EV.png", dpi=300)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, ax in zip(range(2), axes):
        #for _iter in range(soc_power_ess.shape[0]):
        Y = soc_power_ess[:, i, :].mean(axis=0) #[_iter][i]
        if i == 0:
            ax.step(X, Y, where='post', linewidth=1, label="ESS SoC")
        else:
            ax.bar(X, Y, alpha=0.7, label="ESS Power")

        if i == 0:
            ax2 = ax.twinx()
            ax2.step(X, PV_t_96[96+32:96*2+32], where='post', linewidth=1, color="gray", label="PV Output")
            ax.set_ylabel("SoC (%)")
            ax2.set_ylabel("Power (kW)")
            handle1, label1 = ax.get_legend_handles_labels()
            handle2, label2 = ax2.get_legend_handles_labels()
            handles = handle1 + handle2
            labels = label1 + label2
            #ax.legend(handles, labels, loc='upper right')
        else:
            ax.bar(X, Y, color="green", alpha=0.7, label="ESS Power")
            ax.axhline(y=0, color="orange", linestyle="--", label="Set Temperature")
            ax2 = ax.twinx()
            ax2.step(X, real_time_price_96[96+32:96*2+32], where='post', linewidth=1, color="blue", label="Price ($/kWh)")
            ax.set_ylabel("Power (kW)")
            ax2.set_ylabel("Price ($/kWh)")
            handle1, label1 = ax.get_legend_handles_labels()
            handle2, label2 = ax2.get_legend_handles_labels()
            handles = handle1 + handle2
            labels = label1 + label2
            #ax.legend(handles, labels, loc='upper right')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)

    plt.savefig(f"ESS.png", dpi=300)
