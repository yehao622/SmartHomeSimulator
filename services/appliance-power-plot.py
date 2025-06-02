import matplotlib.pyplot as plt
import numpy as np
import csv

def read_csv_file(file_path):
    """
    Read data from the CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with the parsed data
    """
    try:
        # Open and read the CSV file
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Get the header row

            # Initialize a dictionary to store data for each column
            data_dict = {'headers': headers}
            
            # Initialize arrays for each column
            for header in headers:
                data_dict[header] = []
            
            # Read rows and store in column arrays
            for row_idx, row in enumerate(reader):
                if not row:  # Skip empty rows
                    continue
                
                # Add data for each column
                for col_idx, value in enumerate(row):
                    if col_idx < len(headers):  # Ensure we don't go out of bounds
                        header = headers[col_idx]
                        if header == 'TimeStep':
                            data_dict[header].append(int(value))
                        else:
                            data_dict[header].append(float(value))

            # Convert lists to numpy arrays
            for header in headers:
                data_dict[header] = np.array(data_dict[header])

            data_dict['timesteps'] = data_dict['TimeStep']

            return data_dict    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def create_appliance_plots(data):
    """Create the 4-subplot figure with appliance data"""
    # Get data from dictionary
    timesteps = data['timesteps']
    prices = data['prices']
    dishwasher_active = data['dishwasher']
    washing_machine_active = data['washing_machine']
    clothes_dryer_active = data['clothes_dryer']
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    prices = prices[sort_indices]
    dishwasher_active = dishwasher_active[sort_indices]
    washing_machine_active = washing_machine_active[sort_indices]
    clothes_dryer_active = clothes_dryer_active[sort_indices]
    
    # Convert active status to power levels
    dishwasher_power = dishwasher_active * 1.8  # 1.8 kW when active
    washing_machine_power = washing_machine_active * 0.4  # 0.4 kW when active
    clothes_dryer_power = clothes_dryer_active * 1.2  # 1.2 kW when active
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Electricity Price
    # axs[0].plot(timesteps, prices, 'b-', linewidth=2)
    axs[0].step(timesteps, prices, where='post', color='red', linewidth=2)
    axs[0].set_ylabel('Price ($/kWh)')
    axs[0].set_xlim(0, 95)
    # axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].text(90, max(prices) * 0.9, 'Price', color='blue', fontweight='bold')
    
    # Plot 2: Dishwasher
    axs[1].step(timesteps, dishwasher_power, where='post', color='orange', linewidth=2)
    axs[1].fill_between(timesteps, dishwasher_power, step="post", alpha=0.2, color='orange')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_ylim(0, 1.9)
    axs[1].text(90, 1.7, 'Dishwasher', color='orange', fontweight='bold')
    
    # Plot 3: Washing Machine
    axs[2].step(timesteps, washing_machine_power, where='post', color='purple', linewidth=2)
    axs[2].fill_between(timesteps, washing_machine_power, step="post", alpha=0.2, color='purple')
    axs[2].set_ylabel('Power (kW)')
    axs[2].set_ylim(0, 0.5)
    axs[2].text(90, 0.45, 'Washing Machine', color='purple', fontweight='bold')
    
    # Plot 4: Clothes Dryer
    axs[3].step(timesteps, clothes_dryer_power, where='post', color='green', linewidth=2)
    axs[3].fill_between(timesteps, clothes_dryer_power, step="post", alpha=0.2, color='green')
    axs[3].set_ylabel('Power (kW)')
    axs[3].set_ylim(0, 1.3)
    axs[3].text(90, 1.2, 'Clothes Dryer', color='green', fontweight='bold')
    axs[3].set_xlabel('Time Step (15-minute intervals)')
    

    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    plt.xticks(x_ticks)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    
    # Save the figure
    plt.savefig('appliance_power_and_price.png', dpi=300, bbox_inches='tight')
    
    return fig

def compare_temperature_plot(rl_data, user_data):
    print(f"##################### {rl_data} ##################################33")
    """Compare indoor and outdoor temperatures between RL and user data"""
    # Get data from dictionaries
    rl_timesteps = rl_data['timesteps']
    user_timesteps = user_data['timesteps']
    
    # Ensure required columns exist in both datasets
    required_cols = ['IndoorTemperature', 'OutdoorTemperature']
    for dataset, name in [(rl_data, 'RL'), (user_data, 'User')]:
        for col in required_cols:
            if col not in dataset:
                print(f"Warning: '{col}' not found in {name} data")
                # You would need code here to read these columns from CSV
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, 95)
    
    # Plot indoor temperatures on the left y-axis
    ax1.set_xlabel('Time Step (15-minute intervals)')
    ax1.set_ylabel('Indoor Temperature (°C)')
    
    # Plot RL indoor temperature
    rl_sort_indices = np.argsort(rl_timesteps)
    rl_indoor_temp = rl_data['IndoorTemperature'][rl_sort_indices]
    line1 = ax1.plot(rl_timesteps[rl_sort_indices], rl_indoor_temp, 
                    color='tab:blue', linewidth=2, label='RL Indoor Temperature')
    
    # Plot User indoor temperature
    user_sort_indices = np.argsort(user_timesteps)
    user_indoor_temp = user_data['IndoorTemperature'][user_sort_indices]
    line2 = ax1.plot(user_timesteps[user_sort_indices], user_indoor_temp, 
                    color='tab:green', linewidth=2, linestyle='--', label='Manual Indoor Temperature')
    
    # Add horizontal comfort line at y=20
    line3 = ax1.axhline(y=20, color='black', linestyle=':', linewidth=1.5, label='Comfort Temperature (20°C)')
    
    # Create second y-axis for outdoor temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel('Outdoor Temperature (°C)')
    
    # Since outdoor temperature should be the same in both datasets, we'll use RL data
    rl_outdoor_temp = rl_data['OutdoorTemperature'][rl_sort_indices]
    line4 = ax2.plot(rl_timesteps[rl_sort_indices], rl_outdoor_temp, 
                    color='tab:red', linewidth=2, label='Outdoor Temperature')
    
    # Add legend
    lines = line1 + line2 + [line3] + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Add title
    plt.title('Indoor Temperature Comparison: RL vs Manual Control', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('temperature_comparison_rl_vs_manual.png', dpi=300, bbox_inches='tight')
    
    return fig

def compare_water_temperature_plot(rl_data, user_data):
    """Compare water temperature between RL and user data"""
    # Get data from dictionaries
    rl_timesteps = rl_data['timesteps']
    user_timesteps = user_data['timesteps']
    
    # Ensure water temperature column exists in both datasets
    if 'WaterTemperature' not in rl_data or 'WaterTemperature' not in user_data:
        print("Warning: WaterTemperature not found in one or both datasets")
        # You would need code here to read these columns from CSV
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, 95)
    
    # Plot water temperatures
    ax.set_xlabel('Time Step (15-minute intervals)')
    ax.set_ylabel('Water Temperature (°C)')
    
    # Plot RL water temperature
    rl_sort_indices = np.argsort(rl_timesteps)
    rl_water_temp = rl_data['WaterTemperature'][rl_sort_indices]
    ax.plot(rl_timesteps[rl_sort_indices], rl_water_temp, 
            color='tab:blue', linewidth=2, label='RL Water Temperature')
    
    # Plot User water temperature
    user_sort_indices = np.argsort(user_timesteps)
    user_water_temp = user_data['WaterTemperature'][user_sort_indices]
    ax.plot(user_timesteps[user_sort_indices], user_water_temp, 
            color='tab:green', linewidth=2, linestyle='--', label='Manual Water Temperature')
    
    # Add horizontal comfort line at y=60
    ax.axhline(y=60, color='red', linestyle=':', linewidth=1.5, label='Comfort Temperature (60°C)')
    
    # Add legend
    ax.legend(loc='best')
    
    # Add title
    plt.title('Water Temperature Comparison: RL vs Manual Control', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('water_temperature_comparison_rl_vs_manual.png', dpi=300, bbox_inches='tight')
    
    return fig

def compare_ev_charging_plot(rl_data, user_data):
    """Compare EV charging between RL and user data"""
    # Get data from dictionaries
    rl_timesteps = rl_data['timesteps']
    user_timesteps = user_data['timesteps']
    
    # Ensure required columns exist in both datasets
    required_cols = ['evSOC', 'EVChargerPowerNew', 'ElectricityPrice']
    for dataset, name in [(rl_data, 'RL'), (user_data, 'User')]:
        for col in required_cols:
            if col not in dataset:
                print(f"Warning: '{col}' not found in {name} data")
                # You would need code here to read these columns from CSV
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    plt.xticks(x_ticks)
    plt.xlim(0, 95)
    
    # Sort indices for both datasets
    rl_sort_indices = np.argsort(rl_timesteps)
    user_sort_indices = np.argsort(user_timesteps)
    
    # Get sorted data
    rl_ev_soc = rl_data['evSOC'][rl_sort_indices]
    rl_ev_power = rl_data['EVChargerPowerNew'][rl_sort_indices]
    user_ev_soc = user_data['evSOC'][user_sort_indices]
    user_ev_power = user_data['EVChargerPowerNew'][user_sort_indices]
    
    # Since electricity price should be the same, we'll use RL data
    electricity_price = rl_data['ElectricityPrice'][rl_sort_indices]
    
    # Plot 1: EV State of Charge
    ax1.plot(rl_timesteps[rl_sort_indices], rl_ev_soc, 
             color='tab:blue', linewidth=2, label='RL EV SoC')
    ax1.plot(user_timesteps[user_sort_indices], user_ev_soc, 
             color='tab:green', linewidth=2, linestyle='--', label='Manual EV SoC')
    ax1.set_ylabel('State of Charge (%)')
    ax1.set_title('EV State of Charge: RL vs Manual Control')
    ax1.grid(False)
    ax1.legend(loc='upper left')
    
    # Plot 2: EV Charging Power and Electricity Price
    width = 0.35  # width of bars
    
    # Calculate adjusted positions for RL and user bars
    rl_positions = rl_timesteps[rl_sort_indices] - width/2
    user_positions = user_timesteps[user_sort_indices] + width/2
    
    # Plot charging power bars
    ax2.bar(rl_positions, rl_ev_power, width, color='tab:blue', alpha=0.7, label='RL Charging Power')
    ax2.bar(user_positions, user_ev_power, width, color='tab:green', alpha=0.7, label='Manual Charging Power')
    ax2.set_xlabel('Time Step (15-minute intervals)')
    ax2.set_ylabel('Charging Power (kW)')
    ax2.grid(False)
    
    # Add second y-axis for electricity price
    ax3 = ax2.twinx()
    ax3.plot(rl_timesteps[rl_sort_indices], electricity_price, 
             color='tab:red', linewidth=2, label='Electricity Price')
    ax3.set_ylabel('Electricity Price ($/kWh)', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    
    # Add legend for second subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title('EV Charging Power and Electricity Price: RL vs Manual Control')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the figure
    plt.savefig('ev_charging_comparison_rl_vs_manual.png', dpi=300, bbox_inches='tight')
    
    return fig

def compare_solar_battery_plot(rl_data, user_data):
    """Compare solar power and battery usage between RL and user data"""
    # Get data from dictionaries
    rl_timesteps = rl_data['timesteps']
    user_timesteps = user_data['timesteps']
    
    # Ensure required columns exist in both datasets
    required_cols = ['SolarPower', 'BatterySOC', 'ElectricityPrice', 'BatteryChargerPowerNew']
    for dataset, name in [(rl_data, 'RL'), (user_data, 'User')]:
        for col in required_cols:
            if col not in dataset:
                print(f"Warning: '{col}' not found in {name} data")
                # You would need code here to read these columns from CSV
    
    # Sort indices for both datasets
    rl_sort_indices = np.argsort(rl_timesteps)
    user_sort_indices = np.argsort(user_timesteps)
    
    # Get sorted data
    rl_solar_power = rl_data['SolarPower'][rl_sort_indices]
    rl_battery_soc = rl_data['BatterySOC'][rl_sort_indices]
    rl_battery_power = rl_data['BatteryChargerPowerNew'][rl_sort_indices]
    
    user_solar_power = user_data['SolarPower'][user_sort_indices]
    user_battery_soc = user_data['BatterySOC'][user_sort_indices]
    user_battery_power = user_data['BatteryChargerPowerNew'][user_sort_indices]
    
    # Since electricity price should be the same, we'll use RL data
    electricity_price = rl_data['ElectricityPrice'][rl_sort_indices]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    plt.xticks(x_ticks)
    plt.xlim(0, 95)
    
    # Plot 1: Solar Power and Battery SOC
    # First plot solar power (should be the same for both)
    ax1_twin = ax1.twinx()
    solar_line = ax1_twin.plot(rl_timesteps[rl_sort_indices], rl_solar_power, 
                             color='tab:orange', linewidth=2, label='Solar Power')
    ax1_twin.set_ylabel('Solar Power (kW)', color='tab:orange')
    ax1_twin.tick_params(axis='y', labelcolor='tab:orange')
    
    # Now plot battery SOC for both RL and user
    rl_soc_line = ax1.plot(rl_timesteps[rl_sort_indices], rl_battery_soc, 
                           color='tab:blue', linewidth=2, label='RL Battery SOC')
    user_soc_line = ax1.plot(user_timesteps[user_sort_indices], user_battery_soc, 
                            color='tab:green', linewidth=2, linestyle='--', label='Manual Battery SOC')
    ax1.set_ylabel('Battery SOC (%)')
    ax1.grid(False)
    
    # Add legend for first subplot
    lines1 = rl_soc_line + user_soc_line
    labels1 = [l.get_label() for l in lines1]
    lines2 = solar_line
    labels2 = [l.get_label() for l in lines2]
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Solar Power and Battery State of Charge: RL vs Manual Control')
    
    # Plot 2: Battery Power and Electricity Price
    width = 0.35  # width of bars
    
    # Calculate adjusted positions for RL and user bars
    rl_positions = rl_timesteps[rl_sort_indices] - width/2
    user_positions = user_timesteps[user_sort_indices] + width/2
    
    # Plot battery power bars
    rl_bars = ax2.bar(rl_positions, rl_battery_power, width, color='tab:blue', alpha=0.7, label='RL Battery Power')
    user_bars = ax2.bar(user_positions, user_battery_power, width, color='tab:green', alpha=0.7, label='Manual Battery Power')
    
    # Color the bars based on charging/discharging
    for bars, data in [(rl_bars, rl_battery_power), (user_bars, user_battery_power)]:
        for i, bar in enumerate(bars):
            if data[i] < 0:  # Discharging
                bar.set_color('tab:red' if bars == rl_bars else 'darkred')
                bar.set_alpha(0.7)
    
    ax2.set_xlabel('Time Step (15-minute intervals)')
    ax2.set_ylabel('Battery Power (kW)')
    ax2.grid(False)
    
    # Add second y-axis for electricity price
    ax2_twin = ax2.twinx()
    price_line = ax2_twin.plot(rl_timesteps[rl_sort_indices], electricity_price, 
                              color='tab:purple', linewidth=2, label='Electricity Price')
    ax2_twin.set_ylabel('Electricity Price ($/kWh)', color='tab:purple')
    ax2_twin.tick_params(axis='y', labelcolor='tab:purple')
    
    # Create custom legend entries
    from matplotlib.patches import Patch
    rl_charge_patch = Patch(color='tab:blue', alpha=0.7, label='RL Charging')
    rl_discharge_patch = Patch(color='tab:red', alpha=0.7, label='RL Discharging')
    user_charge_patch = Patch(color='tab:green', alpha=0.7, label='Manual Charging')
    user_discharge_patch = Patch(color='darkred', alpha=0.7, label='Manual Discharging')
    price_line_legend = Patch(color='tab:purple', alpha=1.0, label='Electricity Price')
    
    ax2.legend(handles=[rl_charge_patch, rl_discharge_patch, user_charge_patch, user_discharge_patch, price_line_legend], loc='upper right')
    ax2.set_title('Battery Power and Electricity Price: RL vs Manual Control')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the figure
    plt.savefig('solar_battery_comparison_rl_vs_manual.png', dpi=300, bbox_inches='tight')
    
    return fig

def compare_grid_power_price_plot(rl_data, user_data):
    """Compare grid power usage between RL and user data"""
    # Get data from dictionaries
    rl_timesteps = rl_data['timesteps']
    user_timesteps = user_data['timesteps']
    
    # Ensure required columns exist in both datasets
    required_cols = ['GridPower', 'ElectricityPrice']
    for dataset, name in [(rl_data, 'RL'), (user_data, 'User')]:
        for col in required_cols:
            if col not in dataset:
                print(f"Warning: '{col}' not found in {name} data")
                # You would need code here to read these columns from CSV
    
    # Sort indices for both datasets
    rl_sort_indices = np.argsort(rl_timesteps)
    user_sort_indices = np.argsort(user_timesteps)
    
    # Get sorted data
    rl_grid_power = rl_data['GridPower'][rl_sort_indices]
    user_grid_power = user_data['GridPower'][user_sort_indices]
    
    # Since electricity price should be the same, we'll use RL data
    electricity_price = rl_data['ElectricityPrice'][rl_sort_indices]
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, 95)
    
    # Plot grid power as histogram bars
    width = 0.35  # width of bars
    
    # Calculate adjusted positions for RL and user bars
    rl_positions = rl_timesteps[rl_sort_indices] - width/2
    user_positions = user_timesteps[user_sort_indices] + width/2
    
    # Plot grid power bars
    rl_bars = ax1.bar(rl_positions, rl_grid_power, width, color='tab:blue', alpha=0.7, label='RL Grid Power')
    user_bars = ax1.bar(user_positions, user_grid_power, width, color='tab:green', alpha=0.7, label='Manual Grid Power')
    
    # Color the bars based on import/export
    for bars, data in [(rl_bars, rl_grid_power), (user_bars, user_grid_power)]:
        for i, bar in enumerate(bars):
            if data[i] < 0:  # Export to grid
                bar.set_color('tab:red' if bars == rl_bars else 'darkred')
                bar.set_alpha(0.7)
    
    ax1.set_xlabel('Time Step (15-minute intervals)')
    ax1.set_ylabel('Grid Power (kW)')
    ax1.grid(False)
    
    # Add second y-axis for electricity price
    ax2 = ax1.twinx()
    ax2.plot(rl_timesteps[rl_sort_indices], electricity_price, color='tab:purple', linewidth=2, label='Electricity Price')
    ax2.set_ylabel('Electricity Price ($/kWh)', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    # Create custom legend entries
    from matplotlib.patches import Patch
    rl_import_patch = Patch(color='tab:blue', alpha=0.7, label='RL Import')
    rl_export_patch = Patch(color='tab:red', alpha=0.7, label='RL Export')
    user_import_patch = Patch(color='tab:green', alpha=0.7, label='Manual Import')
    user_export_patch = Patch(color='darkred', alpha=0.7, label='Manual Export')
    price_line_legend = Patch(color='tab:purple', alpha=1.0, label='Electricity Price')
    
    ax1.legend(handles=[rl_import_patch, rl_export_patch, user_import_patch, user_export_patch, price_line_legend], loc='upper right')
    
    # Add title
    plt.title('Grid Power and Electricity Price: RL vs Manual Control', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('grid_power_comparison_rl_vs_manual.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_temperature_plot(data):
    """Create a plot showing indoor and outdoor temperatures with comfort line"""
    # Get data from dictionary
    timesteps = data['timesteps']
    
    # Check if temperature columns exist
    if 'IndoorTemperature' not in data or 'OutdoorTemperature' not in data:
        # Read temperature data from CSV if not already in data dictionary
        print("Reading temperature data from CSV...")
        indoor_temp = []
        outdoor_temp = []
        
        # This assumes you've already opened the CSV file and have headers
        # You'll need to adapt this to your actual data loading code
        with open('rl_simulation_data.csv', 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            indoor_idx = headers.index('IndoorTemperature') if 'IndoorTemperature' in headers else -1
            outdoor_idx = headers.index('OutdoorTemperature') if 'OutdoorTemperature' in headers else -1
            
            if indoor_idx == -1 or outdoor_idx == -1:
                raise ValueError("Required temperature columns not found in CSV")
            
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                indoor_temp.append(float(row[indoor_idx]))
                outdoor_temp.append(float(row[outdoor_idx]))
            
            data['IndoorTemperature'] = np.array(indoor_temp)
            data['OutdoorTemperature'] = np.array(outdoor_temp)
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    indoor_temp = data['IndoorTemperature'][sort_indices]
    outdoor_temp = data['OutdoorTemperature'][sort_indices]
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, 95)
    
    # Plot indoor temperature on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time Step (15-minute intervals)')
    ax1.set_ylabel('Indoor Temperature (°C)', color=color)
    line1 = ax1.plot(timesteps, indoor_temp, color=color, linewidth=2, label='Indoor Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add horizontal comfort line at y=20
    ax1.axhline(y=20, color='green', linestyle='--', linewidth=1.5, label='Comfort Temperature (20°C)')
    
    # Create second y-axis for outdoor temperature
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Outdoor Temperature (°C)', color=color)
    line2 = ax2.plot(timesteps, outdoor_temp, color=color, linewidth=2, label='Outdoor Temperature')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Add title
    plt.title('Indoor vs Outdoor Temperature with Comfort Line', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('temperature_comparison.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_water_temperature_plot(data):
    """Create a plot showing water temperature with comfort line"""
    # Get data from dictionary
    timesteps = data['timesteps']
    
    # Check if water temperature column exists
    if 'WaterTemperature' not in data:
        # Read water temperature data from CSV if not already in data dictionary
        print("Reading water temperature data from CSV...")
        water_temp = []
        
        # This assumes you've already opened the CSV file and have headers
        with open('rl_simulation_data.csv', 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            water_temp_idx = headers.index('WaterTemperature') if 'WaterTemperature' in headers else -1
            
            if water_temp_idx == -1:
                raise ValueError("Required WaterTemperature column not found in CSV")
            
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                water_temp.append(float(row[water_temp_idx]))
            
            data['WaterTemperature'] = np.array(water_temp)
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    water_temp = data['WaterTemperature'][sort_indices]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, 95)
    
    # Plot water temperature
    color = 'tab:blue'
    ax.set_xlabel('Time Step (15-minute intervals)')
    ax.set_ylabel('Water Temperature (°C)')
    ax.plot(timesteps, water_temp, color=color, linewidth=2, label='Water Temperature')
    
    # Add horizontal comfort line at y=60
    ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label='Comfort Temperature (60°C)')
    
    # Add legend
    ax.legend(loc='best')
    
    # Add title
    plt.title('Water Temperature with Comfort Line', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('water_temperature.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_ev_charging_plot(data):
    """Create a figure with two subplots: EV SoC and EV charging power with electricity price"""
    # Get data from dictionary
    timesteps = data['timesteps']
    
    # Check if required columns exist
    required_cols = ['evSOC', 'EVChargerPowerNew', 'ElectricityPrice']
    for col in required_cols:
        if col not in data:
            # Read required data from CSV if not already in data dictionary
            print(f"Reading {col} data from CSV...")
            with open('rl_simulation_data.csv', 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)
                
                col_idx = headers.index(col) if col in headers else -1
                if col_idx == -1:
                    raise ValueError(f"Required column {col} not found in CSV")
                
                col_data = []
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                    col_data.append(float(row[col_idx]))
                
                data[col] = np.array(col_data)
                
                # Reset file pointer for next column read
                file.seek(0)
                next(reader)  # Skip header row
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    ev_soc = data['evSOC'][sort_indices]
    ev_power = data['EVChargerPowerNew'][sort_indices]
    electricity_price = data['ElectricityPrice'][sort_indices]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    plt.xticks(x_ticks)
    plt.xlim(0, 95)
    
    # Plot 1: EV State of Charge
    color = 'tab:blue'
    ax1.plot(timesteps, ev_soc, color=color, linewidth=2, label='EV State of Charge')
    ax1.set_ylabel('State of Charge (%)')
    ax1.set_title('EV State of Charge Over Time')
    ax1.grid(False)
    ax1.legend(loc='upper left')
    
    # Plot 2: EV Charging Power and Electricity Price
    color1 = 'tab:green'
    ax2.set_xlabel('Time Step (15-minute intervals)')
    ax2.set_ylabel('Charging Power (kW)', color=color1)
    ax2.bar(timesteps, ev_power, color=color1, alpha=0.7, label='EV Charging Power')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(False)
    
    # Add second y-axis for electricity price
    ax3 = ax2.twinx()
    color2 = 'tab:red'
    ax3.set_ylabel('Electricity Price ($/kWh)', color=color2)
    ax3.plot(timesteps, electricity_price, color=color2, linewidth=2, label='Electricity Price')
    ax3.tick_params(axis='y', labelcolor=color2)
    
    # Add legend for second subplot
    # This creates a transparent patch for both axes to use as legend handles
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    
    # Save the figure
    plt.savefig('ev_charging_analysis.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_solar_battery_plot(data):
    """Create a figure with two subplots: 
    1) Solar power and battery SOC
    2) Electricity price and battery charging power"""
    
    # Get data from dictionary
    timesteps = data['timesteps']
    
    # Check if required columns exist
    required_cols = ['SolarPower', 'BatterySOC', 'ElectricityPrice', 'BatteryChargerPowerNew']
    for col in required_cols:
        if col not in data:
            # Read required data from CSV if not already in data dictionary
            print(f"Reading {col} data from CSV...")
            with open('rl_simulation_data.csv', 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)
                
                col_idx = headers.index(col) if col in headers else -1
                if col_idx == -1:
                    raise ValueError(f"Required column {col} not found in CSV")
                
                col_data = []
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                    col_data.append(float(row[col_idx]))
                
                data[col] = np.array(col_data)
                
                # Reset file pointer for next column read
                file.seek(0)
                next(reader)  # Skip header row
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    solar_power = data['SolarPower'][sort_indices]
    battery_soc = data['BatterySOC'][sort_indices]
    electricity_price = data['ElectricityPrice'][sort_indices]
    battery_power = data['BatteryChargerPowerNew'][sort_indices]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    plt.xticks(x_ticks)
    plt.xlim(0, 95)
    
    # Plot 1: Solar Power and Battery SOC
    color1 = 'tab:green'
    ax1.set_ylabel('Battery SOC (%)', color=color1)
    ax1.plot(timesteps, battery_soc, color=color1, linewidth=2, label='Battery SOC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(False)
    
    # Add second y-axis for solar power
    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel('Solar Power (kW)', color=color2)
    ax1_twin.plot(timesteps, solar_power, color=color2, linewidth=2, label='Solar Power')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add legend for first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Solar Power and Battery State of Charge')
    
    # Plot 2: Electricity Price and Battery Charging Power
    color3 = 'tab:blue'
    ax2.set_xlabel('Time Step (15-minute intervals)')
    ax2.set_ylabel('Battery Power (kW)', color=color3)
    
    # Use bar chart for battery power with positive and negative values
    battery_bars = ax2.bar(timesteps, battery_power, color=color3, alpha=0.7, label='Battery Power')
    
    # Color the bars based on charging (positive) or discharging (negative)
    for i, bar in enumerate(battery_bars):
        if battery_power[i] < 0:  # Discharging
            bar.set_color('tab:red')
            bar.set_alpha(0.7)
    
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.grid(False)
    
    # Add second y-axis for electricity price
    ax2_twin = ax2.twinx()
    color4 = 'tab:purple'
    ax2_twin.set_ylabel('Electricity Price ($/kWh)', color=color4)
    ax2_twin.plot(timesteps, electricity_price, color=color4, linewidth=2, label='Electricity Price')
    ax2_twin.tick_params(axis='y', labelcolor=color4)
    
    # Add legend for second subplot with custom handles for battery charging/discharging
    from matplotlib.patches import Patch
    charge_patch = Patch(color='tab:blue', alpha=0.7, label='Battery Charging')
    discharge_patch = Patch(color='tab:red', alpha=0.7, label='Battery Discharging')
    price_line = plt.Line2D([0], [0], color=color4, linewidth=2, label='Electricity Price')
    
    ax2.legend(handles=[charge_patch, discharge_patch, price_line], loc='upper right')
    ax2.set_title('Battery Power and Electricity Price')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the figure
    plt.savefig('solar_battery_analysis.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_grid_power_price_plot(data):
    """Create a figure showing grid power (as histogram) and electricity price over time"""
    
    # Get data from dictionary
    timesteps = data['timesteps']
    
    # Check if required columns exist
    required_cols = ['GridPower', 'ElectricityPrice']
    for col in required_cols:
        if col not in data:
            # Read required data from CSV if not already in data dictionary
            print(f"Reading {col} data from CSV...")
            with open('rl_simulation_data.csv', 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)
                
                col_idx = headers.index(col) if col in headers else -1
                if col_idx == -1:
                    raise ValueError(f"Required column {col} not found in CSV")
                
                col_data = []
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                    col_data.append(float(row[col_idx]))
                
                data[col] = np.array(col_data)
                
                # Reset file pointer for next column read
                file.seek(0)
                next(reader)  # Skip header row
    
    # Ensure data is sorted by timestep
    sort_indices = np.argsort(timesteps)
    timesteps = timesteps[sort_indices]
    grid_power = data['GridPower'][sort_indices]
    electricity_price = data['ElectricityPrice'][sort_indices]
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Set x-ticks at intervals of 4 (0, 4, 8, 12, ..., 92)
    x_ticks = list(range(0, 96, 4))
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, 95)
    
    # Plot grid power as histogram bars on the left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time Step (15-minute intervals)')
    ax1.set_ylabel('Grid Power (kW)', color=color1)
    
    # Use bar chart for grid power with positive and negative values
    grid_bars = ax1.bar(timesteps, grid_power, color=color1, alpha=0.7, label='Grid Power')
    
    # Color the bars based on import (positive) or export (negative)
    for i, bar in enumerate(grid_bars):
        if grid_power[i] < 0:  # Export to grid
            bar.set_color('tab:green')
            bar.set_alpha(0.7)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(False)
    
    # Add second y-axis for electricity price
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Electricity Price ($/kWh)', color=color2)
    ax2.plot(timesteps, electricity_price, color=color2, linewidth=2, label='Electricity Price')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legend with custom handles for grid import/export
    from matplotlib.patches import Patch
    import_patch = Patch(color='tab:blue', alpha=0.7, label='Grid Import')
    export_patch = Patch(color='tab:green', alpha=0.7, label='Grid Export')
    price_line = plt.Line2D([0], [0], color=color2, linewidth=2, label='Electricity Price')
    
    ax1.legend(handles=[import_patch, export_patch, price_line], loc='upper right')
    
    # Add title
    plt.title('Grid Power and Electricity Price', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('grid_power_price.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_comparison_plots(rl_data, user_data):
    """Create comparison plots between RL model and user manual control"""
    
    # Ensure both datasets have the required columns
    print("Checking data consistency...")
    
    # List of all plotting functions (excluding create_appliance_plots)
    comparison_functions = [
        compare_temperature_plot,
        compare_water_temperature_plot,
        compare_ev_charging_plot,
        compare_solar_battery_plot,
        compare_grid_power_price_plot
    ]
    
    # Generate each comparison plot
    figures = []
    for i, func in enumerate(comparison_functions):
        try:
            print(f"Generating comparison plot {i+1}...")
            fig = func(rl_data, user_data)
            figures.append(fig)
            plt.figure(fig.number)
            plt.show()
        except Exception as e:
            print(f"Error generating comparison plot {i+1}: {e}")
    
    return figures

# Main execution flow
def main():
    try:
        # Read the CSV file - if this fails, the error will be reported and program will exit
        rl_data = read_csv_file('rl_simulation_data.csv')
        user_data = read_csv_file('user_simulation_data.csv')
        
        # Create the plots
        # fig1 = create_appliance_plots(data)
        # fig2 = create_temperature_plot(data)
        # fig3 = create_water_temperature_plot(data)
        # fig4 = create_ev_charging_plot(data)
        # fig5 = create_solar_battery_plot(data)
        # fig6 = create_grid_power_price_plot(data)
        comparison_figures = create_comparison_plots(rl_data, user_data)
        # plt.show()
        
        print("Generated plot from CSV data")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    main()