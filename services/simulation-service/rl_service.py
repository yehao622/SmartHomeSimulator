from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import os, sys, traceback
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from flask_socketio import SocketIO
from flask_cors import CORS

# Import your custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

 # Import the CustomEnv from PPO_RL.py
from PPO_RL import CustomEnv
from util import FlattenActionWrapper, MBD, MND
from parameters import Paras
from DataSample import (temp_out_min, temp_out_max, p_max_hvac, p_max_ewh, p_max_ev, p_max_ess, 
                        generateAllSchedule, tempPriceSolar)

   
# Get schedules and time bounds
T_ini_shift, T_end_shift, t_ini_ev, t_end_ev, T_ini_nocntl, T_end_nocntl = generateAllSchedule()
temp_out_96, temp_out_norm_96, real_time_price_96, real_time_price_norm_96, PV_t_96, PV_t_norm_96, solar_96, solar_norm_96 = tempPriceSolar()

# Power values from parameters
K_shift = np.array([Paras['K_DishWasher'], Paras['K_WashMachine'], Paras['K_ClothesDryer']], dtype=np.uint8)
Power_shift = np.array([Paras['DishWasher'][0], Paras['WashMachine'][0], Paras['ClothesDryer']])
Power_nonctrl = np.array([Paras['TV'][0], Paras['refrige'][0], Paras['light'][0], Paras['vacuum'][0], Paras['hairdryer'][0]])

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

@socketio.on('test_connection')
def handle_test(data):
    print("Received test connection:", data)
    socketio.emit('test_response', {'message': 'Hello from server'})
  
model = None
env = None

# Define the model loading code in a function to ensure proper sequencing
def load_trained_model():
    global model, env
    
    print("Attempting to load the trained model...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight.zip")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print(f"Current directory contents: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
        return False
        
    try:
        # Create the environment for the model first
        print("Creating environment for the model...")
        env = CustomEnv()
        env = FlattenObservation(env)
        env = FlattenActionWrapper(env)
        check_env(env)
        env.reset()
        
        # Get observation and action spaces
        # observation_space = env.observation_space
        # action_space = env.action_space
        
        # Now load the model with the correct environment
        print(f"Loading model from {model_path}...")
        model = PPO.load(
            model_path,
            env=env,
            device="cpu", 
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space
            }
        )
        print("Successfully loaded the trained model")
        return True
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print(traceback.format_exc())
        return False

# Call the function to load the model
success = load_trained_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "rl-service"})

@app.route('/test', methods=['GET'])
def test_route():
    """Simple test route to confirm the server is running"""
    return jsonify({"status": "Flask server is running", "files": os.listdir('/app/')})

active_requests = 0
MAX_CONCURRENT_REQUESTS = 5

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive current state from simulation, return model predictions
    """
    global active_requests, env
    
    # Add request throttling
    if active_requests > MAX_CONCURRENT_REQUESTS:
        return jsonify({"error": "Too many prediction requests"}), 429
    
    active_requests += 1

    try:
        data = request.json
        # print(f"************ {data} ***************")
        
        # Extract current simulation time and day
        current_time_step = data.get('timeStep', 0)  # 0-95 for a day
        day = data.get('day', 1)  # 1-60
        
        # Format state for model input based on the flattened observation space
        observation = format_state(data, current_time_step, day)
            
        # Check if model is loaded
        if model is not None:
            # Get model prediction
            action, _states = model.predict(observation, deterministic=True)
            # print(f"############ {action} #############")
            
            # Use environment's step function to update state
            next_state, reward, terminated, truncated, info = env.step(action)

            # Format the next state for the frontend
            response = format_response(next_state, action, current_time_step, day)

            return jsonify(response)
        else:
            raise Exception("Model not loaded successfully")
        
    finally:
        active_requests -= 1

# Add a new endpoint to run complete day simulation
@app.route('/generate_day_data', methods=['GET'])
def generate_day_data():
    """Generate and return a CSV file with a complete day of simulation data using the RL model"""
    try:
        obs, info = env.reset()
        
        # Data collection for a full day (96 time steps)
        simulation_data = []
        day = 1  # Start with day 1
        
        for step in range(96):            
            try:
                # Get prediction from RL model
                action, _states = model.predict(obs, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Process the action to get the full system state
                prediction = format_response(next_state, action, step, day)
                
                # Record the data
                data_row = {
                    'TimeStep': step,
                    'SolarPower': prediction['environment']['solar_production'],
                    'GridPower': prediction['energy_flow']['grid']['net_power'],
                    'ElectricityPrice': prediction['environment']['price'],
                    'BatterySOC': prediction['battery']['soc'] * 100,  # Convert to percentage
                    'evSOC': prediction['ev']['soc'] * 100,
                    'IndoorTemperature': prediction['temperatures']['home']['current'],
                    'OutdoorTemperature': prediction['environment']['outside_temp'],
                    'WaterTemperature': prediction['temperatures']['water']['current'],
                    'HVACPower': prediction['appliances']['controllable']['hvac']['power'] if prediction['appliances']['controllable']['hvac']['active'] else 0,
                    'WaterHeaterPower': prediction['appliances']['controllable']['water_heater']['power'] if prediction['appliances']['controllable']['water_heater']['active'] else 0,
                    'EVChargerPower': prediction['ev']['power'] if prediction['ev']['connected'] else 0,
                    'DishwasherActive': 1 if prediction['appliances']['shiftable']['dishwasher']['active'] else 0,
                    'WashMachineActive': 1 if prediction['appliances']['shiftable']['wash_machine']['active'] else 0,
                    'ClothesDryerActive': 1 if prediction['appliances']['shiftable']['clothes_dryer']['active'] else 0,
                    'TVActive': 1 if prediction['appliances']['fixed']['tv']['active'] else 0,
                    'RefrigeratorActive': 1 if prediction['appliances']['fixed']['refrigerator']['active'] else 0,
                    'LightsActive': 1 if prediction['appliances']['fixed']['lights']['active'] else 0,
                    'VacuumActive': 1 if prediction['appliances']['fixed']['vacuum']['active'] else 0,
                    'HairDryerActive': 1 if prediction['appliances']['fixed']['hair_dryer']['active'] else 0
                }
                
                simulation_data.append(data_row)

                obs = next_state
                
                # Print progress
                if step % 10 == 0:
                    print(f"RL simulation progress: {step}/96 steps completed")
            
            except Exception as e:
                print(f"Error in step {step}: {str(e)}")
                # Continue with next step even if one fails
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(simulation_data)
        filename = f'rl_simulation_data.csv'
        csv_path = os.path.join('/tmp', filename)
        df.to_csv(csv_path, index=False)
        
        # Return the file as attachment
        return send_file(
            csv_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return {"error": str(e)}, 500

def format_state(data, time_step, day):
    """
    Converts simulation state to the format expected by the RL model
    """

    # Get values from request
    try:
        shift_prog = np.array(data['shift_prog'], dtype=np.int32)
        time_to_start_shift = np.array(data['time_to_start_shift'], dtype=np.int32) 
        home_temp = np.array(data['home_temp'], dtype=np.float32)
        water_temp = np.array(data['water_temp'], dtype=np.float32)
        soc_ess = np.array(data['soc_ess'], dtype=np.float32)
        soc_ev = np.array(data['soc_ev'], dtype=np.float32)
        time_to_start_noctrl = np.array(data['time_to_start_noctrl'], dtype=np.int32)
    except KeyError as e:
        # This will report the specific missing key
        raise KeyError(f"Missing required input state data: {e}")
        
    # Create dictionary observation like in the CustomEnv
    dict_obs = {
        'home_temp': np.array(home_temp),
        'water_temp': np.array(water_temp),
        'soc_ess': np.array(soc_ess),
        'soc_ev': np.array(soc_ev),
        'pv_power': np.array(PV_t_norm_96[(day-1) * 96 + time_step] if (day-1) * 96 + time_step < len(PV_t_norm_96) else 0),
        'grid_price': real_time_price_norm_96[(day-1) * 96 + time_step + 1 : day*96 + time_step + 1] if (day-1) * 96 + time_step < len(real_time_price_norm_96) else np.zeros(Paras['T_end']),
        'solar': solar_norm_96[(day-1) * 96 + time_step + 1 : day*96 + time_step + 1] if (day-1) * 96 + time_step < len(solar_norm_96) else np.zeros(Paras['T_end']),
        'temp_out': temp_out_norm_96[(day-1) * 96 + time_step + 1 : day*96 + time_step + 1] if (day-1) * 96 + time_step < len(temp_out_norm_96) else np.zeros(Paras['T_end']),
    }

    # Each appliance can have progress 0 to K_shift[i]
    shift_prog_encoded = []
    for i, prog in enumerate(shift_prog):
        # Create one-hot vector for this appliance's progress
        # Length is 1+K_shift[i] to account for progress from 0 to K_shift[i]
        one_hot = np.zeros(1 + K_shift[i], dtype=np.int32)
        # Ensure prog is within valid range
        valid_prog = min(max(0, prog), K_shift[i])
        one_hot[valid_prog] = 1
        shift_prog_encoded.extend(one_hot)

    time_to_start_shift_encoded = []
    for i, time_to_start in enumerate(time_to_start_shift):
        # Calculate the range size
        range_size = Paras['T_end']
        # Adjust the time_to_start value by the offset
        t1 = time_step - T_ini_shift[i]
        t2 = time_step - T_end_shift[i]
        adjusted_time = t1 if t1>=0 and t2<0 else 0
        # Create one-hot vector
        one_hot = np.zeros(range_size, dtype=np.int32)
        one_hot[adjusted_time] = 1
        time_to_start_shift_encoded.extend(one_hot)

    time_to_start_noctrl_encoded = []
    for i, time_to_start in enumerate(time_to_start_noctrl):
        # Calculate the range size
        range_size = Paras['T_end']
        # Adjust the time_to_start value by the offset
        t1 = time_step - T_ini_nocntl[i]
        t2 = time_step - T_end_nocntl[i]
        adjusted_time = t1 if t1>=0 and t2<0 else 0
        # Create one-hot vector
        one_hot = np.zeros(range_size, dtype=np.int32)
        one_hot[adjusted_time] = 1
        time_to_start_noctrl_encoded.extend(one_hot)

    # Add the encoded MultiDiscrete spaces to dict_obs
    dict_obs['shift_prog'] = np.array(shift_prog_encoded)
    dict_obs['time_to_start_shift'] = np.array(time_to_start_shift_encoded)
    dict_obs['time_to_start_noctrl'] = np.array(time_to_start_noctrl_encoded)
    
    # Flatten the observation using FlattenObservation wrapper
    obs_array = []

    # If we have the actual environment class, use its flattening logic
    for key in sorted(dict_obs.keys()):
        if isinstance(dict_obs[key], np.ndarray):
            obs_array.extend(dict_obs[key].flatten())
        else:
            obs_array.append(dict_obs[key])

    obs_array = np.array(obs_array)
    assert obs_array.shape == (1078,), f"Observation shape is {obs_array.shape}, expected (1078,)"
        
    return np.array(obs_array)

def format_response(next_state, action, time_step, day):
    """Format the environment's next_state into a structured response for the frontend"""
    
    # Extract values from next_state and action
    home_temp = float(next_state[96]) + float(Paras.get('HVAC_set', 22))
    water_temp = float(next_state[1077]) + float(Paras.get('EWH_set', 60))
    soc_ess = float(next_state[115])
    soc_ev = float(next_state[116])
    
    # Process continuous actions
    ess_power = action[0] * p_max_ess
    ess_power_var = (1 + action[1]) * pow(p_max_ess/2, 2) / 2  
    ev_power = (1 + action[2]) * p_max_ev  / 2 
    ev_power_var = (1 + action[3]) * pow(p_max_ev/2, 2) / 2
    ewh_power = (1 + action[4]) * p_max_ewh  / 2 
    ewh_power_var = (1 + action[5]) * pow(p_max_ewh/2, 2) / 2
    hvac_power = (1 + action[6]) * p_max_hvac / 2 
    hvac_power_var = (1 + action[7]) * pow(p_max_hvac/2, 2) / 2
    p_mu = np.array([hvac_power, ewh_power, ev_power, ess_power])
    p_var = np.array([hvac_power_var, ewh_power_var, ev_power_var, ess_power_var])
    p_ctrl = MND(p_mu, p_var)
    hvac_power = max(0, p_ctrl[0])
    ewh_power = max(0, p_ctrl[1])
    ev_power = max(0, p_ctrl[2])
    ess_power = p_ctrl[3]
    
    # Process shiftable appliance actions
    shift_actions = MBD((1 + action[-3:]) / 2)
    shift_prog = np.where(next_state[98:115])[0] - [0, 4, 11]
    
    # Current environment values
    current_price = real_time_price_96[day*96 + time_step + 32]
    current_solar = PV_t_96[day*96 + time_step + 32]
    current_temp_out = temp_out_96[day*96 + time_step + 32]
    
    # Build response with all visualization-relevant data
    response = {
        'timestamp': time_step,
        'day': day,
        'environment': {
            'price': float(current_price),
            'solar_production': float(current_solar),
            'outside_temp': float(current_temp_out)
        },
        'temperatures': {
            'home': {
                'current': float(home_temp),
                'setpoint': float(Paras.get('HVAC_set', 22))
            },
            'water': {
                'current': float(water_temp),
                'setpoint': float(Paras.get('EWH_set', 60))
            }
        },
        'battery': {
            'soc': float(soc_ess),
            'power': float(ess_power)
        },
        'ev': {
            'soc': float(soc_ev),
            'power': float(ev_power),
            'connected': time_step >= t_ini_ev and time_step <= t_end_ev
        },
        'appliances': {
            'controllable': {
                'hvac': {
                    'power': float(hvac_power),
                    'active': hvac_power > 0
                },
                'water_heater': {
                    'power': float(ewh_power),
                    'active': ewh_power > 0
                }
            },
            'shiftable': {
                'dishwasher': {
                    'active': bool(shift_actions[0]),
                    'power': float(Power_shift[0]) if shift_actions[0] else 0,
                    'progress': shift_prog[0], #int(shift_prog[0]),
                    'total_duration': int(K_shift[0])
                },
                'wash_machine': {
                    'active': bool(shift_actions[1]),
                    'power': float(Power_shift[1]) if shift_actions[1] else 0,
                    'progress': shift_prog[1], #int(shift_prog[1]),
                    'total_duration': int(K_shift[1])
                },
                'clothes_dryer': {
                    'active': bool(shift_actions[2]),
                    'power': float(Power_shift[2]) if shift_actions[2] else 0,
                    'progress': shift_prog[2], #int(shift_prog[2]),
                    'total_duration': int(K_shift[2])
                }
            },
            'fixed': {
                'tv': {
                    'active': time_step >= T_ini_nocntl[0] and time_step < T_end_nocntl[0],
                    'power': float(Power_nonctrl[0])
                },
                'refrigerator': {
                    'active': time_step >= T_ini_nocntl[1] and time_step < T_end_nocntl[1],
                    'power': float(Power_nonctrl[1])
                },
                'lights': {
                    'active': time_step >= T_ini_nocntl[2] and time_step < T_end_nocntl[2],
                    'power': float(Power_nonctrl[2])
                },
                'vacuum': {
                    'active': time_step >= T_ini_nocntl[3] and time_step < T_end_nocntl[3],
                    'power': float(Power_nonctrl[3])
                },
                'hair_dryer': {
                    'active': time_step >= T_ini_nocntl[4] and time_step < T_end_nocntl[4],
                    'power': float(Power_nonctrl[4])
                }
            }
        },
        'energy_flow': calculate_energy_flows(
            shift_actions, 
            Power_shift,
            [hvac_power, ewh_power, ev_power, ess_power],
            Power_nonctrl,
            time_step,
            current_solar
        )
    }
    
    return convert_numpy_types(response)

def convert_numpy_types(obj, round_power=True):
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    Also rounds power and percentage values to 2 decimal places if round_power is True
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Check if this is a power or percentage value that should be rounded
            should_round = round_power and isinstance(v, (float, np.floating)) and (
                # Common power-related keys
                k in ['power', 'powerFlow', 'soc', 'to_house', 'to_battery', 
                     'to_grid', 'from_grid', 'from_solar', 'net_power', 'total', 
                     'solar_production', 'price', 'powerLevel', 'solarOutput', 
                     'batteryLevel', 'gridDraw', 'houseDemand', 'shiftable', 
                     'controllable', 'fixed'] or
                # Keys containing these words
                any(word in k.lower() for word in ['power', 'level', 'output', 'demand', 
                                                  'percentage', 'current', 'setpoint', 
                                                  'production'])
            )
            
            if should_round:
                result[k] = round(float(v), 2)
            else:
                result[k] = convert_numpy_types(v, round_power)
        return result
    elif isinstance(obj, list):
        return [convert_numpy_types(item, round_power) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist(), round_power)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def calculate_energy_flows(shift_actions, shift_powers, ctrl_powers, nonctrl_powers, time_step, solar_power):
    """
    Calculate energy flows between components for visualization
    """
    try:
        # Calculate power consumption by category
        shift_consumption = sum([a * p for a, p in zip(shift_actions, shift_powers)])
        
        # HVAC, water heater, EV (exclude battery which can be negative)
        ctrl_consumption = sum(ctrl_powers[:3])
        
        # Battery power (can be negative)
        battery_power = ctrl_powers[3]
        
        # Fixed appliances
        nonctrl_consumption = sum([
            p if (time_step >= T_ini_nocntl[i] and time_step < T_end_nocntl[i]) else 0
            for i, p in enumerate(nonctrl_powers)
        ])
        
        # Total house demand
        total_demand = shift_consumption + ctrl_consumption + nonctrl_consumption
        
        # Calculate flows
        solar_to_house = min(solar_power, total_demand)
        solar_to_battery = max(0, min(solar_power - solar_to_house, -battery_power if battery_power < 0 else 0))
        solar_to_grid = max(0, solar_power - solar_to_house - solar_to_battery)
        
        battery_to_house = max(0, -battery_power) if battery_power < 0 else 0
        grid_to_house = max(0, total_demand - solar_to_house - battery_to_house)
        grid_to_battery = max(0, battery_power) if battery_power > 0 else 0
        
        return convert_numpy_types({
            'solar': {
                'to_house': float(solar_to_house),
                'to_battery': float(solar_to_battery),
                'to_grid': float(solar_to_grid),
                'total': float(solar_power)
            },
            'battery': {
                'to_house': float(battery_to_house),
                'from_grid': float(grid_to_battery),
                'from_solar': float(solar_to_battery),
                'net_power': float(battery_power)  # Positive = charging, Negative = discharging
            },
            'grid': {
                'to_house': float(grid_to_house),
                'to_battery': float(grid_to_battery),
                'from_solar': float(solar_to_grid),
                'net_power': float(grid_to_house + grid_to_battery - solar_to_grid)  # Positive = import, Negative = export
            },
            'house': {
                'demand': {
                    'shiftable': float(shift_consumption),
                    'controllable': float(ctrl_consumption),
                    'fixed': float(nonctrl_consumption),
                    'total': float(total_demand)
                }
            }
        }, round_power=True)
    except Exception as e:
        print(f"Error in calculate_energy_flows: {e}")

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)