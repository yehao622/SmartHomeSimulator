const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');

// Simple custom retry logic since axios-retry might not be available
const axiosWithRetry = async (config, retries = 3, delay = 1000) => {
  try {
    return await axios(config);
  } catch (error) {
    if (retries === 0) throw error;
    console.log(`Request failed. Retrying in ${delay}ms... (${retries} attempts left)`);
    await new Promise(resolve => setTimeout(resolve, delay));
    return axiosWithRetry(config, retries - 1, delay * 2);
  }
};

// Try to use axios-retry if available, otherwise use our custom implementation
let axiosRetry;
try {
  axiosRetry = require('axios-retry');
  // Configure axios-retry
  axiosRetry(axios, { 
    retries: 3,
    retryDelay: (retryCount) => retryCount * 1000
  });
  console.log('axios-retry configured successfully');
} catch (error) {
  console.warn('axios-retry not available, using custom retry implementation');
  // Add a wrapped version of get/post with retries
  const originalGet = axios.get;
  const originalPost = axios.post;
  
  axios.get = async (url, config = {}) => {
    return axiosWithRetry({ ...config, method: 'get', url });
  };
  
  axios.post = async (url, data, config = {}) => {
    return axiosWithRetry({ ...config, method: 'post', url, data });
  };
}

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
    credentials: true
  }
});

// Configuration for RL service
const RL_SERVICE_URL = process.env.RL_SERVICE_URL || 'http://rl-service:5000';

// Add a state object to track simulation variables needed by the RL model
let rlState = {
  day: 1,
  timeStep: 0,
  shift_prog: [0, 0, 0],
  time_to_start_shift: [0, 0, 0],
  home_temp: 4.0,  // Deviation from setpoint
  water_temp: -5.0,  // Deviation from setpoint
  soc_ev: 0.0,
  soc_ess: 0.0,
  time_to_start_noctrl: [0, 0, 0, 0, 0]
};

// Simulation state
let simulationRunning = false;
let simulationTime = new Date('2024-01-01T00:00:00');
let timeScale = 60; // 1 minute per second
let lastTimeUpdate = Date.now();
let simInterval;
let simulationMode = 'idle'; // 'none', 'manual', 'idle' or 'ai'
let rlPredictionInProgress = false; // Flag to prevent concurrent RL requests
let simulationStartTime = new Date();
let simulationElapsedMinutes = 0;
let simulationTimeLimit = null; // in seconds

// Device state
let devices = [];
let solarOutput = 0;
let batteryLevel = 35;
let batteryStatus = 'empty';
let batteryPower = 0;
let gridDraw = 0;
let houseDemand = 0;

let simulationMaxSteps = null;
let currentSimulationStep = 0;
let hourlyDataAvailable = false;
let lastKnownTimeScale = 60;
let energyModels = {
  indoorTemperature: 24.0,
  waterTemperature: 55.0,
  evSoC: 0.0,
  evConnected: false,
  hvacPower: 0,
  waterHeaterPower: 0,
  evPower: 0
};
let timeScaleJustChanged = false;
let pendingTimeScale = null;

// Fetch devices from device-service
async function fetchDevices() {
  try {
    console.log("Attempting to fetch devices from device service...");
    const response = await axios.get('http://device-service:8080/devices');
    devices = response.data;
    console.log('Fetched devices:', devices);
  } catch (error) {
    console.error('Error fetching devices:', error.message);
    devices = [
      { ID: 'solar1', Name: 'Solar Panel', Type: 'solar', PowerLevel: 0, Status: 'active' },
      { ID: 'batt1', Name: 'Battery', Type: 'battery', PowerLevel: 0, Status: 'idle' }
    ];
    console.log("Using default devices:", devices);
  }
}

// Update simulation state
async function updateSimulation() {
  if (!simulationRunning) return;

  // Check if we've reached the maximum steps from hourly data
  if (simulationMaxSteps && currentSimulationStep >= simulationMaxSteps) {
    console.log(`Reached maximum simulation steps (${simulationMaxSteps}). Stopping simulation.`);
    simulationRunning = false;
    clearInterval(simInterval);
    simInterval = null;

    // Notify clients
    io.emit('simulation_max_steps_reached', {
      message: "Reached the end of available hourly data.",
      currentStep: currentSimulationStep,
      maxSteps: simulationMaxSteps
    });

    // Also send status update
    io.emit('simulation_status', { 
      running: simulationRunning,
      mode: simulationMode
    });

    return;
  }

  // Calculate elapsed real time since last update
  const now = Date.now();
  let elapsedRealMs = now - lastTimeUpdate;

  // If this is the first update or after a speed change, use a minimum time delta
  if (lastTimeUpdate === 0 || timeScaleJustChanged) {
    elapsedRealMs = 500; // Force a minimum update on first cycle
    timeScaleJustChanged = false;
  } 

  // Always update the last time
  lastTimeUpdate = now;

  // Scale the elapsed time by the timeScale factor
  // This ensures time advances proportionally to the speed setting
  const elapsedSimulationSeconds = (elapsedRealMs / 1000) * timeScale;

  // For battery updates, ensure there's always a minimum visible change at any speed
  const minBatteryUpdateSeconds = 0.5; // 0.5 second minimum for battery updates
  const batteryUpdateSeconds = Math.max(elapsedSimulationSeconds, minBatteryUpdateSeconds);
  const batteryUpdateMinutes = batteryUpdateSeconds / 60;

  // Update simulation elapsed minutes
  simulationElapsedMinutes += elapsedSimulationSeconds / 60;

  // Update actual simulation time object based on elapsed minutes
  const totalMinutes = Math.floor(simulationElapsedMinutes);
  const hours = Math.floor(totalMinutes / 60) % 24; // Ensure day wrapping
  const minutes = totalMinutes % 60;
  const seconds = Math.floor((simulationElapsedMinutes * 60) % 60);

  simulationTime = new Date('2024-01-01T00:00:00');
  simulationTime.setHours(hours);
  simulationTime.setMinutes(minutes);
  simulationTime.setSeconds(seconds);

  // Increment step counter
  currentSimulationStep++;

  // Check if time limit has been reached
  if (simulationTimeLimit !== null && simulationElapsedMinutes >= simulationTimeLimit) {
    console.log(`Simulation time limit of ${simulationTimeLimit} minutes reached, stopping simulation`);
    simulationRunning = false;
    clearInterval(simInterval);
    simInterval = null;

    // Emit both status update and specific time limit reached event
    io.emit('simulation_status', { 
      running: simulationRunning, 
      mode: simulationMode,
      timeLimitReached: true 
    });

    io.emit('simulation_time_limit_reached', { 
      elapsedTime: simulationElapsedMinutes,
      timeLimit: simulationTimeLimit,
      simTime: simulationTime.toISOString()
    });
    
    return;
  }
  
  if (simulationMode === 'manual') {
    // Calculate solar output based on time of day
    const hour = simulationTime.getHours();

    if (hour >= 6 && hour <= 18) {
      // Bell curve for solar production with peak at noon
      const peak = 12;
      const factor = 1 - Math.abs(hour - peak) / 6;
      solarOutput = parseFloat((5 * factor * (0.8 + Math.random() * 0.4)).toFixed(1)); // Max 5kW
    } else {
      solarOutput = 0.0;
    }

    // Calculate total house demand from active devices
    houseDemand = devices.reduce((total, device) => {
      if (device.Type !== 'solar' && device.Type !== 'battery' && device.Type !== 'grid') {
        return total + (device.Status === 'active' ? device.PowerLevel : 0);
      }
      return total;
    }, 0);
    houseDemand = parseFloat(houseDemand.toFixed(1));

    // Define minimum battery level and other constants
    const BATTERY_MIN_LEVEL = 10.01; 
    const MAX_BATTERY_CHARGE_RATE = 2.4; // kW
    const MAX_BATTERY_DISCHARGE_RATE = 2.4; // kW

    // Calculate solar to house directly
    const solarToHouse = Math.min(solarOutput, houseDemand);
    const remainingDemand = Math.max(0, houseDemand - solarToHouse);
    const excessSolar = Math.max(0, solarOutput - solarToHouse);
    gridDraw = remainingDemand;

    // First handle battery status change when at minimum level
    if (batteryLevel <= BATTERY_MIN_LEVEL && (batteryStatus === 'discharging' || batteryPower < 0)) {
      batteryStatus = 'empty';
      batteryPower = 0;
      console.log(`Battery reached minimum level (${batteryLevel}%). Switching to empty status.`);
    }

    // This is a single decision point for the battery's state
    let batteryAction = 'none'; // 'none', 'charge', or 'discharge'
    let gridToSolar = 0; // Track power flowing directly from solar to grid

    // Check if we're in a preferred charging time window (12pm-3pm or 12am-5am)
    const isPreferredChargingTime = (hour >= 12 && hour <= 15) || (hour >= 0 && hour <= 5);

    // 4. Determine battery behavior based on conditions
    if (isPreferredChargingTime) {
      // First try to charge from solar if available
      if (excessSolar > 0 && batteryLevel < 100) {
        batteryAction = 'charge';
        batteryStatus = 'charging';
        batteryPower = Math.min(excessSolar, MAX_BATTERY_CHARGE_RATE);
        // Update battery level (limited to 100%)
        const batteryIncrease = Math.max(0.05, batteryPower * 0.95 * batteryUpdateMinutes); // 90% efficiency factor
        batteryLevel = Math.min(100, batteryLevel + batteryIncrease);
        //console.log(`Battery charging: +${batteryIncrease.toFixed(2)}%, power=${batteryPower}, minutes=${batteryUpdateMinutes.toFixed(3)}`);
        
        // If there's still excess solar after charging battery, send it to grid
        const remainingSolar = excessSolar - batteryPower;
        if (remainingSolar > 0) {
          gridToSolar = remainingSolar;
          // Negative grid draw indicates exporting to grid
          gridDraw -= gridToSolar;
        } else {
          gridDraw = 0;
        }
      } 
      // If no excess solar but still preferred charging time, charge from grid during night hours
      else if (hour >= 0 && hour <= 5 && batteryLevel < 100) {
        const trickleCharge = batteryLevel < 20 ? MAX_BATTERY_CHARGE_RATE : 1.0; // Higher charge rate if very low
        batteryAction = 'charge';
        batteryStatus = 'charging';
        batteryPower = trickleCharge;
        const batteryIncrease = Math.max(0.05, batteryPower * 0.95 * batteryUpdateMinutes);
        batteryLevel = Math.min(100, batteryLevel + batteryIncrease);
        gridDraw = remainingDemand + batteryPower; // Grid powers both house and battery
      }
      // If battery is full and we have excess solar, send it directly to grid
      else if (batteryLevel >= 99 && excessSolar > 0) {
        batteryAction = 'none';
        batteryStatus = 'empty';
        batteryPower = 0;
        gridToSolar = excessSolar;
        gridDraw = remainingDemand - gridToSolar; // Negative indicates exporting to grid
      }
      // Otherwise, use grid for remaining demand
      else {
        batteryAction = 'none';
        batteryStatus = 'empty';
        batteryPower = 0;
        gridDraw = remainingDemand;
      }
    } else {
      // PEAK ELECTRICITY PRICE TIME - Battery discharge has priority
      
      // Check if we have remaining demand and battery can discharge
      if (remainingDemand > 0 && batteryLevel > BATTERY_MIN_LEVEL) {
        batteryAction = 'discharge';
        batteryStatus = 'discharging';
        // Calculate max possible discharge based on battery level
        const availableCapacity = batteryLevel - BATTERY_MIN_LEVEL;
        const safeDischargeRate = Math.min(MAX_BATTERY_DISCHARGE_RATE, availableCapacity * 0.15);
        // Calculate actual discharge (limited by remaining demand)
        batteryPower = -Math.min(remainingDemand, safeDischargeRate);
        // Update battery level - ensure it doesn't go below minimum
        const batteryDecrease = Math.min(-0.05, batteryPower * batteryUpdateMinutes);
        batteryLevel = Math.max(BATTERY_MIN_LEVEL, batteryLevel + batteryDecrease);
        //console.log(`Battery discharging: ${dischargeAmount.toFixed(2)}%, power=${batteryPower}, minutes=${batteryUpdateMinutes.toFixed(3)}`);
        // Remaining demand after battery discharge goes to grid
        gridDraw = Math.max(0, remainingDemand + batteryPower); // batteryPower is negative
      }
      // If excess solar available, charge battery (lower priority than discharging during peak)
      else if (excessSolar > 0 && batteryLevel < 100 && batteryAction === 'none') {
        batteryAction = 'charge';
        batteryStatus = 'charging';
        batteryPower = Math.min(excessSolar, MAX_BATTERY_CHARGE_RATE);
        // Update battery level (limited to 100%)
        const batteryIncrease = Math.max(0.05, batteryPower * 0.95 * batteryUpdateMinutes); // 90% efficiency factor
        batteryLevel = Math.min(100, batteryLevel + batteryIncrease);
        // If there's still excess solar after charging battery, send it to grid
        const remainingSolar = excessSolar - batteryPower;
        if (remainingSolar > 0) {
          gridToSolar = remainingSolar;
          // Negative grid draw indicates exporting to grid
          gridDraw = remainingDemand - gridToSolar;
        } else {
          gridDraw = remainingDemand; // Grid handles any remaining demand
        }
      }
      // In all other cases, grid handles remaining demand
      else {
        batteryAction = 'none';
        batteryStatus = 'empty';
        batteryPower = 0;
        gridDraw = remainingDemand;
      }
    }

    // Format for consistency
    batteryLevel = parseFloat(batteryLevel.toFixed(1));
    batteryPower = parseFloat(batteryPower.toFixed(1));
    gridDraw = parseFloat(gridDraw.toFixed(1));

    // Add a flag for the direct solar to grid connection
    const solarToGrid = gridToSolar > 0;

    // This addresses the issue where grid wasn't fully covering the demand
    if (batteryStatus === 'empty' && solarOutput + gridDraw < houseDemand && gridDraw >= 0) {
      const gridShortfall = houseDemand - solarOutput - gridDraw;
      if (gridShortfall > 0.01) { // Only fix if the shortfall is significant
        //console.log(`Correcting grid draw: ${gridDraw} -> ${gridDraw + gridShortfall} to match demand`);
        gridDraw = parseFloat((gridDraw + gridShortfall).toFixed(1));
      }
    }

    io.emit('simulation_update', {
      time: simulationTime.toISOString(),
      devices,
      solarOutput,
      batteryLevel,
      batteryStatus,
      batteryPower,
      gridDraw,
      houseDemand,
      simulationMode,
      elapsedMinutes: simulationElapsedMinutes,
      solarToGrid: solarToGrid,
      formattedTime: formatSimulationTime(),
      currentStep: currentSimulationStep,
      energyModels: energyModels,
      currentTimeScale: timeScale, // Include current timeScale
      simulationDay: Math.floor(simulationElapsedMinutes / (24 * 60)) + 1 
    });
  } else if (simulationMode === 'ai' && !rlPredictionInProgress) {
    try {
      rlPredictionInProgress = true; // Set flag to prevent concurrent requests
      
      // Update timeStep for day tracking (96 steps per day)
      rlState.timeStep = (rlState.timeStep + 1) % 96;
      if (rlState.timeStep === 0) {
        rlState.day = (rlState.day % 60) + 1;
      }

      // Call RL microservice for prediction
      const rlResponse = await axios.post(`${RL_SERVICE_URL}/predict`, rlState);
      const rlPrediction = rlResponse.data;
      
      // Apply RL model's recommendations to simulation state
      applyRlPrediction(rlPrediction);
      
      // Broadcast RL insights to connected clients
      io.emit('rl_prediction', rlPrediction);
      
      // Update RL state for next step based on current simulation state
      updateRlState(rlPrediction);
      
      rlPredictionInProgress = false; // Reset flag
    } catch (error) {
      console.error('Error getting RL prediction:', error.message);
      rlPredictionInProgress = false; // Reset flag even on error
    }
  }
}

// Then add this function to format the time:
function formatSimulationTime() {
  // Calculate hours and minutes directly from simulationTime object
  // This ensures correct time is displayed regardless of time scale
  const hours = simulationTime.getHours();
  const minutes = simulationTime.getMinutes();
  
  // Format as HH:MM
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
}

function applyRlPrediction(prediction) {
  if (!prediction || prediction.error) return;
  
  // Apply RL model outputs to simulation state
  try {
    // Update battery level
    batteryLevel = prediction.battery.soc * 100; // Convert to percentage
    
    // Determine battery status
    if (prediction.battery.power > 0) {
      batteryStatus = 'charging';
      batteryPower = prediction.battery.power;
    } else if (prediction.battery.power < 0) {
      batteryStatus = 'discharging';
      batteryPower = prediction.battery.power;
    } else {
      batteryStatus = 'empty';
      batteryPower = 0;
    }
    
    // Update solar output
    solarOutput = prediction.environment.solar_production;

    // Update grid draw
    gridDraw = prediction.energy_flow.grid.net_power;
    
    // Update house demand
    houseDemand = prediction.energy_flow.house.demand.total;
    
    // Update devices state - combine all appliances
    devices = [
      // Controllable appliances
      {
        ID: 'hvac',
        Name: 'HVAC',
        Type: 'hvac',
        PowerLevel: prediction.appliances.controllable.hvac.power,
        Status: prediction.appliances.controllable.hvac.active ? 'active' : 'inactive'
      },
      {
        ID: 'water_heater',
        Name: 'Water Heater',
        Type: 'water_heater',
        PowerLevel: prediction.appliances.controllable.water_heater.power,
        Status: prediction.appliances.controllable.water_heater.active ? 'active' : 'inactive'
      },
      {
        ID: 'ev_charger',
        Name: 'EV Charger',
        Type: 'ev_charger',
        PowerLevel: prediction.ev.power,
        Status: prediction.ev.connected ? (prediction.ev.power > 0 ? 'charging' : 'connected') : 'disconnected'
      },
      // Shiftable appliances
      {
        ID: 'dishwasher',
        Name: 'Dishwasher',
        Type: 'dishwasher',
        PowerLevel: prediction.appliances.shiftable.dishwasher.power,
        Status: prediction.appliances.shiftable.dishwasher.active ? 'active' : 'inactive'
      },
      {
        ID: 'wash_machine',
        Name: 'Wash Machine',
        Type: 'wash_machine',
        PowerLevel: prediction.appliances.shiftable.wash_machine.power,
        Status: prediction.appliances.shiftable.wash_machine.active ? 'active' : 'inactive'
      },
      {
        ID: 'clothes_dryer',
        Name: 'Clothes Dryer',
        Type: 'clothes_dryer',
        PowerLevel: prediction.appliances.shiftable.clothes_dryer.power,
        Status: prediction.appliances.shiftable.clothes_dryer.active ? 'active' : 'inactive'
      },
      // Fixed appliances
      {
        ID: 'tv',
        Name: 'TV',
        Type: 'tv',
        PowerLevel: prediction.appliances.fixed.tv.power,
        Status: prediction.appliances.fixed.tv.active ? 'active' : 'inactive'
      },
      {
        ID: 'refrigerator',
        Name: 'Refrigerator',
        Type: 'refrigerator',
        PowerLevel: prediction.appliances.fixed.refrigerator.power,
        Status: prediction.appliances.fixed.refrigerator.active ? 'active' : 'inactive'
      },
      {
        ID: 'lights',
        Name: 'Lights',
        Type: 'lights',
        PowerLevel: prediction.appliances.fixed.lights.power,
        Status: prediction.appliances.fixed.lights.active ? 'active' : 'inactive'
      },
      {
        ID: 'vacuum',
        Name: 'Vacuum Cleaner',
        Type: 'vacuum',
        PowerLevel: prediction.appliances.fixed.vacuum.power,
        Status: prediction.appliances.fixed.vacuum.active ? 'active' : 'inactive'
      },
      {
        ID: 'hair_dryer',
        Name: 'Hair Dryer',
        Type: 'hair_dryer',
        PowerLevel: prediction.appliances.fixed.hair_dryer.power,
        Status: prediction.appliances.fixed.hair_dryer.active ? 'active' : 'inactive'
      }
    ];
  } catch (error) {
    console.error('Error applying RL prediction:', error);
  }
}

function updateRlState(prediction) {
  // Update RL state based on current prediction
  try {
    rlState = {
      day: prediction.day,
      timeStep: prediction.timestamp,
      shift_prog: [
        prediction.appliances.shiftable.dishwasher.progress,
        prediction.appliances.shiftable.wash_machine.progress,
        prediction.appliances.shiftable.clothes_dryer.progress
      ],
      time_to_start_shift: [
        // This might need adjustment based on your specific schedule logic
        prediction.appliances.shiftable.dishwasher.active ? 1 : 0,
        prediction.appliances.shiftable.wash_machine.active ? 1 : 0,
        prediction.appliances.shiftable.clothes_dryer.active ? 1 : 0
      ],
      home_temp: prediction.temperatures.home.current - prediction.temperatures.home.setpoint,
      water_temp: prediction.temperatures.water.current - prediction.temperatures.water.setpoint,
      soc_ev: prediction.ev.soc,
      soc_ess: prediction.battery.soc,
      time_to_start_noctrl: [
        prediction.appliances.fixed.tv.active ? 1 : 0,
        prediction.appliances.fixed.refrigerator.active ? 1 : 0,
        prediction.appliances.fixed.lights.active ? 1 : 0,
        prediction.appliances.fixed.vacuum.active ? 1 : 0,
        prediction.appliances.fixed.hair_dryer.active ? 1 : 0
      ]
    };

    return rlState
  } catch (error) {
    console.error('Error updating RL state:', error);
  }
}

// API routes
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'simulation-service' });
});

// Socket.io connection
io.on('connection', (socket) => {
  console.log('Client connected');

  // Add error handling to updateSimulation
  const safeUpdateSimulation = () => {
    try {
      // Wrap original function
      updateSimulation();
    } catch (error) {
      console.error('Error in simulation update:', error);
      
      // Attempt to recover
      io.emit('simulation_error', { 
        message: 'Simulation error occurred, attempting to recover',
        time: new Date().toISOString()
      });
      
      // Calculate total house demand from active devices
      houseDemand = devices.reduce((total, device) => {
        if (device.Type !== 'solar' && device.Type !== 'battery' && device.Type !== 'grid') {
          return total + (device.Status === 'active' ? device.PowerLevel : 0);
        }
        return total;
      }, 0);
      houseDemand = parseFloat(houseDemand.toFixed(1));
    }
  };
  
  // Send initial state
  socket.emit('simulation_update', {
    time: simulationTime.toISOString(),
    devices,
    solarOutput,
    batteryLevel,
    batteryStatus,
    batteryPower,
    gridDraw,
    houseDemand,
    simulationMode
  });

  // Add handler for state refresh
  socket.on('get_current_state', () => {
    // Send current simulation state
    socket.emit('simulation_update', {
      time: simulationTime.toISOString(),
      devices,
      solarOutput,
      batteryLevel,
      batteryStatus,
      batteryPower,
      gridDraw,
      houseDemand,
      simulationMode,
      elapsedMinutes: simulationElapsedMinutes,
      formattedTime: formatSimulationTime()
    });
  });
  
  // Handle simulation control
  socket.on('start_simulation', (data) => {
    // Stop any existing simulation
    if (simulationRunning) {
      clearInterval(simInterval);
    }

    // Reset simulation state
    simulationMode = data.initialState?.simulationMode || 'manual';

    // Reset the last time update to now
    lastTimeUpdate = 0;//Date.now();
    timeScaleJustChanged = true;

    // Handle preserved time state
    if (data.initialState && data.initialState.preserveTime === true) {
      // If preserveTime is true, use the provided elapsed minutes
      if (typeof data.initialState.elapsedMinutes === 'number') {
        simulationElapsedMinutes = data.initialState.elapsedMinutes;
        
        // Reconstruct the simulation time based on elapsed minutes
        const totalMinutes = Math.floor(simulationElapsedMinutes);
        const hours = Math.floor(totalMinutes / 60) % 24;
        const minutes = totalMinutes % 60;
        
        simulationTime = new Date('2024-01-01T00:00:00');
        simulationTime.setHours(hours);
        simulationTime.setMinutes(minutes);
        
        //console.log(`Preserved simulation time: ${formatSimulationTime()}, elapsed minutes: ${simulationElapsedMinutes}`);
      }
    } else {
      // Reset simulation time tracking only if not preserving
      simulationElapsedMinutes = 0;
      simulationTime = new Date('2024-01-01T00:00:00');
      // timeScale = data.timeScale || 60;
    }

    // Set other parameters from data
    simulationTimeLimit = data.timeLimit ? data.timeLimit / 60 : null;
    if (data.initialState?.timeScale) {
      timeScale = data.initialState.timeScale;
    }

    if (data.initialState && data.initialState.devices) {
      devices = data.initialState.devices.map(dev => ({
        ID: dev.id || dev.ID,
        Name: dev.name || dev.Name,
        Type: dev.type || dev.Type,
        PowerLevel: parseFloat(dev.power || dev.PowerLevel || 0),
        Status: dev.active === true || dev.Status === 'active' ? 'active' : 'inactive'
      }));

      batteryLevel = data.initialState.batteryLevel || batteryLevel;
    }

    // Check for max steps from hourly data
    if (data.maxSteps) {
      simulationMaxSteps = data.maxSteps;
      //console.log(`Setting maximum simulation steps to ${simulationMaxSteps}`);
    }

    // Start with a clean slate - no previous update loop
    simulationRunning = true;
    const interval = data.interval || 1000;

    console.log(`Starting simulation in ${simulationMode} mode with interval ${interval}ms` + 
              (simulationTimeLimit ? `, time limit: ${simulationTimeLimit} seconds` : '') +
              (simulationMaxSteps ? `, max steps: ${simulationMaxSteps}` : ''));
  
    // Clear any existing interval before setting a new one
    if (simInterval) {
      clearInterval(simInterval);
    }

    simInterval = setInterval(safeUpdateSimulation, interval);

    // Send initial state immediately
    io.emit('simulation_status', { 
      running: simulationRunning,
      mode: simulationMode,
      interval: interval,
      timeScale: timeScale,
      elapsedMinutes: simulationElapsedMinutes,
      formattedTime: formatSimulationTime()
    });
  });
  
  socket.on('stop_simulation', () => {
    if (simulationRunning) {
      // Remember current timeScale before stopping
      lastKnownTimeScale = timeScale;

      simulationRunning = false;
      clearInterval(simInterval);
      simInterval = null;

      // Notify all clients
      io.emit('simulation_status', { 
        running: simulationRunning,
        mode: simulationMode,
        timeScale: timeScale
      });
    }
  });
  
  socket.on('reset_simulation', () => {
    // Clear any existing interval
    if (simInterval) {
      clearInterval(simInterval);
      simInterval = null;
    }

    // Reset all simulation variables
    simulationTime = new Date('2024-01-01T00:00:00');
    simulationRunning = false;
    simulationMode = 'idle';
    simulationElapsedMinutes = 0;
    simulationTimeLimit = null;

    solarOutput = 0;
    batteryLevel = 35;
    batteryStatus = 'empty';
    batteryPower = 0;
    gridDraw = 0;
    houseDemand = 0;
    
    // Reset devices to inactive state
    if (devices && devices.length > 0) {
      devices = devices.map(device => ({
        ...device,
        Status: device.Type === 'refrigerator' ? 'active' : 'inactive',
        PowerLevel: device.Type === 'refrigerator' ? device.PowerLevel : 0
      }));
    }

    // Reset RL state
    rlState = {
      day: 1,
      timeStep: 0,
      shift_prog: [0, 0, 0],
      time_to_start_shift: [0, 0, 0],
      home_temp: 4.0,
      water_temp: -5.0,
      soc_ev: 0.0,
      soc_ess: 0.0,
      time_to_start_noctrl: [0, 0, 0, 0, 0]
    };

    // Stop any AI prediction in progress
    rlPredictionInProgress = false;

    // Notify clients
    io.emit('simulation_reset', {
      time: simulationTime.toISOString(),
      devices,
      solarOutput,
      batteryLevel,
      batteryStatus,
      batteryPower: 0,
      gridDraw,
      houseDemand,
      simulationMode: 'idle'
    });

    // Reset energy models
    energyModels = {
      indoorTemperature: 24.0,
      waterTemperature: 55.0,
      evSoC: 0.0,
      evConnected: false,
      hvacPower: 0,
      waterHeaterPower: 0,
      evPower: 0
    };

    // Also emit simulation status update
    io.emit('simulation_status', { 
      running: simulationRunning,
      mode: 'idle'
    });
  });

  socket.on('resume_simulation', (data) => {
    if (!simulationRunning) {
      simulationRunning = true;

      // Apply any pending time scale changes that happened during pause
      if (pendingTimeScale !== null) {
        timeScale = pendingTimeScale;
        pendingTimeScale = null; // Reset after applying
        timeScaleJustChanged = true; // Flag the change for the update function
        //console.log(`Applied pending timeScale change: ${timeScale}`);
      }

      // Reset time tracking for smooth resume
      lastTimeUpdate = 0;
      timeScaleJustChanged = true;

      // Preserve timeScale from previous state
      if (data.initialState?.timeScale) {
        timeScale = data.initialState.timeScale;
        //console.log(`Resuming with timeScale: ${timeScale}`);
      }
      
      simInterval = setInterval(safeUpdateSimulation, data.interval || 1000);

      // Important: Preserve the current elapsed minutes rather than resetting
      const preservedElapsedMinutes = simulationElapsedMinutes;
      
      // Notify clients about resumed state
      io.emit('simulation_status', { 
        running: simulationRunning,
        mode: simulationMode,
        preservedTime: true,
        elapsedMinutes: preservedElapsedMinutes,
        timeScale: timeScale
      });
    }
  });
  
  socket.on('set_time_scale', (data) => {
    const oldTimeScale = timeScale;
    timeScale = data.timeScale;

    // Store the new timeScale for when simulation resumes
    pendingTimeScale = timeScale;

    // Flag that time scale just changed - for next update cycle
    timeScaleJustChanged = true;

    // Reset the last time update to prevent time jumps when changing speed
    lastTimeUpdate = 0;//Date.now();

    // For active simulations, we need to ensure the next update is significant
    if (simulationRunning && simInterval) {
      // Cancel and restart the interval to apply the new timing immediately
      clearInterval(simInterval);

      // Immediate update to apply new speed
      updateSimulation();

      simInterval = setInterval(safeUpdateSimulation, data.interval || 1000);
    }

    // Notify clients about the speed change
    io.emit('time_scale_updated', { 
      oldTimeScale: oldTimeScale,
      newTimeScale: timeScale
    });
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });

  socket.on('set_simulation_mode', (data) => {
    // First stop any existing simulation
    if (simulationRunning && simInterval) {
      clearInterval(simInterval);
    }
    
    // Update mode and notify about the change
    simulationMode = data.mode;
    //console.log(`Simulation mode set to: ${simulationMode}`);
    
    // If previously running, restart with the new mode
    if (simulationRunning) {
      simInterval = setInterval(safeUpdateSimulation, data.interval || 1000);
    }
    
    // Reset RL prediction flag when changing modes
    rlPredictionInProgress = false;
    
    // Broadcast mode change to all clients
    io.emit('simulation_mode_changed', { mode: simulationMode });
  });

  socket.on('update_devices', (data) => {
    if (data.devices) {
      try {
        // Update devices array with new states
        devices = data.devices.map(device => ({
          ID: device.id || device.ID,
          Name: device.name || device.Name,
          Type: device.type || device.Type,
          PowerLevel: parseFloat(device.power || device.PowerLevel) || 0,
          Status: device.active || device.Status === 'active' ? 'active' : 'inactive'
        }));

        // Calculate house demand
        houseDemand = parseFloat(
          devices
            .filter(device => device.Status === 'active' && device.Type !== 'solar' && device.Type !== 'battery' && device.Type !== 'grid')
            .reduce((total, device) => total + device.PowerLevel, 0)
            .toFixed(1)
        );

        // Acknowledge the update
        socket.emit('devices_updated', {
          houseDemand: houseDemand,
          success: true
        });
      } catch (error) {
        console.error('Error updating devices:', error);
        socket.emit('devices_updated', { 
          success: false, 
          error: error.message 
        });
      }
    }
  });

  // Check for hourly data
  socket.on('hourly_data_available', (data) => {
    //console.log('Received hourly data info from client:', data);

    hourlyDataAvailable = true;
    
    // Store hourly data info for use in simulation
    if (data && data.maxSteps) {
      simulationMaxSteps = data.maxSteps;
      //console.log(`Set maximum simulation steps to ${simulationMaxSteps}`);
    }
  });

  // Handle RL prediction requests for specific time steps
  socket.on('request_rl_prediction', async (data) => {
    try {
      // Validate required parameters
      if (data.timeStep === undefined || data.day === undefined) {
        socket.emit('rl_error', { 
          message: 'Missing required parameters',
          error: `Required parameters: timeStep=${data.timeStep}, day=${data.day}`
        });
        return;
      }
    
      console.log(`Processing RL prediction request for day=${data.day}, timeStep=${data.timeStep}`);
    
      // Call the RL service
      const response = await axios.post(`${RL_SERVICE_URL}/predict`, data);
    
      // Emit the prediction back to the client
      socket.emit('rl_prediction', response.data);
    } catch (error) {
      console.error('Error getting RL prediction:', error.message);
    
      // Send detailed error back to client
      socket.emit('rl_error', {
        message: 'Failed to get AI prediction',
        error: error.message,
        data: data  // Return the original request data for debugging
      });
    }
  });

  // Add a socket event handler for time limit reached notifications
  socket.on('simulation_time_limit_reached', (data) => {
    console.log(`Simulation time limit of ${data.timeLimit} seconds reached.`);
    socket.emit('simulation_time_limit_reached_ack', { 
      received: true 
    });
  });

  // Add this socket handler for max steps reached acknowledgment
  socket.on('simulation_max_steps_reached_ack', (data) => {
    console.log('Client acknowledged max steps reached:', data);
  });

  // Add this socket handler to receive energy model updates
  socket.on('energy_models_update', (data) => {
    console.log('Received energy models update:', data);
    
    // Update server-side state
    energyModels = {
      ...energyModels,
      ...data
    };
    
    // Acknowledge the update
    socket.emit('energy_models_update_ack', { 
      received: true,
      timestamp: new Date().toISOString()
    });
  });

  // Add this socket handler to receive hourly data updates
  socket.on('hourly_data_update', (data) => {
    console.log('Received hourly data update:', data);
    
    if (simulationRunning && simulationMode === 'manual') {
      // Update simulation values from hourly data
      if (data.solarOutput !== undefined) {
        solarOutput = data.solarOutput;
      }
      
      // Update energy models if provided
      if (data.energyModels) {
        energyModels = {
          ...energyModels,
          ...data.energyModels
        };
      }
      
      // Log the update
      console.log(`Updated from hourly data: solar=${solarOutput}, temp=${data.temperature}, price=${data.price}`);
    }
  });

  socket.on('update_battery_state', (data) => {
    if (data && typeof data.batteryStatus !== 'undefined') {
      const oldStatus = batteryStatus;
      batteryStatus = data.batteryStatus;
      batteryPower = data.batteryPower || 0;
    
      // Log the battery update
      console.log(`Battery state updated: status=${batteryStatus}, power=${batteryPower}kW, level=${data.batteryLevel}%`);
    
      // If useGridForRemaining flag is set, update grid draw
      if (data.useGridForRemaining) {
        // Calculate remaining house demand after solar
        const solarToHouse = Math.min(solarOutput, houseDemand);
        const remainingDemand = Math.max(0, houseDemand - solarToHouse);
      
        // Use grid for all remaining demand
        gridDraw = remainingDemand;
        console.log(`Grid draw updated to ${gridDraw.toFixed(1)}kW to cover remaining demand`);
      }
    
      // Emit the update to all clients
      io.emit('simulation_update', {
        time: simulationTime.toISOString(),
        devices,
        solarOutput,
        batteryLevel,
        batteryStatus,
        batteryPower,
        gridDraw,
        houseDemand,
        simulationMode,
        elapsedMinutes: simulationElapsedMinutes
      });
    }
  });
});

// Fetch devices on startup
fetchDevices().then(() => {
  // Start server
  const PORT = process.env.PORT || 3000;
  server.listen(PORT, () => {
    console.log(`Simulation service running on port ${PORT}`);
  });
});