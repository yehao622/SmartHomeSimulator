<template>
  <div class="container">
    <header>
      <h1>Smart Home Energy Simulator</h1>
    </header>

    <div class="main-layout">
      <!-- Energy Flow Diagram -->
      <div class="diagram-section">
        <!-- Energy flow diagram -->
        <energy-flow-diagram
           :solar-output="solarOutput"
           :battery-level="batteryLevel"
           :battery-status="batteryStatus"
           :battery-power="batteryPower"
           :grid-power="gridPower"
           :house-demand="houseDemand"
           :appliances="appliances"
           :is-running="simulationRunning"
           :simulation-state="simulationState"
           :rl-prediction="rlPrediction"
           :simulation-elapsed-minutes="simulationElapsedMinutes"
           :simulation-formatted-time="simulationFormattedTime"
           :simulation-day="simulationDay"
           :home-temperature-history="homeTemperatureHistory"
           :water-temperature-history="waterTemperatureHistory"
           :appliance-power-history="appliancePowerHistory"
           :simulation-completed="simulationCompleted"
           @start-simulation="startSimulation"
           @pause-simulation="pauseSimulation"
           @resume-simulation="resumeSimulation"
           @reset-simulation="resetSimulation"
           @toggle-appliance="toggleAppliance"
           @demand-updated="updateHouseDemand"
           @grid-power-updated="updateGridPower"
           @config-updated="handleConfigUpdated"
           @speed-changed="handleSpeedChange"
           @battery-critical="handleBatteryCritical"
           @history-updated="onHistoryUpdated"
           @hourly-data-update="handleHourlyDataUpdate"
           @simulation-complete="handleSimulationComplete"
           @energy-models-updated="handleEnergyModelsUpdate"
           @solar-output-updated="solarOutput = $event"
           @battery-level-updated="batteryLevel = $event"
           @battery-status-updated="batteryStatus = $event"
           @update-appliance-power="handleAppliancePowerUpdate"
        />
      </div>
    </div>

    <!-- AI Optimization controls -->
    <div class="rl-simulation-controls" v-if="showRlControls">
       <h3>AI Optimization Simulation</h3>
  
      <div class="rl-simulation-info">
        <div class="rl-step-display">
          <div class="rl-step-label">Day:</div>
          <div class="rl-step-value">{{ currentDay }}</div>
        </div>
    
        <div class="rl-step-display">
          <div class="rl-step-label">Time:</div>
          <div class="rl-step-value">{{ formattedTimeStep }}</div>
        </div>
    
        <div class="rl-step-display">
          <div class="rl-step-label">Step:</div>
          <div class="rl-step-value">{{ formattedStepDisplay }}</div>
        </div>
      </div>
  
      <div class="rl-step-buttons">
        <button @click="previousStep" class="rl-btn rl-prev-btn" :disabled="!canGoBack || autoPlayActive">
          Previous
        </button>
    
        <button @click="nextStep" class="rl-btn rl-next-btn" :disabled="autoPlayActive">
          Next
        </button>
    
        <button @click="toggleAutoplay" class="rl-btn" :class="{'rl-pause-btn': autoPlayActive, 'rl-play-btn': !autoPlayActive}">
          {{ autoPlayActive ? 'Pause' : 'Auto Play' }}
        </button>
      </div>
    </div>

  </div>
</template>

<script>
import EnergyFlowDiagram from './components/EnergyFlowDiagram.vue';
import ChartManager from './components/chartConfig.js';

export default {
  components: {
    EnergyFlowDiagram
  },

  data() {
    return {
      currentSimulationStep: 0,

      showRlControls: true, // Default to showing controls

      // Simulation state
      simulationRunning: false,
      simulationState: 'idle', // 'idle', 'manual', or 'ai'
      simulationTimer: null,
      simulationInterval: 500, // Default to medium speed
      simulationElapsedTime: 0, // Elapsed time in seconds
      
      // Energy data
      solarOutput: 0,
      batteryLevel: 35,
      batteryStatus: 'empty', // 'charging', 'discharging', or 'empty'
      batteryPower: 0,
      gridPower: 0,
      houseDemand: 0,

      // Simulation clock
      simulationElapsedMinutes: 0,
      simulationFormattedTime: "00:00",
      simulationDay: 1,

      // plot data for manual simulation
      hourlyData: {
        temperature: 0,
        solarOutput: 0,
        price: 0,
        hour: "00:00"
      },
      simulationSteps: 0,
      maxSimulationSteps: null,
      simulationCompleted: false,

      // Add these properties for energy models
      energyModelState: {
        indoorTemperature: 24.0,
        waterTemperature: 55.0,
        evSoC: 0.0,
        evConnected: false,
        hvacPower: 0,
        waterHeaterPower: 0,
        evPower: 0
      },
      
      // Devices/appliances - will be loaded from configuration
      appliances: [],
      
      // Initial values to reset to
      initialState: {
        solarOutput: 0,
        batteryLevel: 35,
        batteryStatus: 'empty',
        batteryPower: 0,
        gridPower: 0,
        houseDemand: 0
      },

      // AI/RL related data
      rlPrediction: null,
      currentDay: 1,
      currentTimeStep: 0,
      autoPlayActive: false,
      autoPlayTimer: null,
      
      // History data for charts
      homeTemperatureHistory: new Array(96).fill(null),
      waterTemperatureHistory: new Array(96).fill(null),
      appliancePowerHistory: {
        dishwasher: new Array(96).fill(0),
        wash_machine: new Array(96).fill(0),
        clothes_dryer: new Array(96).fill(0)
      },
    };
  },
  
  mounted() {
    // Initialize appliances with defaults if none provided
    if (this.appliances.length === 0) {
      this.appliances = [
        // Group 1: Fixed power, must run for full duration
        { id: 1, name: 'Dishwasher', type: 'dishwasher', power: 1.8, active: false, group: 1 },
        { id: 2, name: 'Wash Machine', type: 'wash_machine', power: 0.4, active: false, group: 1 },
        { id: 3, name: 'Clothes Dryer', type: 'clothes_dryer', power: 1.2, active: false, group: 1 },
      
        // Group 2: Variable power, can be toggled during allowed time
        { id: 4, name: 'HVAC', type: 'hvac', power: 2.5, active: false, group: 2 },
        { id: 5, name: 'Water Heater', type: 'water_heater', power: 4.5, active: false, group: 2 },
        { id: 6, name: 'EV Charger', type: 'ev_charger', power: 6.0, active: false, group: 2 },
      
        // Group 3: Fixed power, on for entire duration
        { id: 7, name: 'TV', type: 'tv', power: 0.1, active: false, group: 3 },
        { id: 8, name: 'Refrigerator', type: 'refrigerator', power: 0.2, active: false, group: 3 },
        { id: 9, name: 'Lights', type: 'lights', power: 0.2, active: false, group: 3 },
        { id: 10, name: 'Vacuum Cleaner', type: 'vacuum', power: 1.2, active: false, group: 3 },
        { id: 11, name: 'Hair Dryer', type: 'hair_dryer', power: 1.0, active: false, group: 3 }
      ];
    }
    
    // Set up socket connection
    const socket = io('http://localhost:3000', {
      transports: ['websocket', 'polling']
    });
    this.socket = socket;
  
    socket.on('connect', () => {
      console.log("Socket connected!");
      this.socket.emit('get_current_state');
    });
  
    socket.on('disconnect', () => {
      console.log("Socket disconnected!");
    });
  
    socket.on('rl_prediction', (data) => {
      // Only update if data is for current timeStep to avoid race conditions
      if (data && typeof data.timestamp !== 'undefined' && typeof data.day !== 'undefined') {
        // Store the actual prediction
        this.rlPrediction = data;

        // Update application state based on RL prediction
        this.updateSimulationFromRl(data);
        
        // Update history data
        this.updateHistoryData(data);
      } else {
        console.warn("Received invalid RL prediction data:", data);
      }
    });

    socket.on('rl_error', (error) => {
      console.error("RL prediction error:", error);
      // Handle error - maybe display to user
    });

    socket.on('simulation_mode_changed', (data) => {
      this.simulationState = data.mode;
      console.log(`Simulation mode changed to: ${data.mode}`);
    });

    socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
    });
  
    socket.on('simulation_update', (data) => {
      // Only update if this is for manual simulation mode
      if (data.simulationMode === 'manual') {
        // Update app state directly from the simulation
        this.solarOutput = data.solarOutput || 0;
        this.batteryLevel = data.batteryLevel || 35;
        this.batteryStatus = data.batteryStatus || 'empty';
        this.batteryPower = data.batteryPower || 0;
        this.gridPower = data.gridDraw || 0;
        this.houseDemand = data.houseDemand || 0;

        // Update time information
        this.simulationElapsedMinutes = data.elapsedMinutes || 0;
        this.simulationFormattedTime = data.formattedTime || "00:00";
        this.simulationDay = Math.floor(this.simulationElapsedMinutes / (24 * 60)) + 1;

        // Update energy model state if provided
        if (data.energyModels) {
          this.energyModelState = {
            ...this.energyModelState,
            ...data.energyModels
          };
        }

        // Increment simulation steps
        this.simulationSteps++;
    
        // Update appliances based on devices if available
        if (data.devices && data.devices.length > 0) {
          this.appliances = this.appliances.map(app => {
            // Find matching device
            const device = data.devices.find(d => d.Type === app.type);
            if (device) {
              return {
                ...app,
                active: device.Status === 'active',
                power: device.PowerLevel
              };
            }
            return app;
          });
        }
      }
    });

    socket.on('simulation_status', (status) => {
      this.simulationRunning = status.running;

      //console.log('Received simulation_status:', status);

      if (status.mode) {
        this.simulationState = status.mode;
        this.showRlControls = true;
      }

      // Handle preserved time state
      if (status.preservedTime && status.elapsedMinutes) {
        this.simulationElapsedMinutes = status.elapsedMinutes;
      }

      // Handle time limit reached
      if (status.timeLimitReached) {
        this.simulationRunning = false;
      }
    });

    // Handle socket error events
    socket.on('error', (error) => {
      console.error('Socket error:', error);
      alert(`Socket error: ${error.message || 'Unknown error'}`);
    });

    socket.on('simulation_reset', (data) => {
      // Reset all local state
      this.simulationRunning = false;
      this.simulationState = 'idle';
      this.solarOutput = data.solarOutput || 0;
      this.batteryLevel = data.batteryLevel || 35;
      this.batteryStatus = data.batteryStatus || 'empty';
      this.batteryPower = data.batteryPower || 0;
      this.gridPower = data.gridDraw || 0;
      this.houseDemand = data.houseDemand || 0;
      this.currentDay = 1;
      this.currentTimeStep = 0;
      this.rlPrediction = null;
      
      // Stop autoplay if active
      if (this.autoPlayActive) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayActive = false;
      }
      
      // Reset appliances
      this.appliances = this.appliances.map(app => ({
        ...app,
        active: false
      }));
      
      // Reset history data
      this.homeTemperatureHistory = new Array(96).fill(null);
      this.waterTemperatureHistory = new Array(96).fill(null);
      this.appliancePowerHistory = {
        dishwasher: new Array(96).fill(0),
        wash_machine: new Array(96).fill(0),
        clothes_dryer: new Array(96).fill(0)
      };
    });

    socket.on('devices_updated', (response) => {
      // Just acknowledge - don't trigger another update
      //console.log('Devices updated:', response);
  
      // If server calculated a new demand, use it
      if (response.houseDemand) {
        this.houseDemand = response.houseDemand;
      }
    });

    socket.on('simulation_time_limit_reached', (data) => {
      // console.log(`Simulation time limit reached after ${data.elapsedTime.toFixed(1)} seconds`);
    
      // Update local simulation state
      this.simulationRunning = false;
      this.simulationCompleted = true;
    
      // Show a notification to the user
      this.showTimeLimitNotification(data);
    });

    // Add this code to the socket handler section
    socket.on('energy_models_update_ack', (data) => {
      console.log('Energy models update acknowledged by server', data);
    });
  },

  methods: {
    handleBatteryCritical() {
      // Use consistent threshold with the diagram
      const BATTERY_MIN_LEVEL = 10.01;
  
      // Only force empty status when battery is at or below minimum and currently discharging
      if (this.batteryLevel <= BATTERY_MIN_LEVEL && 
         (this.batteryStatus === 'discharging' || this.batteryPower < 0)) {
        console.log(`Battery protection: Level ${this.batteryLevel.toFixed(1)}% reached minimum threshold`);
    
        // Update local state
        this.batteryStatus = 'empty';
        this.batteryPower = 0;
    
        // Update server if connected
        if (this.socket) {
          this.socket.emit('update_battery_state', {
            batteryStatus: 'empty',
            batteryPower: 0,
            batteryLevel: this.batteryLevel,
            useGridForRemaining: true
          });
        }
      }
    },

    handleAppliancePowerUpdate(data) {
      // Find the appliance in the appliances array
      const appIndex = this.appliances.findIndex(app => app.id === data.id);
      
      if (appIndex !== -1) {
        console.log(`App.vue updating ${data.name} power from ${this.appliances[appIndex].power} to ${data.power} kW`);
        
        // Create a new array with the updated appliance to ensure reactivity
        const updatedAppliances = [...this.appliances];

        updatedAppliances[appIndex] = {
          ...updatedAppliances[appIndex],
          power: data.power
        };
        
        // Replace the entire array to trigger reactivity
        this.appliances = updatedAppliances;
        
        // If needed, also update houseDemand
        this.calcHouseDemand();
        
        // If you have a socket connection, send update to server
        if (this.socket) {
          this.socket.emit('update_devices', {
            devices: this.appliances.map(app => ({
              id: app.id,
              name: app.name,
              type: app.type,
              power: app.power,
              active: app.active
            }))
          });
        }
      } else {
        console.warn(`Appliance with id ${data.id} not found in App.vue`);
      }
    },

    calcHouseDemand() {
      const newHouseDemand = parseFloat(
      this.appliances
        .filter(app => app.active)
        .reduce((sum, app) => sum + (parseFloat(app.power) || 0), 0)
        .toFixed(1)
      );
    
      // Update houseDemand
      this.houseDemand = newHouseDemand;
    },

    onDataLoaded(data) {
      console.log('Hourly data loaded:', data);
      // You can store the data if needed
      this.hourlyDataLoaded = true;
    },

    onHourDataUpdated(hourData) {
      //console.log('Hour data updated:', hourData);
      // Update local state with the latest hour data
      this.currentHourData = hourData;
    },

    showAlert(message) {
      // Use setTimeout to allow UI updates to complete first
      setTimeout(() => {
        try {
          window.alert(message);
        } catch (e) {
          console.error("Error showing alert:", e);
        }
      }, 100);
    },

    startSimulation(options = {}) {
      // Reset simulation state
      this.simulationRunning = true;
      this.simulationState = 'manual';
      this.simulationSteps = 0;
      this.completionAlertShown = false; // Add this to track if alert was shown
      this.simulationMessage = ""; // Add this for UI messages
      
      // Convert hours to steps (each step is 15 minutes)
      const requestedHours = options.timeLimit ? options.timeLimit / 3600 : this.simulationHours || 1;
      const requestedSteps = Math.floor(requestedHours * 4); // 4 steps per hour
      
      console.log(`Starting simulation for ${requestedHours} hours (${requestedSteps} steps)`);
      
      // Set maximum steps - either from options or calculated from hours
      this.maxSimulationSteps = options.maxSteps || requestedSteps;
      
      // Reset simulation time tracking
      this.simulationElapsedMinutes = 0;
      
      // Tell server to start simulation
      if (this.socket) {
        this.socket.emit('start_simulation', {
          timeLimit: requestedHours * 3600, // Convert to seconds
          initialState: {
            simulationMode: 'manual',
            // Rest of initialState as before
          },
          interval: this.simulationInterval
        });
        
        // Set up a progress checking interval
        if (this.simulationProgressTimer) {
          clearInterval(this.simulationProgressTimer);
        }
        
        // Check progress every second
        this.simulationProgressTimer = setInterval(() => {
          this.checkSimulationCompletion();
        }, 1000);
      }
    },

    // Add this method to validate appliance power values
    validateApplianceState(appliances) {
      if (!appliances || !Array.isArray(appliances)) return appliances;
      
      return appliances.map(app => {
        // Create a new object to avoid mutation issues
        const validatedApp = {...app};
        
        // If appliance is active but has no power, use default
        if (validatedApp.active && 
            (validatedApp.power === 0 || isNaN(validatedApp.power))) {
          validatedApp.power = validatedApp.defaultPower || 1.0;
          console.log(`Fixed zero power for active appliance: ${validatedApp.name}`);
        }
        
        return validatedApp;
      });
    },

    checkSimulationCompletion() {
      // Only check if we have a set maximum and we're running
      if (this.maxSimulationSteps && this.simulationRunning) {
        // Convert elapsed time to steps (each step is 15 minutes)
        const timeBasedSteps = Math.floor(this.simulationElapsedMinutes / 15);
        
        // Log current progress
        //console.log(`Simulation progress: ${timeBasedSteps} steps of ${this.maxSimulationSteps} (${this.simulationElapsedMinutes} minutes elapsed)`);
        
        // Check if we've reached the limit based on time elapsed, not data updates
        if (timeBasedSteps >= this.maxSimulationSteps) {
          console.log("Simulation complete - reached time limit");
          this.pauseSimulation();
          
          // Show completion message without alert - add a message to UI instead
          this.simulationMessage = "Simulation complete - reached end of data";
          
          // Only show alert if not previously shown
          if (!this.completionAlertShown) {
            this.completionAlertShown = true;
            
            // Use setTimeout to avoid blocking UI
            setTimeout(() => {
              try {
                alert("Simulation complete. Reached the end of available data.");
              } catch (e) {
                console.error("Error showing completion alert:", e);
              }
            }, 100);
          }
        }
      }
    },

    // Handle hourly data updates
    handleHourlyDataUpdate(data) {
      // Debug logging for simulation steps
      // console.log(`Hourly data update - step: ${this.simulationSteps}, max: ${this.maxSimulationSteps}`);
 
      if (data) {
        data.devices = this.validateApplianceState(data.devices);
        this.hourlyData = data;

        // If energy models data is included, update that too
        if (data.energyModels) {
          this.energyModelState = {
            ...this.energyModelState,
            ...data.energyModels
          };
        }
      }
      
      // If in manual mode, update simulation values
      if (data && this.simulationState === 'manual' && this.simulationRunning) {
        if (typeof data.solarOutput === 'number') {
          // Update solar output from hourly data
          this.$emit('solar-output-updated', data.solarOutput);

          // Send the hourly data to the server
          if (this.socket) {
            this.socket.emit('hourly_data_update', {
              solarOutput: data.solarOutput,
              temperature: data.temperature || 0,
              price: data.price || 0,
              hour: data.hour || "00:00",
              simulationStep: data.simulationStep,
              energyModels: data.energyModels || null
            });
          }
        }

        // Increment simulation steps
        this.simulationSteps++;
      }
    },
  
    // Add method to handle simulation completion
    handleSimulationComplete() {
      this.pauseSimulation();
      // this.showAlert("Simulation complete. Reached the end of available hourly data.");
    },

    // Handle energy models update from EnergyFlowDiagram
    handleEnergyModelsUpdate(data) {
      // Update local state
      this.energyModelState = data;
      
      // Update server if in manual mode
      if (this.simulationState === 'manual' && this.simulationRunning && this.socket) {
        this.socket.emit('energy_models_update', data);
      }
    },
    
    pauseSimulation() {
      this.simulationRunning = false;
      //console.log(`pauseSimulation in App.vue: state=${this.simulationState}, running=${this.simulationRunning}`);
  
      // Pause autoplay if active
      if (this.autoPlayActive) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
        this.autoPlayActive = false;
      }

      // Call memory optimization
      this.optimizeMemoryUsage();
  
      // Tell server to stop
      if (this.socket) {
        this.socket.emit('stop_simulation');
      }
    },
    
    resumeSimulation() {
      this.simulationRunning = true;

      //console.log(`resumeSimulation in App.vue: state=${this.simulationState}, running=${this.simulationRunning}`);
 
      // Tell server to restart
      if (this.socket) {
        let timeScale = 60;
        // Safely try to get timeScale from selected speed
        try {
          if (this.selectedSpeed !== undefined && Array.isArray(this.speedOptions)) {
            const speedOption = this.speedOptions.find(option => option.value === this.selectedSpeed);
            if (speedOption) {
              timeScale = speedOption.value * 30; // Convert to timeScale
            }
          }
        } catch (error) {
          console.warn("Could not determine timeScale from speed options:", error);
        }
 
        this.socket.emit('start_simulation', {
          initialState: {
            simulationMode: this.simulationState,
            devices: this.appliances.map(app => ({
              id: app.id,
              name: app.name,
              type: app.type,
              power: app.power,
              active: app.active
            })),
            // Add current elapsed time info
            preserveTime: true,
            elapsedMinutes: this.simulationElapsedMinutes,
            timeScale: timeScale // Pass current speed setting
          },
          interval: this.simulationInterval || 1000,
          maxSteps: this.maxSimulationSteps
        });
      }
    },
    
    resetSimulation() {
      // Stop any active simulations
      this.simulationRunning = false;
      this.simulationState = 'idle';
      this.simulationCompleted = false;

      //console.log(`resetSimulation in App.vue: state=${this.simulationState}, running=${this.simulationRunning}`);
 
      // Stop autoplay if active
      if (this.autoPlayActive) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
        this.autoPlayActive = false;
      }
  
      // Reset all values to initial state
      this.solarOutput = this.initialState.solarOutput;
      this.batteryLevel = this.initialState.batteryLevel;
      this.batteryStatus = this.initialState.batteryStatus;
      this.batteryPower = 0;
      this.gridPower = this.initialState.gridPower;
      this.houseDemand = this.initialState.houseDemand;

      // Reset RL simulation state
      this.currentDay = 1;
      this.currentTimeStep = 0;
      this.rlPrediction = null;
  
      // Reset appliances to inactive
      this.appliances = this.appliances.map(app => ({
        ...app,
        active: false
      }));

      // Reset history data
      this.homeTemperatureHistory = new Array(96).fill(null);
      this.waterTemperatureHistory = new Array(96).fill(null);
      this.appliancePowerHistory = {
        dishwasher: new Array(96).fill(0),
        wash_machine: new Array(96).fill(0),
        clothes_dryer: new Array(96).fill(0)
      };

      // Tell server to reset
      if (this.socket) {
        this.socket.emit('reset_simulation');
      }

      ChartManager.optimizeMemoryUsage();

      // Reset energy model state
      this.energyModelState = {
        indoorTemperature: 24.0,
        waterTemperature: 55.0,
        evSoC: 0.0,
        evConnected: false,
        hvacPower: 0,
        waterHeaterPower: 0,
        evPower: 0
      };
    },

    optimizeMemoryUsage() {
      // Clear references to large objects
      this.pendingAnimationFrames = [];
        
      // Force cleanup of temperature history for data points we don't need
      if (this.currentTimeStep > 0) {
        const keepPoints = 10; // Only keep recent points and future points
        const startClear = Math.max(0, this.currentTimeStep - keepPoints);
        //const endClear = Math.min(startClear + keepPoints, this.temperatureData.length);
          
        // Clear unneeded history - make sure we use the correct property names
        for (let i = 0; i < startClear; i++) {
          if (this.homeTemperatureHistory) {
            this.homeTemperatureHistory[i] = null;
          }
          if (this.waterTemperatureHistory) {
            this.waterTemperatureHistory[i] = null;
          }
        }
      } 
    },  
    
    toggleAppliance(data) {
      // Handle both simple ID toggle and more complex data update
      if (typeof data === 'number' || typeof data === 'string') {
        // Find the appliance and toggle its status
        this.appliances = this.appliances.map(app => {
          if (app.id === data) {
            return { ...app, active: !app.active };
          }
          return app;
        });
    
        // Calculate new house demand for immediate UI feedback
        const newHouseDemand = parseFloat(
          this.appliances
            .filter(app => app.active)
            .reduce((sum, app) => sum + (parseFloat(app.power) || 0), 0)
            .toFixed(1)
        );

        // If in manual mode, send device update to server
        if (this.simulationRunning && this.simulationState === 'manual' && this.socket) {
          // Prepare device data for server
          const deviceData = {
            devices: this.appliances.map(app => ({
              id: app.id,
              name: app.name,
              type: app.type,
              power: app.power,
              active: app.active
            })),
            houseDemand: newHouseDemand
          };
      
          // Send update to server
          this.socket.emit('update_devices', deviceData);
        }
      } else if (data && data.devices) {
        // Update from structured device data
        this.appliances = this.appliances.map(app => {
          const device = data.devices.find(d => d.id === app.id);
          if (device) {
            return { ...app, active: device.active };
          }
          return app;
        });
    
        // Update house demand if provided
        if (data.newDemand !== undefined) {
          this.houseDemand = data.newDemand;
        }
      }
    },

    updateHouseDemand(newDemand) {
      this.houseDemand = newDemand;
    },

    // Handler for grid-power-updated event
    updateGridPower(newPower) {
      this.gridPower = newPower;
    },

    handleSpeedChange(speedData) {
      // Store the new simulation interval
      this.simulationInterval = speedData.interval;

      const timeScale = speedData.animationSpeed * 30;
    
      // If simulation is running, update the server with new interval
      if (this.simulationRunning && this.socket) {
        // Tell server about new speed
        this.socket.emit('set_time_scale', {
          timeScale: timeScale, // Convert animation speed to timeScale
          animationSpeed: speedData.animationSpeed 
        });
        
        // If autoplay is active, reset it with new interval
        if (this.autoPlayActive) {
          clearInterval(this.autoPlayTimer);
          this.autoPlayTimer = setInterval(() => {
            this.nextStep();
          }, speedData.interval);
        }
      }
    },
    
    handleConfigUpdated(configData) {
      // Update the appliances from the config
      if (configData.appliances) {
        this.appliances = configData.appliances.map(app => ({
          id: app.id,
          name: app.name,
          type: app.type || 'other',
          power: parseFloat(app.power || app.defaultPower) || 0,
          defaultPower: parseFloat(app.defaultPower) || 0,
          active: Boolean(app.active),
          group: app.group || 3
        }));
        console.log("Updated appliances from config:", this.appliances);
      }

      // Reset simulation state to ensure proper recalculation
      this.simulationRunning = false;
      this.simulationState = 'idle';
      this.batteryStatus = 'empty';
      this.solarOutput = 0;
      this.gridPower = 0;
      this.houseDemand = 0;
    },

    updateSimulationFromRl(rlPrediction) {
      if (!rlPrediction || rlPrediction.error) {
        console.warn("Invalid RL prediction:", rlPrediction?.error || "Missing data");
        return;
      }

      // Validate the RL prediction data structure
      if (!rlPrediction.appliances || !rlPrediction.environment || 
          !rlPrediction.battery || !rlPrediction.energy_flow) {
        console.error("Invalid RL prediction data structure:", rlPrediction);
        return;
      }
      
      // Update appliance states
      this.appliances = this.appliances.map(app => {
        let isActive = app.active; // Default to current state
        let power = app.power;
      
        try {
          // Match appliance type to determine new active state from RL prediction
          if (app.type === 'hvac' && rlPrediction.appliances?.controllable?.hvac) {
            isActive = rlPrediction.appliances.controllable.hvac.active;
            power = rlPrediction.appliances.controllable.hvac.power;
          } else if (app.type === 'water_heater' && rlPrediction.appliances?.controllable?.water_heater) {
            isActive = rlPrediction.appliances.controllable.water_heater.active;
            power = rlPrediction.appliances.controllable.water_heater.power;
          } else if (app.type === 'ev_charger' && rlPrediction.ev) {
            isActive = rlPrediction.ev.power > 0;
            power = rlPrediction.ev.power;
          } else if (app.type === 'dishwasher' && rlPrediction.appliances?.shiftable?.dishwasher) {
            isActive = rlPrediction.appliances.shiftable.dishwasher.active;
            power = rlPrediction.appliances.shiftable.dishwasher.power;
          } else if (app.type === 'wash_machine' && rlPrediction.appliances?.shiftable?.wash_machine) {
            isActive = rlPrediction.appliances.shiftable.wash_machine.active;
            power = rlPrediction.appliances.shiftable.wash_machine.power;
          } else if (app.type === 'clothes_dryer' && rlPrediction.appliances?.shiftable?.clothes_dryer) {
            isActive = rlPrediction.appliances.shiftable.clothes_dryer.active;
            power = rlPrediction.appliances.shiftable.clothes_dryer.power;
          } else if (app.type === 'tv' && rlPrediction.appliances?.fixed?.tv) {
            isActive = rlPrediction.appliances.fixed.tv.active;
            power = rlPrediction.appliances.fixed.tv.power;
          } else if (app.type === 'refrigerator' && rlPrediction.appliances?.fixed?.refrigerator) {
            isActive = rlPrediction.appliances.fixed.refrigerator.active;
            power = rlPrediction.appliances.fixed.refrigerator.power;
          } else if (app.type === 'lights' && rlPrediction.appliances?.fixed?.lights) {
            isActive = rlPrediction.appliances.fixed.lights.active;
            power = rlPrediction.appliances.fixed.lights.power;
          } else if (app.type === 'vacuum' && rlPrediction.appliances?.fixed?.vacuum) {
            isActive = rlPrediction.appliances.fixed.vacuum.active;
            power = rlPrediction.appliances.fixed.vacuum.power;
          } else if (app.type === 'hair_dryer' && rlPrediction.appliances?.fixed?.hair_dryer) {
            isActive = rlPrediction.appliances.fixed.hair_dryer.active;
            power = rlPrediction.appliances.fixed.hair_dryer.power;
          }
        } catch (error) {
          console.error(`Error updating ${app.name}:`, error);
        }

        return {
          ...app,
          active: isActive,
          power: power
        };
      });
    
      // Update power values
      this.solarOutput = rlPrediction.environment?.solar_production || 0;
      this.gridPower = rlPrediction.energy_flow?.grid?.net_power || 0;
      this.houseDemand = rlPrediction.energy_flow?.house?.demand?.total || 0;
      this.batteryLevel = (rlPrediction.battery?.soc || 0) * 100; // Convert to percentage
      this.batteryPower = rlPrediction.battery?.power || 0;
    
      // Update battery status
      if (rlPrediction.battery?.power > 0) {
        this.batteryStatus = 'charging';
      } else if (rlPrediction.battery?.power < 0) {
        this.batteryStatus = 'discharging';
      } else {
        this.batteryStatus = 'empty';
      }
    },

    onHistoryUpdated(data) {
      // Ensure our parent component knows when history has been updated
      console.log(`History updated for timeStep ${data.timeStep}`);
  
      // This is optional, but helps ensure parent-child sync
      if (data.timeStep === this.currentTimeStep) {
        if (data.homeTemp !== null) {
          this.homeTemperatureHistory[data.timeStep] = data.homeTemp;
        }
        if (data.waterTemp !== null) {
          this.waterTemperatureHistory[data.timeStep] = data.waterTemp;
        }
        // Don't need to update appliance history as it's passed by reference
      }
    },
    
    updateHistoryData(data) {
      if (!data) return;

      try {
        const timeStep = data.timestamp;
        
        // Update temperature histories
        if (timeStep >= 0 && timeStep < 96) {
          // Create new arrays to ensure reactivity
          const newHomeHistory = [...this.homeTemperatureHistory];
          const newWaterHistory = [...this.waterTemperatureHistory];
          const newApplianceHistory = JSON.parse(JSON.stringify(this.appliancePowerHistory));
      
          // Update home temperature
          const homeTemp = data.temperatures?.home?.current || null;
          newHomeHistory[timeStep] = homeTemp;
      
          // Update water temperature
          const waterTemp = data.temperatures?.water?.current || null;
          newWaterHistory[timeStep] = waterTemp;
      
          const washMachine = data.appliances.shiftable.wash_machine;
          const washingCompleted = washMachine.progress >= washMachine.total_duration;
          const clothesDryer = data.appliances.shiftable.clothes_dryer;

          // Update appliance power histories
          if (data.appliances && data.appliances.shiftable) {

            for (const appKey in data.appliances.shiftable) {
              if (newApplianceHistory[appKey]) {
                const app = data.appliances.shiftable[appKey];
                let power = 0;

                if (appKey === 'clothes_dryer') {
                  power = (app.progress < app.total_duration && washingCompleted) ? app.power : 0;
                } else {
                  power = app.active && app.progress < app.total_duration ? app.power : 0;
                }

                newApplianceHistory[appKey][timeStep] = power;
              }
            }
          }

          // Force update to make sure arrays are reactive
          this.homeTemperatureHistory = newHomeHistory;
          this.waterTemperatureHistory = newWaterHistory;
          this.appliancePowerHistory = newApplianceHistory;
        }
      } catch (err) {
        console.error('Error in updateHistoryData:', err);
      }
    },
    
    formatTime(timeStep) {
      // Add 8 hours (32 quarters) to start from 8:00 AM
      const adjustedStep = (timeStep + 32) % 96;
      // Each time step is 15 minutes
      const minutes = (adjustedStep % 4) * 15;
      const hours = Math.floor(adjustedStep / 4);
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    },

    previousStep() {
      if (this.canGoBack && !this.autoPlayActive) {
        // Store current state before changing
        const prevDay = this.currentDay;
        const prevStep = this.currentTimeStep;

        this.currentTimeStep = (this.currentTimeStep - 1 + 96) % 96;
        if (this.currentTimeStep === 95) {
          this.currentDay = Math.max(1, this.currentDay - 1);
        }

        // Log for debugging
        console.log(`Time step changed: ${prevDay}:${prevStep} -> ${this.currentDay}:${this.currentTimeStep}`);

        this.requestRlPrediction();
      }
    },

    nextStep() {
      if (this.autoPlayActive) return;
      
      // Store current state before changing
      const prevDay = this.currentDay;
      const prevStep = this.currentTimeStep;

      // Update step consistently
      this.currentTimeStep = (this.currentTimeStep + 1) % 96;
      if (this.currentTimeStep === 0) {
        this.currentDay = Math.min(60, this.currentDay + 1);
      }

      // Log for debugging
      console.log(`Time step changed: ${prevDay}:${prevStep} -> ${this.currentDay}:${this.currentTimeStep}`);

      this.requestRlPrediction();
    },

    toggleAutoplay() {
      // Stop any existing timer first
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
      }
  
      // Toggle state
      this.autoPlayActive = !this.autoPlayActive;
  
      if (this.autoPlayActive) {
        // Change to AI mode
        this.simulationState = 'ai';
    
        // Tell server to start AI mode
        if (this.socket) {
          this.socket.emit('set_simulation_mode', { 
            mode: 'ai',
            interval: this.simulationInterval
          });
        }
    
        // Start the timer with a simple implementation
        this.autoPlayTimer = setInterval(() => {
          // Simple direct state update
          this.currentTimeStep = (this.currentTimeStep + 1) % 96;
          if (this.currentTimeStep === 0) {
            this.currentDay = Math.min(60, this.currentDay + 1);
          }
      
          // Request new prediction after state update
          this.requestRlPrediction();

          if (this.currentTimeStep % 10 === 0) {
            this.optimizeMemoryUsage();
          }
        }, this.simulationInterval);
      } else {
        // Stop autoplay
        if (this.socket) {
          this.socket.emit('set_simulation_mode', { mode: 'none' });
        }
      }
    },
    
    requestRlPrediction() {
      // Emit a socket event to request updated RL prediction for current time step
      if (this.socket) {
        try {
          // Change to AI mode
          this.simulationState = 'ai';

          // Log current state for debugging
          console.log(`Requesting RL prediction for day ${this.currentDay}, step ${this.currentTimeStep}`);

          // Add a timeout to prevent hanging
          const requestTimeout = setTimeout(() => {
            console.warn('RL prediction request timed out');
          }, 5000);

          // Set default initial values
          if (!this.rlPrediction) {
            const initialRequest = {
              day: this.currentDay,
              timeStep: this.currentTimeStep,
              shift_prog: [0, 0, 0],
              time_to_start_shift: [0, 0, 0],
              home_temp: 4.0,  // 4°C above setpoint
              water_temp: -5.0,  // 5°C below setpoint
              soc_ess: 0.0,
              soc_ev: 0.0,
              time_to_start_noctrl: [0, 0, 0, 0, 0]
            };

            this.socket.emit('request_rl_prediction', initialRequest);
            return;
          }
          
          const requestData = {
            day: this.currentDay,
            timeStep: this.currentTimeStep,
            
            // Appliance progress state
            shift_prog: [
              this.rlPrediction?.appliances?.shiftable?.dishwasher?.progress || 0,
              this.rlPrediction?.appliances?.shiftable?.wash_machine?.progress || 0,
              this.rlPrediction?.appliances?.shiftable?.clothes_dryer?.progress || 0
            ],
            
            // Time to start flags for appliances
            time_to_start_shift: [
              this.rlPrediction?.appliances?.shiftable?.dishwasher?.active ? 1 : 0,
              this.rlPrediction?.appliances?.shiftable?.wash_machine?.active ? 1 : 0,
              this.rlPrediction?.appliances?.shiftable?.clothes_dryer?.active ? 1 : 0
            ],
            
            // IMPORTANT: Convert absolute temperatures to differences from setpoints
            home_temp: (this.rlPrediction?.temperatures?.home?.current || 26) - 
                      (this.rlPrediction?.temperatures?.home?.setpoint || 22),
                      
            water_temp: (this.rlPrediction?.temperatures?.water?.current || 55) - 
                       (this.rlPrediction?.temperatures?.water?.setpoint || 60),
            
            // Battery and EV state
            soc_ess: this.rlPrediction?.battery?.soc || 0.0,
            soc_ev: this.rlPrediction?.ev?.soc || 0.0,
            
            // Fixed appliance state
            time_to_start_noctrl: [
              this.rlPrediction?.appliances?.fixed?.tv?.active ? 1 : 0,
              this.rlPrediction?.appliances?.fixed?.refrigerator?.active ? 1 : 0,
              this.rlPrediction?.appliances?.fixed?.lights?.active ? 1 : 0,
              this.rlPrediction?.appliances?.fixed?.vacuum?.active ? 1 : 0,
              this.rlPrediction?.appliances?.fixed?.hair_dryer?.active ? 1 : 0
            ]
          };

          this.socket.emit('request_rl_prediction', requestData);

          // Clear the timeout when socket response is received
          this.socket.once('rl_prediction', () => {
            clearTimeout(requestTimeout);
          });
        } catch (error) {
          console.error('Error requesting RL prediction:', error);
        }
      } else {
        console.warn('Socket not available for RL prediction request');
      }
    },

    showTimeLimitNotification(data) {
      // For simplicity, using alert, but you can replace with a more elegant UI notification
      const hours = Math.floor(data.timeLimit / 60);
      const minutes = Math.floor(data.timeLimit % 60);
    
      let timeString = '';
      if (hours > 0) timeString += `${hours} hour${hours > 1 ? 's' : ''} `;
      if (minutes > 0) timeString += `${minutes} minute${minutes > 1 ? 's' : ''} `;
      //if (seconds > 0) timeString += `${seconds} second${seconds > 1 ? 's' : ''}`;
    
      //alert(`Simulation complete! Time limit of ${timeString} reached.`);
      this.showAlert(`Simulation complete! Time limit of ${timeString} reached.`);
    },

    resetChartData() {
      if (!this.chart || !this.isChartInitialized) return;
      
      try {
        // Reset data cache
        this.dataCache = {};
        
        // Create empty datasets for each appliance
        for (const appKey in this.shiftableAppliances) {
          this.dataCache[appKey] = new Array(this.maxDataPoints).fill(0);
        }
        
        // Update chart datasets
        if (this.chart.data && this.chart.data.datasets) {
          let i = 0;
          for (const appKey in this.shiftableAppliances) {
            if (i < this.chart.data.datasets.length) {
              this.chart.data.datasets[i].data = [...this.dataCache[appKey]];
              i++;
            }
          }
          
          // Update chart without animation
          this.chart.update('none');
        }
      } catch (error) {
        console.warn('Error resetting appliance chart data:', error.message);
      }
    },
  },

  watch: {
    // Add a watcher to detect changes to simulationState
    simulationState(newState, oldState) {
      //console.log(`Simulation state changed from ${oldState} to ${newState}`);
      
      // Always keep RL controls visible regardless of state
      this.showRlControls = true;
    },

    timeStep(newValue) {
      // Special case: if timeStep is reset to 0, we might be doing a full reset
      if (newValue === 0) {
        // Check if we need to reset - look at the power history
        let needsReset = true;
        for (const appKey in this.powerHistory) {
          // If any appliance has non-zero data, don't reset
          if (this.powerHistory[appKey].some(v => v > 0)) {
            needsReset = false;
            break;
          }
        }
        
        if (needsReset) {
          this.resetChartData();
        }
      }
      
      this.updateChart();
    }
  },

  computed: {
    canGoBack() {
      // Can only go back if not at the beginning of day 1
      return !(this.currentDay === 1 && this.currentTimeStep === 0);
    },

    showStartButton() {
      console.log(`Computing showStartButton: simulationState=${this.simulationState}`);
      return this.simulationState === 'idle';
    },
    showPauseButton() {
      console.log(`Computing showPauseButton: isRunning=${this.isRunning}, simulationState=${this.simulationState}`);
      return this.isRunning && this.simulationState !== 'idle';
    },
    showResumeButton() {
      console.log(`Computing showResumeButton: isRunning=${this.isRunning}, simulationState=${this.simulationState}`);
      return !this.isRunning && this.simulationState !== 'idle';
    },
    showResetButton() {
      console.log(`Computing showResetButton: simulationState=${this.simulationState}`);
      return this.simulationState !== 'idle';
    },
    formattedTimeStep() {
      return this.formatTime(this.currentTimeStep);
    },
    formattedStepDisplay() {
      return `${this.currentTimeStep + 1}/96`;
    }
  },
  
  beforeUnmount() {
    // Clean up chart manager completely
    if (window.ChartManager) {
      // Reset all charts
      window.ChartManager.resetAllCharts();
    }

    // Clean up any timers when component is destroyed
    if (this.autoPlayTimer) {
      clearInterval(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    
    // Disconnect socket
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }

    // Clear any large data structures
    this.rlPrediction = null;
    this.homeTemperatureHistory = null;
    this.waterTemperatureHistory = null;
    this.appliancePowerHistory = null;
  }
};

</script>

<style>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 10px;
  background-color: #f0f0f0;
  border-radius: 8px;
}

.rl-simulation-controls {
  margin-top: 20px;
  padding: 20px;
  background-color: #f8fafc;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.rl-simulation-controls h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #1f2937;
}

.rl-simulation-info {
  display: flex;
  gap: 20px;
  margin-bottom: 15px;
}

.rl-step-display {
  display: flex;
  align-items: center;
  gap: 8px;
}

.rl-step-label {
  font-size: 14px;
  color: #6b7280;
}

.rl-step-value {
  font-size: 16px;
  font-weight: bold;
  color: #1f2937;
}

.rl-step-buttons {
  display: flex;
  gap: 10px;
}

.rl-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.rl-prev-btn {
  background-color: #e5e7eb;
  color: #4b5563;
}

.rl-next-btn {
  background-color: #3b82f6;
  color: white;
}

.rl-play-btn {
  background-color: #10b981;
  color: white;
}

.rl-pause-btn {
  background-color: #ef4444;
  color: white;
}

.rl-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.main-layout {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 20px;
}

.diagram-section {
  flex: 6;
  min-width: 500px; /* Adjust based on your layout needs */
}

.charts-section {
  flex: 4;
  min-width: 400px; /* Adjust based on your layout needs */
}

/* For mobile devices */
@media (max-width: 1200px) {
  .main-layout {
    flex-direction: column;
  }
  
  .diagram-section,
  .charts-section {
    width: 100%;
    min-width: 100%;
  }
}
</style> 
