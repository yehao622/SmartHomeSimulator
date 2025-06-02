<template>
  <div class="energy-flow-container">
    <div class="controls">   
      <div class="config-controls" v-if="simulationState === 'idle'">
        <div class="simulation-time-input">
          <label for="simulation-hours">Simulation hours:</label>
          <input 
            id="simulation-hours" 
            type="number" 
            v-model.number="simulationHours" 
            min="0.1" 
            step="0.1"
            placeholder="Hours"
          />
          <span>hours</span>
        </div>

        <button @click="loadConfigFile" class="config-btn">
          Load Configuration
        </button>
        <input 
          type="file" 
          ref="configFileInput" 
          @change="onConfigFileSelected" 
          accept=".json,.yaml,.yml" 
          style="display: none"
        >
        
        <button @click="showConfigEditor = true" class="config-btn">
          Edit Configuration
        </button>
      </div>
      
      <div class="control-buttons">
        <button 
          :style="{ display: simulationState === 'idle' ? 'inline-block' : 'none' }"
          @click="startSimulation"
          class="start-btn"
        >
          Start
        </button>

        <button 
          :style="{ display: isRunning && simulationState !== 'idle' ? 'inline-block' : 'none' }"
          @click="pauseSimulation"
          class="pause-btn"
        >
          Pause
        </button>

        <button 
          :style="{ display: !isRunning && simulationState !== 'idle' && !simulationCompleted ? 'inline-block' : 'none' }"
          @click="resumeSimulation"
          class="resume-btn"
        >
          Resume
        </button>

        <button 
          :style="{ display: simulationState !== 'idle' ? 'inline-block' : 'none' }"
          @click="resetSimulation"
          class="reset-btn"
        >
          Reset
        </button>
        
        <div class="speed-control" :style="{ display: 'flex' }">
          <span>Animation speed:</span>
          <select v-model="selectedSpeed" @change="onSpeedChange">
            <option v-for="option in speedOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </div>
      </div>
    </div>

    <div class="simulation-time-display" v-if="isRunning || simulationState !== 'idle'">
      <div class="time-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
      </div>
      <div class="time-value">{{ formattedSimulationTime }}</div>
      <div class="time-day" v-if="localSimulationDay > 1">Day {{ localSimulationDay }}</div>
    </div>
    
    <div class="diagram">
      <svg width="100%" height="400" viewBox="0 0 1000 400">
        <!-- Background Grid -->
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#eee" stroke-width="1" />
        </pattern>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        <!-- Energy Flow Lines -->
        <g v-for="(connection, idx) in activeConnections" :key="`conn-${idx}`">
          <line 
            :x1="connection.x1" 
            :y1="connection.y1" 
            :x2="connection.x2" 
            :y2="connection.y2" 
            :stroke="connection.color"
            :stroke-width="connection.strokeWidth" 
            stroke-dasharray="5,5"
            :class="{
              'flow-line': true,
              [`flow-${connection.direction}`]: 
                isRunning && 
                connection.powerFlow > 0.1 && 
                connection.active &&
                // Special check for battery discharge - no animation when close to minimum
                (connection.from !== 'battery' || batteryLevel > 10.1)
            }"
          />
          <text 
            v-if="connection.active && connection.powerFlow > 0.1"
            :x="connection.labelX" 
            :y="connection.labelY" 
            :fill="connection.color" 
            text-anchor="middle" 
            font-size="12"
            font-weight="bold"
            class="flow-value"
          >
            {{ connection.powerFlow.toFixed(1) }} kW
          </text>
        </g>
        
        <!-- Power Sources -->
        <g 
          v-for="source in powerSources" 
          :key="source.id" 
          class="device-icon" 
          :class="{ 'pulse': source.active && simulationState === 'running' }" 
          :transform="`translate(${source.x}, ${source.y})`"
        >
          <rect 
            :x="source.rectX" 
            :y="source.rectY" 
            :width="source.width" 
            :height="source.height" 
            :rx="source.rx" 
            :class="['device-bg', { 'active': source.active }]"
            :style="{ fill: source.bgFill, stroke: source.stroke }"
          />
          
          <image 
            v-if="source.customIcon" 
            :x="source.iconX" 
            :y="source.iconY" 
            :width="source.iconWidth" 
            :height="source.iconHeight" 
            :href="source.customIcon" 
            :style="{ opacity: source.active ? 1 : 0.5 }"
          />
          
          <svg 
            v-else 
            :x="source.iconX" 
            :y="source.iconY" 
            :width="source.iconWidth" 
            :height="source.iconHeight" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            stroke-width="2" 
            stroke-linecap="round" 
            stroke-linejoin="round" 
            :style="{ color: source.active ? source.stroke : '#9ca3af' }"
            v-html="getIconSvg(source.type)"
          >
          </svg>
          
          <text 
            :x="source.labelX" 
            :y="source.labelY" 
            class="device-label"
          >
            {{ source.name }}
          </text>
          
          <text 
            :x="source.valueX" 
            :y="source.valueY" 
            class="device-value" 
            :style="{ fill: source.active ? source.stroke : '#9ca3af' }"
          >
            {{ source.displayValue }}
          </text>
        </g>
      </svg>
    </div>
    
    <!-- NEW LAYOUT: Two-column layout for appliances and charts -->
    <div class="appliances-and-charts">
      <!-- Column 1: Appliances -->
      <div class="appliances-column">
        <!-- Appliances by Group -->
        <div 
          v-for="group in [1, 2, 3]" 
          :key="`group-${group}`" 
          class="appliance-group"
        >
          <div :class="`appliance-group-header group-${group}`">
            {{ getGroupTitle(group) }}
          </div>
          <div class="appliances-group">
            <div 
              v-for="app in simulatorAppliances.filter(a => a.group === group)" 
              :key="app.id" 
              :class="`appliance-card group-${group} ${app.active ? 'active' : ''}`"
              @click="toggleApplianceStatus(app.id)"
            >
              <div class="appliance-icon" :class="{ 'pulse': app.active && simulationState === 'running' }">
                <image 
                  v-if="getApplianceIcon(app.type).type === 'image'" 
                  :src="getApplianceIcon(app.type).path" 
                  width="24" 
                  height="24" 
                  :style="{ opacity: app.active ? 1 : 0.5 }"
                />
                <svg 
                  v-else 
                  width="24" 
                  height="24" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  stroke-width="2" 
                  stroke-linecap="round" 
                  stroke-linejoin="round" 
                  :style="{ color: app.active ? getGroupColor(group) : '#9ca3af' }"
                  v-html="getApplianceSvg(app.type)"
                >
                </svg>
              </div>
              <div class="appliance-name">{{ app.name }}</div>
              <div class="appliance-power" :class="{ 'active': app.active }">
                {{ app.power }} kW
              </div>
              <div class="appliance-status" :class="{ 'on': app.active, 'off': !app.active }">
                {{ app.active ? 'ON' : 'OFF' }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Column 2: Environmental Charts -->
      <div class="charts-column" v-if="hourlyDataLoaded">
        <div class="environmental-charts">
          <h3>Environmental Data</h3>
          <template v-for="chartType in ['solar', 'price']" :key="chartType">
            <div class="chart-container">
              <h4>{{ getChartTitle(chartType) }}</h4>
              <div class="chart-wrapper" :id="`${chartType}-chart-container`"></div>
              <div class="current-value" v-if="currentHourData">
                Current: {{ formatChartValue(chartType, currentHourData[chartType]) }}
              </div>
            </div>
          </template>
          <div class="chart-container">
            <h4>{{ getChartTitle('temperature') }}</h4>
  
            <!-- Use two separate divs with their own heights -->
            <div class="chart-wrapper">
              <div id="air-temp-chart-container" style="height: 160px; width: 100%;"></div>
            </div>
            
            <div class="chart-wrapper" style="margin-top: 10px;">
              <div id="water-temp-chart-container" style="height: 160px; width: 100%;"></div>
            </div>
            
            <div class="current-value" v-if="currentHourData">
              Current: {{ formatChartValue('temperature', currentHourData.temperature) }}
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Energy Stats -->
    <div class="energy-stats">
      <div 
        v-for="stat in energyStats" 
        :key="stat.type" 
        :class="`stat-card ${stat.type}`"
      >
        <div class="stat-title">{{ stat.title }}</div>
        <div class="stat-value">{{ stat.value }}</div>
      </div>
    </div>
    
    <div class="rl-insights" v-if="rlPrediction">
      <h3>AI Energy Optimizer</h3>
  
      <div class="rl-stats">
        <div v-for="stat in rlStats" :key="stat.id" class="rl-stat-card">
          <div class="rl-stat-title">{{ stat.title }}</div>
          <div class="rl-stat-value" :class="stat.class">{{ stat.value }}</div>
        </div>
      </div>

      <!-- Temperature Charts -->
      <div 
        v-for="temp in ['home', 'water']" 
        :key="`${temp}-temp`" 
        class="rl-chart-container"
      >
        <temperature-chart
          :key="`${temp}-temp-${safeGet(rlPrediction, 'timestamp', 0)}-reset-${resetCounter}`"
          :title="`${temp.charAt(0).toUpperCase() + temp.slice(1)} Temperature`"
          :current-temperature="safeGet(rlPrediction, `temperatures.${temp}.current`, temp === 'home' ? 22 : 60)"
          :set-point="safeGet(rlPrediction, `temperatures.${temp}.setpoint`, temp === 'home' ? 22 : 60)"
          :time-step="safeGet(rlPrediction, 'timestamp', 0)"
          :history="temp === 'home' ? homeTemperatureHistory : waterTemperatureHistory"
        />
      </div>

      <div class="rl-chart-container">
        <appliance-power-chart
          :key="`appliance-power-${safeGet(rlPrediction, 'timestamp', 0)}-reset-${resetCounter}`"
          v-if="rlPrediction && rlPrediction.appliances && rlPrediction.appliances.shiftable"
          :shiftable-appliances="rlPrediction.appliances.shiftable"
          :time-step="safeGet(rlPrediction, 'timestamp', 0)"
          :power-history="appliancePowerHistory"
        />
      </div>

      <!-- Keep the original progress bars for reference (can be removed later) -->
      <div v-if="showProgressBars && rlPrediction && rlPrediction.appliances && rlPrediction.appliances.shiftable" class="rl-progress">
        <div v-for="(app, key) in rlPrediction.appliances.shiftable" :key="`progress-${key}`" 
             class="rl-progress-item" :class="{'active': app.active}">
          <div class="rl-progress-name">{{ formatAppName(key) }}</div>
          <div class="rl-progress-bar-container">
            <div class="rl-progress-bar" :style="{width: `${(app.progress / app.total_duration) * 100}%`}"></div>
          </div>
          <div class="rl-progress-status">
            {{ app.progress }}/{{ app.total_duration }} {{ app.active ? '(Running)' : '' }}
          </div>
        </div>
      </div>
    </div>
    
    <!-- Configuration Editor Modal -->
    <div v-if="showConfigEditor" class="config-editor-modal">
      <div class="config-editor-content">
        <h3>Edit Configuration</h3>
        <div class="tabs">
          <button 
            v-for="tab in configTabs" 
            :key="tab.key" 
            :class="{ active: activeTab === tab.key }" 
            @click="activeTab = tab.key"
          >
            {{ tab.label }}
          </button>
        </div>
        
        <div class="tab-content">
          <!-- Dynamic content rendering based on activeTab -->
          <component 
            :is="getConfigTabComponent(activeTab)" 
            :config="editableConfig" 
            @update="updateEditableConfig"
          />
        </div>
        
        <div class="config-editor-actions">
          <button @click="saveConfigChanges" class="save-btn">Save Changes</button>
          <button @click="exportConfig" class="export-btn">Export Config</button>
          <button @click="showConfigEditor = false" class="cancel-btn">Cancel</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import './EnergyFlowDiagram.css';
import TemperatureChart from './TemperatureChart.vue';
import AppliancePowerChart from './AppliancePowerChart.vue';
import ChartManager from './chartConfig.js';
import { loadConfig, processApplianceConfig, processPowerSourcesConfig, getSimulationSettings, getDefaultConfig } from '../../configLoader.js';
import { readHourlyData, getCurrentHourData } from './csvReader.js';
import { calculateHvacPower, getInitialIndoorTemperature } from './hvacModel.js';
import { calculateWaterHeaterPower, getInitialWaterTemperature } from './waterHeaterModel.js';
import { calculateEvChargingPower, isEvConnected, getInitialEvSoC } from './evChargingModel.js';

// Add these components for the configuration editor
const AppliancesConfigComponent = {
  props: ['config'],
  emits: ['update'],
  template: `
    <div class="appliances-config">
      <div v-for="app in config.appliances" :key="'edit-' + app.id" class="config-item">
        <div class="config-item-header">
          <h4>{{ app.name }}</h4>
          <button @click="$emit('update', { action: 'remove', id: app.id })" class="remove-btn">Remove</button>
        </div>
        <div class="config-fields">
          <div class="field">
            <label>Name:</label>
            <input type="text" v-model="app.name">
          </div>
          <div class="field">
            <label>Type:</label>
            <select v-model="app.type">
              <option value="hvac">HVAC</option>
              <option value="refrigerator">Refrigerator</option>
              <option value="lights">Lights</option>
              <option value="ev_charger">EV Charger</option>
              <option value="dishwasher">Dishwasher</option>
              <option value="water_heater">Water Heater</option>
              <option value="wash_machine">Washing Machine</option>
              <option value="clothes_dryer">Clothes Dryer</option>
              <option value="tv">TV</option>
              <option value="vacuum">Vacuum</option>
              <option value="hair_dryer">Hair Dryer</option>
              <option value="other">Other</option>
            </select>
          </div>
          <div class="field">
            <label>Power (kW):</label>
            <input type="number" v-model.number="app.defaultPower" step="0.1" min="0">
          </div>
          <div class="field">
            <label>Default State:</label>
            <select v-model="app.active">
              <option :value="true">ON</option>
              <option :value="false">OFF</option>
            </select>
          </div>
          <div class="field">
            <label>Appliance Group:</label>
            <select v-model.number="app.group">
              <option :value="1">Fixed Power (Run Full Duration)</option>
              <option :value="2">Variable Power</option>
              <option :value="3">Fixed Power (Toggle Anytime)</option>
            </select>
          </div>
        </div>
      </div>
      <button @click="$emit('update', { action: 'add' })" class="add-btn">Add New Appliance</button>
    </div>
  `
};

const PowerConfigComponent = {
  props: ['config'],
  template: `
    <div class="power-config">
      <div v-for="source in config.powerSources" :key="'edit-' + source.id" class="config-item">
        <div class="config-item-header">
          <h4>{{ source.name }}</h4>
        </div>
        <div class="config-fields">
          <div class="field">
            <label>Name:</label>
            <input type="text" v-model="source.name">
          </div>
          
          <template v-if="source.id === 'solar'">
            <div class="field">
              <label>Capacity (kW):</label>
              <input type="number" v-model.number="source.capacity" step="0.5" min="0">
            </div>
            <div class="field">
              <label>Efficiency (%):</label>
              <input type="number" v-model.number="source.efficiency" step="0.01" min="0.5" max="1">
            </div>
          </template>
          
          <template v-if="source.id === 'battery'">
            <div class="field">
              <label>Capacity (kWh):</label>
              <input type="number" v-model.number="source.capacity" step="0.5" min="0">
            </div>
            <div class="field">
              <label>Initial Level (%):</label>
              <input type="number" v-model.number="source.initialLevel" step="1" min="0" max="100">
            </div>
          </template>
          
          <template v-if="source.id === 'grid'">
            <div class="field">
              <label>Max Power (kW):</label>
              <input type="number" v-model.number="source.maxPower" step="1" min="0">
            </div>
            <div class="field">
              <label>Export Enabled:</label>
              <input type="checkbox" v-model="source.exportEnabled">
            </div>
          </template>
        </div>
      </div>
    </div>
  `
};

const SettingsConfigComponent = {
  props: ['config'],
  template: `
    <div class="settings-config">
      <div class="config-item">
        <h4>Location</h4>
        <div class="config-fields">
          <div class="field">
            <label>Latitude:</label>
            <input type="number" v-model.number="config.settings.location.latitude" step="0.0001">
          </div>
          <div class="field">
            <label>Longitude:</label>
            <input type="number" v-model.number="config.settings.location.longitude" step="0.0001">
          </div>
        </div>
      </div>
      
      <div class="config-item">
        <h4>Simulation</h4>
        <div class="config-fields">
          <div class="field">
            <label>Time Step (seconds):</label>
            <select v-model.number="config.settings.timeStep">
              <option value="60">1 minute</option>
              <option value="300">5 minutes</option>
              <option value="900">15 minutes</option>
              <option value="3600">1 hour</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  `
};


export default {
  components: {
    TemperatureChart,
    AppliancePowerChart
  },

  props: {
    solarOutput: {
      type: Number,
      default: 0
    },
    batteryLevel: {
      type: Number,
      default: 50
    },
    batteryStatus: {
      type: String,
      default: 'empty' // 'charging', 'discharging', or 'empty'
    },
    batteryPower: {
      type: Number,
      default: 0
    },
    gridPower: {
      type: Number,
      default: 0
    },
    houseDemand: {
      type: Number,
      default: 0
    },
    appliances: {
      type: Array,
      default: () => []
    },
    isRunning: {
      type: Boolean,
      default: false
    },
    simulationState: {
      type: String,
      default: 'idle' // 'idle', 'manual', or 'ai'
    },
    simulationCompleted: {
      type: Boolean,
      default: false
    },
    rlPrediction: {
      type: Object,
      default: () => null
    },
    simulationElapsedMinutes: {
      type: Number,
      default: 0
    },
    simulationFormattedTime: {
      type: String,
      default: "00:00"
    },
    simulationDay: {
      type: Number,
      default: 1
    }
  },
  
  data() {
    return {
      animationSpeed: 2,
      simulationInterval: null,
      simulationHours: 1,

      speedOptions: [
        { value: 1, label: "Slow", interval: 1000 },
        { value: 2, label: "Medium", interval: 300 },
        { value: 3, label: "Fast", interval: 100 }
      ],
      selectedSpeed: 1,
      
      customIcons: {},
      
      showConfigEditor: false,
      activeTab: 'appliances',
      editableConfig: null,
      
      powerSources: [],
      connections: [],
      activeConnections: [],
      
      simulatorAppliances: [],

      simulationTime: "00:00",
      localSimulationDay: 1,
      elapsedMinutes: 0,

      hourlyData: [],
      hourlyDataLoaded: false,
      simulationSteps: 0,

      energyModels: {
        indoorTemperature: 22.0,
        waterTemperature: 58.0,
        evSoC: 0.3,
        evConnected: false,
        hvacHistory: [],
        waterHeaterHistory: [],
        evHistory: []
      },
      
      sourcesLayout: {
        solar: { x: 100, y: 90 },
        battery: { x: 500, y: 120 },
        grid: { x: 900, y: 90 },
        house: { x: 500, y: 280 }
      },
      
      defaultConnections: this.getDefaultConnections(),

      homeTemperatureHistory: new Array(96).fill(null),
      waterTemperatureHistory: new Array(96).fill(null),
      appliancePowerHistory: {
        dishwasher: new Array(96).fill(0),
        wash_machine: new Array(96).fill(0),
        clothes_dryer: new Array(96).fill(0)
      },
    
      showProgressBars: false,
      resetCounter: 0,

      chartData: {
        solar: { primary: Array(96).fill(null), battery: Array(96).fill(null) },
        temperature: { primary: Array(96).fill(null), indoor: Array(96).fill(null), water: Array(96).fill(null) },
        price: { primary: Array(96).fill(null) }
      },
      charts: {
        solar: null,
        temperature: null,
        price: null
      },
      chartInitTimeout: null,
      currentHourData: {
        hour: "00:00",
        solar: 0,
        temperature: 20,
        price: 0.01
      },
      lastUpdatedStep: -1,
      chartUpdateInterval: null,
      comfortLineAdded: false,

      configTabs: [
        { key: 'appliances', label: 'Appliances' },
        { key: 'power', label: 'Power Sources' },
        { key: 'settings', label: 'Settings' }
      ]
    };
  },

  computed: {
    formattedSimulationTime() {
      return this.simulationTime;
    },

    powerData() {
      return {
        solarOutput: this.solarOutput,
        batteryLevel: this.batteryLevel,
        batteryStatus: this.batteryStatus,
        gridPower: this.gridPower,
        houseDemand: this.houseDemand
      };
    },

    energyStats() {
      return [
        { type: 'solar', title: 'Solar Production', value: `${this.solarOutput} kW` },
        { type: 'battery', title: 'Battery Level', value: `${this.batteryLevel.toFixed(1)}%` },
        { type: 'grid', title: 'Grid Usage', value: `${this.gridPower} kW` },
        { type: 'home', title: 'Home Demand', value: `${this.houseDemand} kW` }
      ];
    },

    rlStats() {
      if (!this.rlPrediction) return [];
      return [
        {
          id: 'time',
          title: 'Current Time',
          value: `Day ${this.safeGet(this.rlPrediction, 'day', 1)}, ${this.formatTime(this.safeGet(this.rlPrediction, 'timestamp', "00:00"))}`
        },
        {
          id: 'price',
          title: 'Grid Price',
          value: `$${this.safeGet(this.rlPrediction, 'environment.price', 0.003).toFixed(3)}/kWh`
        },
        {
          id: 'demand',
          title: 'Total Demand',
          value: `${this.safeGet(this.rlPrediction, 'energy_flow.house.demand.total', -0.1).toFixed(1)} kW`
        },
        {
          id: 'grid-flow',
          title: 'Grid Import/Export',
          value: `${Math.abs(this.safeGet(this.rlPrediction, 'energy_flow.grid.net_power', 0)).toFixed(1)} kW ${this.safeGet(this.rlPrediction, 'energy_flow.grid.net_power', 0) < 0 ? 'Export' : 'Import'}`,
          class: this.safeGet(this.rlPrediction, 'energy_flow.grid.net_power', 0) < 0 ? 'export' : ''
        }
      ];
    }
  },
  
  async created() {
    await this.loadDefaultConfig();
    await this.loadHourlyData();

    if (this.hourlyDataLoaded) {
      this.$nextTick(() => this.initCharts());
    }
  },

  methods: {
    getConfigTabComponent(tab) {
      switch(tab) {
        case 'appliances':
          return AppliancesConfigComponent;
        case 'power':
          return PowerConfigComponent;
        case 'settings':
          return SettingsConfigComponent;
        default:
          return {
            template: '<div>Unknown tab</div>'
          };
      }
    },

    // Group Helper Methods
    getGroupTitle(group) {
      const titles = {
        1: 'Fixed Power Appliances (Run Full Duration)',
        2: 'Variable Power Appliances',
        3: 'Fixed Power Appliances (Toggle Anytime)'
      };
      return titles[group] || 'Unknown Group';
    },

    getGroupColor(group) {
      const colors = {
        1: '#be123c',  // Pink
        2: '#047857',  // Green
        3: '#1d4ed8'   // Blue
      };
      return colors[group] || '#000000';
    },

    // Chart Helper Methods
    getChartTitle(chartType) {
      const titles = {
        solar: 'Solar & Battery (kW)',
        temperature: 'Temperature (°C)',
        price: 'Electricity Price ($/kWh)'
      };
      return titles[chartType] || chartType;
    },

    formatChartValue(chartType, value) {
      if (value === null || value === undefined) return 'N/A';
      
      switch(chartType) {
        case 'solar':
          return `${value} kW`;
        case 'temperature':
          return `${value}°C`;
        case 'price':
          return `$${value}/kWh`;
        default:
          return value;
      }
    },

        // Add this method to your component
    updateBatteryHistory(step, batteryLevel) {
      // Ensure step is within bounds
      const safeStep = step % 96;
      
      // Update battery history at current step
      if (!this.batteryHistory) {
        this.batteryHistory = Array(96).fill(null);
        this.batteryHistory[0] = this.batteryLevel;
      }
      
      // Only store the battery level at the current step
      this.batteryHistory[safeStep] = batteryLevel;
      
      // Now update the chart data from history
      if (this.chartData?.solar?.battery) {
        // First, set the initial value if not already set
        if (this.chartData.solar.battery[0] === null) {
          this.chartData.solar.battery[0] = this.batteryLevel;
        }

        // Just update the specific point that changed
        this.chartData.solar.battery[safeStep] = batteryLevel;

        // This ensures the line connects properly
        let lastDataStep = 0;
        for (let i = 1; i < safeStep; i++) {
          if (this.chartData.solar.battery[i] !== null) {
            lastDataStep = i;
          }
        }

        // Fill gaps with interpolated values if needed
        if (lastDataStep < safeStep - 1 && lastDataStep > 0) {
          const lastValue = this.chartData.solar.battery[lastDataStep];
          const steps = safeStep - lastDataStep;
          const increment = (batteryLevel - lastValue) / steps;
          
          for (let i = 1; i < steps; i++) {
            const fillStep = lastDataStep + i;
            if (this.chartData.solar.battery[fillStep] === null) {
              this.chartData.solar.battery[fillStep] = lastValue + (increment * i);
            }
          }
        }
        
        // Update chart if it exists
        if (this.charts?.solar) {
          this.charts.solar.updateSeries([
            { name: 'Solar', data: this.chartData.solar.primary },
            { name: 'Battery', data: this.chartData.solar.battery }
          ]);
        }
      }
    },

    cleanupCharts() {
      const chartKeys = ['solar', 'airTemp', 'waterTemp', 'price'];
  
      chartKeys.forEach(key => {
        if (this.charts[key]) {
          try {
            this.charts[key].destroy();
          } catch (e) {
            console.warn(`Error destroying ${key} chart:`, e);
          }
          this.charts[key] = null;
        }
      });
    },

    initSolarChart(container, timeLabels, customLabels) {
      const chartConfig = {
        series: [
          { name: 'Solar', data: this.chartData.solar.primary },
          { name: 'Battery', data: this.chartData.solar.battery }
        ],
        chart: {
          height: 160,
          type: 'line',
          animations: {enabled: false},
          toolbar: {show: false},
          zoom: {enabled: false},
          fontFamily: 'Arial, sans-serif',
          background: '#ffffff',
          redrawOnWindowResize: false 
        },
        colors: ['#f59e0b', '#10b981'],
        stroke: {
          curve: 'smooth',
          width: [3, 3],
          lineCap: 'round',
          dashArray: [0, 0]
        },
        markers: {
          size: 0,
          hover: {
            size: 4,
            sizeOffset: 2
          }
        },
        xaxis: {
          categories: timeLabels,
          tickPlacement: 'on',
          min: 0,
          max: 95,
          labels: {
            formatter: value => customLabels[value] || ''
          }
        },
        yaxis: [
          { 
            min: 0, 
            max: 6, 
            tickAmount: 6, 
            title: { text: 'Solar Output (kW)' },
            labels: {
              formatter: val => val === null || val === undefined ? '' : val.toFixed(1)
            }
          },
          { 
            min: 0, 
            max: 100, 
            tickAmount: 5, 
            opposite: true, 
            title: { text: 'Battery Level (%)' },
            labels: {
              formatter: val => val === null || val === undefined ? '' : `${val.toFixed(0)}%`
            }
          }
        ],
        tooltip: {
          enabled: true,
          shared: true,
          x: {
            formatter: value => {
              const hour = Math.floor((value * 15) / 60);
              const minute = (value * 15) % 60;
              return `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
            }
          },
          y: {
            formatter: function(value, { seriesIndex }) {
              if (seriesIndex === 0) {
                return value === null || value === undefined ? 'N/A' : `${value.toFixed(1)} kW`;
              } else {
                return value === null || value === undefined ? 'N/A' : `${value.toFixed(0)}%`;
              }
            }
          }
        },
        dataLabels: { enabled: false },
        grid: { show: false },
        legend: { show: true }
      };
      
      this.charts.solar = new window.ApexCharts(container, chartConfig);
      this.charts.solar.render();
    },
    initAirTempChart(container, timeLabels, customLabels) {
      const chartConfig = {
        series: [
          { name: 'Outdoor', data: this.chartData.temperature.primary },
          { name: 'Indoor', data: this.chartData.temperature.indoor }
        ],
        chart: {
          height: 160,
          type: 'line',
          animations: {enabled: false},
          toolbar: {show: false},
          zoom: {enabled: false},
          fontFamily: 'Arial, sans-serif',
          background: '#ffffff',
          redrawOnWindowResize: false 
        },
        colors: ['#ef4444', '#3b82f6'],
        stroke: {
          curve: 'smooth',
          width: [2, 3], // Make indoor line thicker
          lineCap: 'round',
          dashArray: [0, 0]
        },
        markers: {
          size: 0, // Only show markers for indoor temperature
          hover: {
            size: 4,
            sizeOffset: 2
          }
        },
        xaxis: {
          categories: timeLabels,
          tickPlacement: 'on',
          min: 0,
          max: 95,
          labels: {
            formatter: value => customLabels[value] || ''
          }
        },
        yaxis: {
          min: 0, 
          max: 30, 
          tickAmount: 5, 
          title: { text: 'Air Temperature (°C)' },
          labels: {
            formatter: val => val === null || val === undefined || isNaN(val) ? '' : val.toFixed(1)
          }
        },
        tooltip: {
          enabled: true,
          shared: true,
          x: {
            formatter: value => {
              const hour = Math.floor((value * 15) / 60);
              const minute = (value * 15) % 60;
              return `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
            }
          },
          y: {
            formatter: value => value === null || value === undefined ? 'N/A' : `${value.toFixed(1)} °C`
          }
        },
        dataLabels: { enabled: false },
        grid: { show: false },
        legend: { show: true },
        annotations: {
          yaxis: [{
            y: 20, // Indoor comfort temperature
            borderColor: 'rgba(4, 120, 87, 0.4)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              // text: 'Comfort Temp',
              position: 'left',
              offsetX: -5,
              style: {
                fontSize: '10px',
                color: '#047857',
                background: '#f8fafc'
              }
            }
          }]
        }
      };
      
      this.charts.airTemp = new window.ApexCharts(container, chartConfig);
      this.charts.airTemp.render();
    },
    initWaterTempChart(container, timeLabels, customLabels) {
      const chartConfig = {
        series: [
          { name: 'Water', data: this.chartData.temperature.water }
        ],
        chart: {
          height: 160,
          type: 'line',
          animations: {enabled: false},
          toolbar: {show: false},
          zoom: {enabled: false},
          fontFamily: 'Arial, sans-serif',
          background: '#ffffff', // Transparent background
          redrawOnWindowResize: false
        },
        colors: ['#f59e0b'], // Water temperature color
        stroke: {
          curve: 'smooth',
          width: 3,
          lineCap: 'round'
        },
        markers: {
          size: 0,
          hover: {
            size: 4,
            sizeOffset: 2
          }
        },
        xaxis: {
          categories: timeLabels,
          tickPlacement: 'on',
          min: 0,
          max: 95,
          labels: {
            formatter: value => customLabels[value] || ''
          }
        },
        yaxis: {
          min: 40, 
          max: 70, 
          tickAmount: 5, 
          // opposite: true, // Right side
          title: { text: 'Water Temperature (°C)' },
          labels: {
            formatter: val => val === null || val === undefined || isNaN(val) ? '' : val.toFixed(1)
          }
        },
        tooltip: {
          enabled: true,
          shared: true,
          x: {
            formatter: value => {
              const hour = Math.floor((value * 15) / 60);
              const minute = (value * 15) % 60;
              return `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
            }
          },
          y: {
            formatter: value => value === null || value === undefined ? 'N/A' : `${value.toFixed(1)} °C`
          }
        },
        dataLabels: { enabled: false },
        grid: { show: false },
        legend: { show: true }, // Hide legend for water chart
        annotations: {
          yaxis: [{
            y: 60, // Water comfort temperature
            borderColor: 'rgba(245, 158, 11, 0.4)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              // text: 'Comfort Temp',
              position: 'left',
              offsetX: -5,
              style: {
                fontSize: '10px',
                color: '#f59e0b',
                background: '#f8fafc'
              }
            }
          }]
        }
      };
      
      this.charts.waterTemp = new window.ApexCharts(container, chartConfig);
      this.charts.waterTemp.render();
    },
    initPriceChart(container, timeLabels, customLabels) {
      const chartConfig = {
        series: [{
          name: 'Price',
          data: this.chartData.price.primary
        }],
        chart: {
          height: 160,
          type: 'line',
          animations: {enabled: false},
          toolbar: {show: false},
          zoom: {enabled: false},
          fontFamily: 'Arial, sans-serif',
          background: '#ffffff',
          redrawOnWindowResize: false 
        },
        colors: ['#8b5cf6'],
        stroke: {
          curve: 'smooth',
          width: 3,
          lineCap: 'round'
        },
        markers: {
          size: 0,
          hover: {
            size: 4,
            sizeOffset: 2
          }
        },
        xaxis: {
          categories: timeLabels,
          tickPlacement: 'on',
          min: 0,
          max: 95,
          labels: {
            formatter: value => customLabels[value] || ''
          }
        },
        yaxis: {
          min: 0, 
          max: 0.1, 
          tickAmount: 5,
          labels: {
            formatter: val => val === null || val === undefined || isNaN(val) ? '' : '$' + val.toFixed(3)
          }
        },
        tooltip: {
          enabled: true,
          shared: true,
          x: {
            formatter: value => {
              const hour = Math.floor((value * 15) / 60);
              const minute = (value * 15) % 60;
              return `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
            }
          },
          y: {
            formatter: value => value === null || value === undefined ? 'N/A' : `$${value.toFixed(3)}/kWh`
          }
        },
        dataLabels: { enabled: false },
        grid: { show: false },
        legend: { show: false }
      };
      
      this.charts.price = new window.ApexCharts(container, chartConfig);
      this.charts.price.render();
    },

    // Chart Methods
    initCharts() {
      // Initialize data structure just once
      if (!this.chartData) {
        this.chartData = {
          solar: {
            primary: Array(96).fill(null),
            battery: Array(96).fill(null)
          },
          temperature: { 
            primary: Array(96).fill(null),
            indoor: Array(96).fill(null),
            water: Array(96).fill(null)
          },
          price: { primary: Array(96).fill(null) }
        };
        
        // Create battery history array (just once)
        this.batteryHistory = Array(96).fill(null);
        this.batteryHistory[0] = this.batteryLevel; // Set initial value at time 0
        this.chartData.solar.battery[0] = this.batteryLevel; // Also set in chart data
        this.chartData.temperature.indoor[0] = this.energyModels.indoorTemperature
        this.chartData.temperature.water[0] = this.energyModels.waterTemperature;
      }

      // Clear previous attempt timeouts
      if (this.chartInitTimeout) {
        clearTimeout(this.chartInitTimeout);
        this.chartInitTimeout = null;
      }
      
      if (typeof window.ApexCharts === 'undefined') {
        console.error('ApexCharts not available');
        return;
      }

      // Make sure chart containers exist
      const containers = [
        document.getElementById('solar-chart-container'),
        document.getElementById('air-temp-chart-container'), // New container
        document.getElementById('water-temp-chart-container'), // New container
        document.getElementById('price-chart-container')
      ];

      if (containers.some(el => !el)) {
        // console.log('Chart containers not ready, retrying in 100ms');
        // Store timeout ID so it can be cleared if component is destroyed
        this.chartInitTimeout = setTimeout(() => this.initCharts(), 100);
        return;
      }

      // Safely destroy any existing charts first
      this.cleanupCharts();
      
      try {
        const timeLabels = Array(96).fill(0).map((_, i) => i);
        const customLabels = {
          0: '0',
          16: '4',
          32: '8', 
          48: '12',
          64: '16',
          80: '20'
        };

        this.initSolarChart(containers[0], timeLabels, customLabels);
        this.initAirTempChart(containers[1], timeLabels, customLabels);
        this.initWaterTempChart(containers[2], timeLabels, customLabels);
        this.initPriceChart(containers[3], timeLabels, customLabels);
        
        console.log('Charts initialized successfully');
      } catch (err) {
        console.error('Error initializing charts:', err);
      }
    },

    // Helper method for updating chart markers
    updateChartMarkers(chart, step, colors) {
      try {
        const discreteMarkers = [];
        
        // Create a marker for each series
        for (let i = 0; i < chart.w.config.series.length; i++) {
          discreteMarkers.push({
            seriesIndex: i,
            dataPointIndex: step,
            size: 4,
            strokeColor: colors[i] || colors[0],
            fillColor: '#ffffff'
          });
        }
        
        chart.updateOptions({
          markers: {
            discrete: discreteMarkers
          }
        }, false, false);
      } catch (error) {
        console.warn('Error updating chart markers:', error);
      }
    },

    updateChartDataPoint(chartName, step, value, newVal = null, waterTemp = null) {
      if (!this.charts[chartName]) return;
      
      // Update primary data directly
      if (this.chartData[chartName]) {
        this.chartData[chartName].primary[step] = value;
      }

      // For battery data, use the dedicated update method
      if (chartName === 'solar' && newVal !== null) {
        this.chartData[chartName].battery[step] = newVal;

        // Update chart if it exists
        if (this.charts.solar) {
          this.charts.solar.updateSeries([
            { name: 'Solar', data: this.chartData[chartName].primary },
            { name: 'Battery', data: this.chartData[chartName].battery }
          ], false, false);
        }
      } else if (chartName === 'temperature') {
        if (newVal !== null) {
          this.chartData[chartName].indoor[step] = newVal;
        }

        if (waterTemp !== null) {
          this.chartData[chartName].water[step] = waterTemp;
        }

        // Update air temperature chart
        if (this.charts.airTemp) {
          this.charts.airTemp.updateSeries([
            { name: 'Outdoor', data: this.chartData[chartName].primary },
            { name: 'Indoor', data: this.chartData[chartName].indoor }
          ], false, false);
        }
        
        // Update water temperature chart
        if (this.charts.waterTemp) {
          this.charts.waterTemp.updateSeries([
            { name: 'Water', data: this.chartData[chartName].water }
          ], false, false);
        }
      } else if (chartName === 'price') {
        // Update price chart
        if (this.charts.price) {
          this.charts.price.updateSeries([{
            name: 'Price',
            data: this.chartData[chartName].primary
          }], false, false);
        }
      }
      
      try {
        if (step % 4 === 0) { // Only add markers every hour
          if (chartName === 'solar' && this.charts.solar) {
            this.updateChartMarkers(this.charts.solar, step, ['#f59e0b', '#10b981']);
          } 
          else if (chartName === 'temperature') {
            if (this.charts.airTemp) {
              this.updateChartMarkers(this.charts.airTemp, step, ['#ef4444', '#3b82f6']);
            }
            if (this.charts.waterTemp) {
              this.updateChartMarkers(this.charts.waterTemp, step, ['#f59e0b']);
            }
          }
          else if (chartName === 'price' && this.charts.price) {
            this.updateChartMarkers(this.charts.price, step, ['#8b5cf6']);
          }
        }
      } catch (error) {
        console.warn(`Error updating ${chartName} chart:`, error);
      }
    },

    updateCharts(step) {
      if (!this.hourlyDataLoaded || !this.isRunning) return;

      const chartsInitialized = this.charts.solar && 
                            this.charts.airTemp && 
                            this.charts.waterTemp && 
                            this.charts.price;
      if (!chartsInitialized) {
        // Charts not ready, try to initialize them
        this.initCharts();
        return;
      }

      // Get current battery level from state
      const currentBatteryLevel = this.batteryLevel;
      const currentIndoorTemp = this.energyModels.indoorTemperature;
      const currentWaterTemp = this.energyModels.waterTemperature; 

      // Only update if current step is different from last updated
      if (step === this.lastUpdatedStep) return;
      this.lastUpdatedStep = step;

      // Get hourly data - more efficient check
      const hourIndex = Math.floor(step / 4) % (this.hourlyData?.length || 1);
      const hourData = this.hourlyData?.[hourIndex];
      if (!hourData) return;

      // Update current hour data
      this.currentHourData = {
        hour: hourData.hour || '99:99',
        solar: hourData.solar || 0,
        temperature: hourData.temperature || 99,
        price: hourData.price || -0.001
      };

      // Update all chart types
      this.updateSolarChart(step, hourData.solar, currentBatteryLevel);
      this.updateTemperatureCharts(step, hourData.temperature, currentIndoorTemp, currentWaterTemp);
      this.updatePriceChart(step, hourData.price);

      // Update battery history with current value
      this.updateBatteryHistory(step, currentBatteryLevel);
    },

    // Helper methods for updating specific chart types
    updateSolarChart(step, solarValue, batteryLevel) {
      const hourStart = Math.floor(step / 4) * 4;
      
      // Update points in the current hour
      for (let i = 0; i < 4; i++) {
        const stepIndex = (hourStart + i) % 96;
        
        if (stepIndex === step % 96) {
          this.updateChartDataPoint('solar', stepIndex, solarValue, batteryLevel);
        } else {
          this.updateChartDataPoint('solar', stepIndex, solarValue);
        }
      }
    },

    updateTemperatureCharts(step, outdoorTemp, indoorTemp, waterTemp) {
      const hourStart = Math.floor(step / 4) * 4;
      
      // Update points in the current hour
      for (let i = 0; i < 4; i++) {
        const stepIndex = (hourStart + i) % 96;
        this.chartData.temperature.primary[stepIndex] = outdoorTemp;

        this.updateChartDataPoint('temperature', stepIndex, outdoorTemp, indoorTemp, waterTemp);
      }

      // Explicitly update chart markers for better visualization
      if (this.charts.airTemp) {
        this.charts.airTemp.updateSeries([
          { name: 'Outdoor', data: this.chartData.temperature.primary },
          { name: 'Indoor', data: this.chartData.temperature.indoor }
        ], false, false);
        
        // Add marker for current step
        if (step % 4 === 0) {
          this.updateChartMarkers(this.charts.airTemp, step, ['#ef4444', '#3b82f6']);
        }
      }

      if (this.charts.waterTemp) {
        this.charts.waterTemp.updateSeries([
          { name: 'Water', data: this.chartData.temperature.water }
        ], false, false);
        
        // Add marker for current step
        if (step % 4 === 0) {
          this.updateChartMarkers(this.charts.waterTemp, step, ['#f59e0b']);
        }
      }
    },

    updatePriceChart(step, priceValue) {
      const hourStart = Math.floor(step / 4) * 4;
      
      // Update points in the current hour
      for (let i = 0; i < 4; i++) {
        const stepIndex = (hourStart + i) % 96;
        this.updateChartDataPoint('price', stepIndex, priceValue);
      }
    },

    // Format timestep (0-95) to hour:minute format
    formatTime(step) {
      if (step === undefined || step === null) return "00:00";
      
      const totalMinutes = step * 15;
      const hours = Math.floor(totalMinutes / 60);
      const minutes = totalMinutes % 60;
      
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    },

    formatAppName(key) {
      return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase());
    },

    // Icon/SVG Methods
    getIconSvg(type) {
      const icons = {
        solar: `<g>
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </g>`,
        battery: `<g>
          <rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect>
          <line x1="23" y1="13" x2="23" y2="11"></line>
        </g>`,
        grid: `<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>`,
        house: `<g>
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
          <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </g>`,
      };
      
      return icons[type] || icons.house;
    },
    
    getApplianceSvg(type) {
      const icons = {
        hvac: `<g>
          <rect x="2" y="3" width="20" height="18" rx="2" ry="2"></rect>
          <line x1="7" y1="3" x2="7" y2="21"></line>
          <line x1="17" y1="3" x2="17" y2="21"></line>
          <line x1="2" y1="9" x2="22" y2="9"></line>
          <line x1="2" y1="15" x2="22" y2="15"></line>
        </g>`,
        refrigerator: `<g>
          <rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect>
          <path d="M4 13h16"></path>
          <path d="M8 2v2"></path>
          <path d="M8 13v2"></path>
        </g>`,
        lights: `<g>
          <circle cx="12" cy="12" r="5"></circle>
          <path d="M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"></path>
        </g>`,
        ev_charger: `<g>
          <rect x="6" y="11" width="12" height="10" rx="2" ry="2"></rect>
          <path d="M12 2v9"></path>
          <path d="M9 5h6"></path>
          <path d="M9 16h6"></path>
        </g>`,
        dishwasher: `<g>
          <rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect>
          <path d="M6 4v16"></path>
          <path d="M18 4v16"></path>
          <path d="M2 10h4"></path>
          <path d="M2 14h4"></path>
          <path d="M18 10h4"></path>
          <path d="M18 14h4"></path>
        </g>`,
        water_heater: `<g>
          <rect x="6" y="3" width="12" height="18" rx="2" ry="2"></rect>
          <path d="M12 7v10"></path>
          <path d="M10 13h4"></path>
        </g>`,
        wash_machine: `<g>
          <rect x="2" y="2" width="20" height="20" rx="2" ry="2"></rect>
          <circle cx="12" cy="12" r="6"></circle>
          <circle cx="12" cy="12" r="2"></circle>
        </g>`,
        clothes_dryer: `<g>
          <rect x="2" y="2" width="20" height="20" rx="2" ry="2"></rect>
          <circle cx="12" cy="12" r="6"></circle>
          <line x1="12" y1="6" x2="12" y2="18"></line>
          <line x1="6" y1="12" x2="18" y2="12"></line>
        </g>`,
        tv: `<g>
          <rect x="2" y="7" width="20" height="15" rx="2" ry="2"></rect>
          <polyline points="17 2 12 7 7 2"></polyline>
        </g>`,
        vacuum: `<g>
          <circle cx="12" cy="12" r="9"></circle>
          <path d="M7 15h10"></path>
          <path d="M12 6v6"></path>
         </g>`,
        hair_dryer: `<g>
          <path d="M10 6h4v14a2 2 0 01-2 2 2 2 0 01-2-2V6z"></path>
          <path d="M6 6h8a0 0 0 011 1v3a0 0 0 01-1 1H6a2 2 0 01-2-2V8a2 2 0 012-2z"></path>
          <path d="M18 4v6"></path>
          <path d="M18 6l-4-2"></path>
          <path d="M18 8l-4 2"></path>
        </g>`
      };
      
      return icons[type] || icons.hvac;
    },

    getApplianceIcon(type) {
      if (this.customIcons[type]) {
        return { type: 'image', path: this.customIcons[type] };
      }
      return { type: 'component', component: type };
    },

    onSpeedChange() {
      const speedOption = this.speedOptions.find(option => option.value === this.selectedSpeed);
      this.updateAnimationSpeed(speedOption.value);
      const timeScale = speedOption.value * 30;
      
      this.$emit('speed-changed', {
        animationSpeed: speedOption.value,
        interval: speedOption.interval,
        timeScale: timeScale
      });

      if (this.selectedSpeed === 3) {
        ChartManager.optimizeMemoryUsage();
      }
    },

    updateAnimationSpeed(speedValue) {
      const duration = 1 / speedValue;
      document.documentElement.style.setProperty('--animation-duration', `${duration}s`);
      
      const flowLines = document.querySelectorAll('.flow-right, .flow-left, .flow-down, .flow-up');
      flowLines.forEach(line => {
        line.style.setProperty('--animation-duration', `${duration}s`);
      });
    },
    
    // Configuration Methods
    async loadDefaultConfig() {
      try {
        const config = await loadConfig('/appliances-config.json').catch(() => getDefaultConfig());
        this.initializeFromConfig(config);
      } catch (error) {
        console.error('Error loading default configuration:', error);
        const defaultConfig = getDefaultConfig();
        this.initializeFromConfig(defaultConfig);
      }
    },
    
    initializeFromConfig(config) {
      this.editableConfig = JSON.parse(JSON.stringify(config));
      this.simulatorAppliances = processApplianceConfig(config);
      const powerSourcesConfig = processPowerSourcesConfig(config);
      this.initializePowerSources(powerSourcesConfig);
      this.initializeConnections();
    },
    
    initializePowerSources(sourcesConfig) {
      this.powerSources = [
        {
          id: 'solar',
          name: sourcesConfig.solar?.name || 'Solar Panel',
          type: 'solar',
          capacity: sourcesConfig.solar?.capacity || 5,
          efficiency: sourcesConfig.solar?.efficiency || 0.85,
          active: this.solarOutput > 0,
          x: this.sourcesLayout.solar.x,
          y: this.sourcesLayout.solar.y,
          rectX: -40, rectY: -40, width: 80, height: 80, rx: 5,
          iconX: -20, iconY: -20, iconWidth: 40, iconHeight: 40,
          labelX: 0, labelY: 60, valueX: 0, valueY: 80,
          customIcon: null,
          bgFill: '#fff9e6', stroke: '#f59e0b',
          displayValue: `${this.solarOutput} kW`
        },
        {
          id: 'battery',
          name: sourcesConfig.battery?.name || 'Battery',
          type: 'battery',
          capacity: sourcesConfig.battery?.capacity || 13.5,
          active: this.batteryStatus !== 'empty',
          x: this.sourcesLayout.battery.x,
          y: this.sourcesLayout.battery.y,
          rectX: -40, rectY: -40, width: 80, height: 80, rx: 5,
          iconX: -20, iconY: -20, iconWidth: 40, iconHeight: 40,
          labelX: 0, labelY: 60, valueX: 0, valueY: 80,
          customIcon: null,
          bgFill: '#f0fdf4', stroke: '#10b981',
          displayValue: `${this.batteryLevel.toFixed(1)}%`
        },
        {
          id: 'grid',
          name: sourcesConfig.grid?.name || 'Power Grid',
          type: 'grid',
          maxPower: sourcesConfig.grid?.maxPower || 10,
          exportEnabled: sourcesConfig.grid?.exportEnabled || true,
          active: this.gridPower > 0,
          x: this.sourcesLayout.grid.x,
          y: this.sourcesLayout.grid.y,
          rectX: -40, rectY: -40, width: 80, height: 80, rx: 5,
          iconX: -20, iconY: -20, iconWidth: 40, iconHeight: 40,
          labelX: 0, labelY: 60, valueX: 0, valueY: 80,
          customIcon: null,
          bgFill: '#f5f3ff', stroke: '#8b5cf6',
          displayValue: `${this.gridPower} kW`
        },
        {
          id: 'house',
          name: sourcesConfig.house?.name || 'Home',
          type: 'house',
          active: true,
          x: this.sourcesLayout.house.x,
          y: this.sourcesLayout.house.y,
          rectX: -60, rectY: -60, width: 120, height: 120, rx: 5,
          iconX: -30, iconY: -30, iconWidth: 60, iconHeight: 60,
          labelX: 0, labelY: 80, valueX: 0, valueY: 100,
          customIcon: null,
          bgFill: '#eff6ff', stroke: '#3b82f6',
          displayValue: `${this.houseDemand} kW`
        }
      ];
    },

    getDefaultConnections() {
      return [
        {
          from: 'solar', to: 'battery',
          x1: 140, y1: 90, x2: 460, y2: 120,
          direction: 'right', color: '#f59e0b'
        },
        {
          from: 'solar', to: 'house',
          x1: 140, y1: 90, x2: 450, y2: 230,
          direction: 'right', color: '#f59e0b'
        },
        {
          from: 'solar', to: 'grid',
          x1: 140, y1: 60, x2: 860, y2: 60,
          direction: 'right', color: '#f59e0b'
        },
        {
          from: 'battery', to: 'house',
          x1: 500, y1: 120, x2: 500, y2: 230,
          direction: 'down', color: '#10b981'
        },
        {
          from: 'grid', to: 'house',
          x1: 860, y1: 90, x2: 560, y2: 230,
          direction: 'right', color: '#8b5cf6'
        },
        {
          from: 'grid', to: 'battery',
          x1: 860, y1: 90, x2: 540, y2: 120,
          direction: 'right', color: '#8b5cf6'
        }
      ];
    },
    
    initializeConnections() {
      this.connections = this.defaultConnections.map(conn => ({
        ...conn,
        active: false,
        strokeWidth: 2,
        powerFlow: 0,
        labelX: (conn.x1 + conn.x2) / 2,
        labelY: ((conn.y1 + conn.y2) / 2) - 10
      }));
      
      this.updateConnectionsStatus();
    },
    
    updateConnectionsStatus() {
      this.validateAppliancePower();

      this.connections.forEach(conn => {
        conn.active = false;
        conn.powerFlow = 0;
        conn.strokeWidth = 2;
      });

      const calculatedHouseDemand = this.recalculateHouseDemand();

      if (Math.abs(calculatedHouseDemand - this.houseDemand) > 0.1) {
        this.$emit('demand-updated', calculatedHouseDemand);
      }

      const BATTERY_MIN_LEVEL = 10.0;
      const MIN_FLOW_THRESHOLD = 0.099;

      const effectiveHouseDemand = calculatedHouseDemand;
      let totalGridPower = 0; // Add this to track total grid power

      if (this.solarOutput > MIN_FLOW_THRESHOLD && effectiveHouseDemand > MIN_FLOW_THRESHOLD) {
        const solarToHouse = Math.min(this.solarOutput, effectiveHouseDemand);
        if (solarToHouse > MIN_FLOW_THRESHOLD) {
          this.activateConnection('solar', 'house', solarToHouse);
        }
      }
      
      const remainingDemand = Math.max(0, effectiveHouseDemand - Math.min(this.solarOutput, effectiveHouseDemand));
      
      let batteryToHouse = 0;
      const effectiveMinLevel = BATTERY_MIN_LEVEL + 
                              (this.batteryLevel < BATTERY_MIN_LEVEL + 5 ? 5 : 0);
      if (this.batteryStatus === 'discharging' && this.batteryPower < 0 && 
          this.batteryLevel > effectiveMinLevel && remainingDemand > MIN_FLOW_THRESHOLD) {
        batteryToHouse = Math.min(remainingDemand, Math.abs(this.batteryPower));
        if (batteryToHouse > MIN_FLOW_THRESHOLD) {
          this.activateConnection('battery', 'house', batteryToHouse);
        }
      }

      const gridToHouse = Math.max(0, remainingDemand - batteryToHouse);
      if (gridToHouse > MIN_FLOW_THRESHOLD) {
        this.activateConnection('grid', 'house', gridToHouse);
        totalGridPower += gridToHouse;
      } 

      const excessSolar = Math.max(0, this.solarOutput - Math.min(this.solarOutput, effectiveHouseDemand));
      
      if (excessSolar > MIN_FLOW_THRESHOLD && this.batteryStatus === 'charging' && 
          this.batteryLevel < 100 && this.batteryPower > 0) {
        const solarToBattery = Math.min(excessSolar, this.batteryPower);
        if (solarToBattery > MIN_FLOW_THRESHOLD) {
          this.activateConnection('solar', 'battery', solarToBattery);
        }
      }
      
      const remainingExcessSolar = Math.max(0, excessSolar - 
        (this.batteryStatus === 'charging' ? Math.min(excessSolar, this.batteryPower) : 0));
      
      if (remainingExcessSolar > MIN_FLOW_THRESHOLD) {
        this.activateConnection('solar', 'grid', remainingExcessSolar);
      }

      if (this.batteryStatus === 'charging' && this.batteryPower > 0 && excessSolar < MIN_FLOW_THRESHOLD) {
        const gridToBattery = this.batteryPower;
        if (gridToBattery > MIN_FLOW_THRESHOLD) {
          this.activateConnection('grid', 'battery', gridToBattery);
          totalGridPower += gridToBattery;
        }
      }

      // Update total grid power 
      if (Math.abs(totalGridPower - this.gridPower) > 0.1) {
        this.$emit('grid-power-updated', parseFloat(totalGridPower.toFixed(1)));
      }

      this.activeConnections = [...this.connections];

      if (this.batteryLevel <= BATTERY_MIN_LEVEL && this.batteryStatus === 'discharging') {
        this.$emit('battery-critical');
      }
    },

    activateConnection(from, to, powerFlow) {
      const conn = this.connections.find(c => c.from === from && c.to === to);
      if (conn) {
        conn.active = true;
        conn.powerFlow = parseFloat(powerFlow.toFixed(1));
        conn.strokeWidth = Math.max(2, Math.min(8, powerFlow * 1.5));
      }
    },

    validateAppliancePower() {
      let updated = false;
      
      const validatedAppliances = this.simulatorAppliances.map(app => {
        if (app.active && (app.power === 0 || isNaN(parseFloat(app.power)))) {
          updated = true;
          //console.warn(`Fixing zero power for active appliance: ${app.name}`);
          
          const defaultPower = app.defaultPower || this.getDefaultPowerForType(app.type);
          
          return {
            ...app,
            power: defaultPower
          };
        }
        return app;
      });
      
      if (updated) {
        this.simulatorAppliances = validatedAppliances;
        this.recalculateHouseDemand();
      }
      
      return updated;
    },

    getDefaultPowerForType(type) {
      const defaults = {
        'dishwasher': 1.2,
        'wash_machine': 0.9,
        'clothes_dryer': 3.0,
        'hvac': 2.0,
        'water_heater': 3.0,
        'ev_charger': 7.0,
        'tv': 0.2,
        'refrigerator': 0.8,
        'lights': 0.5,
        'vacuum': 1.4,
        'hair_dryer': 1.8
      };
      return defaults[type] || 1.0;
    },

    updatePowerSourcesFromProps() {
      const solar = this.powerSources.find(s => s.id === 'solar');
      if (solar) {
        solar.active = this.solarOutput > 0;
        solar.displayValue = `${this.solarOutput} kW`;
      }
      
      const battery = this.powerSources.find(s => s.id === 'battery');
      if (battery) {
        battery.active = this.batteryStatus !== 'empty';
        battery.displayValue = `${this.batteryLevel.toFixed(1)}%`;
      }
      
      const grid = this.powerSources.find(s => s.id === 'grid');
      if (grid) {
        grid.active = this.gridPower > 0;
        grid.displayValue = `${this.gridPower} kW`;
      }
      
      const house = this.powerSources.find(s => s.id === 'house');
      if (house) {
        house.displayValue = `${this.houseDemand} kW`;
      }
    },
    
    // Simulation Control Methods
    startSimulation() {
      this.simulationSteps = 0;
      this.elapsedMinutes = 0;
      this.localSimulationDay = 1;
      this.simulationTime = "00:00";
      this.lastUpdatedStep = -1;

      const timeLimit = this.simulationHours ? this.simulationHours * 3600 : null;

      let maxSteps = null;
      if (this.hourlyDataLoaded && this.hourlyData.length > 0) {
        maxSteps = this.hourlyData.length * 4;
      }

      this.$nextTick(() => {
        if (this.hourlyDataLoaded && !this.charts.solar) {
          this.initCharts();
        }
      });

      this.$emit('start-simulation', {
        timeLimit: timeLimit,
        maxSteps: maxSteps
      });

      this.chartUpdateTimer = setInterval(() => {
        if (this.isRunning && this.hourlyDataLoaded) {
          const currentStep = this.simulationSteps % 96;
          this.updateCharts(currentStep);
        }
      }, 2000);
    },
    
    pauseSimulation() {
      if (this.chartUpdateTimer) {
        clearInterval(this.chartUpdateTimer);
        this.chartUpdateTimer = null;
      }

      this.$emit('pause-simulation');
    },
    
    resumeSimulation() {
      this.$emit('resume-simulation');
    },
    
    resetSimulation() {
      if (this.chartUpdateTimer) {
        clearInterval(this.chartUpdateTimer);
        this.chartUpdateTimer = null;
      }

      this.simulationSteps = 0;
      this.simulationTime = "00:00";
      this.localSimulationDay = 1;
      this.elapsedMinutes = 0;
      this.lastUpdatedStep = -1;

      this.resetCharts();
      this.resetCounter++;

      // Reset all connections and recalculate energy flows
      this.updateConnectionsStatus();

      // Emit events to update parent component values
      this.$emit('solar-output-updated', 0);
      this.$emit('battery-level-updated', 50);
      this.$emit('battery-status-updated', 'empty');
      this.$emit('grid-power-updated', 0);
      this.$emit('demand-updated', this.recalculateHouseDemand());

      this.$emit('reset-simulation');
      ChartManager.optimizeMemoryUsage();
    },

    // Appliance Control Methods
    toggleApplianceStatus(id) {
      const appIndex = this.simulatorAppliances.findIndex(app => app.id === id);
  
      if (appIndex !== -1) {
        const updatedAppliance = {
          ...this.simulatorAppliances[appIndex],
          active: !this.simulatorAppliances[appIndex].active
        };
    
        this.simulatorAppliances = [
          ...this.simulatorAppliances.slice(0, appIndex),
          updatedAppliance,
          ...this.simulatorAppliances.slice(appIndex + 1)
        ];

        const newHouseDemand = parseFloat(
          this.simulatorAppliances
            .filter(app => app.active)
            .reduce((sum, app) => sum + (parseFloat(app.power) || 0), 0)
            .toFixed(1)
        );
    
        this.$emit('demand-updated', newHouseDemand);
        this.$emit('toggle-appliance', id);
      } else {
        console.warn(`Appliance with id ${id} not found`);
      }

      this.updateConnectionsStatus();
    },
    
    // Config Editor Methods
    loadConfigFile() {
      if (this.$refs.configFileInput) {
        this.$refs.configFileInput.click();
      }
    },
    
    onConfigFileSelected(event) {
      const file = event.target.files[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const config = JSON.parse(e.target.result);
          if (!config.appliances || !config.powerSources) {
            throw new Error("Invalid configuration format: missing appliances or powerSources");
          }
          this.initializeFromConfig(config);
        } catch (error) {
          console.error('Error parsing configuration file:', error);
          alert('Invalid configuration file format. Please upload a valid JSON file.');
        }
      };
      reader.readAsText(file);
    },
    
    addNewAppliance() {
      const newId = Math.max(...this.editableConfig.appliances.map(a => a.id), 0) + 1;
      this.editableConfig.appliances.push({
        id: newId,
        name: 'New Appliance',
        type: 'other',
        defaultPower: 1.0,
        active: false,
        group: 3,
        connections: ['house']
      });
    },
    
    removeAppliance(id) {
      this.editableConfig.appliances = this.editableConfig.appliances.filter(a => a.id !== id);
    },
    
    saveConfigChanges() {
      const newConfig = JSON.parse(JSON.stringify(this.editableConfig));
  
      const processedAppliances = newConfig.appliances.map(app => ({
        id: app.id,
        name: app.name,
        type: app.type || 'other',
        power: parseFloat(app.defaultPower) || 0,
        defaultPower: parseFloat(app.defaultPower) || 0,
        active: Boolean(app.active),
        group: app.group || 3,
        connections: app.connections || ['house']
      }));
  
      this.simulatorAppliances = processedAppliances;
  
      const powerSourcesConfig = processPowerSourcesConfig(newConfig);
      this.initializePowerSources(powerSourcesConfig);
      this.initializeConnections();
      this.editableConfig = newConfig;
      this.showConfigEditor = false;
  
      this.$emit('config-updated', {
        appliances: this.simulatorAppliances,
      });
    },
    
    exportConfig() {
      const configStr = JSON.stringify(this.editableConfig, null, 2);
      const blob = new Blob([configStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = 'smart-home-config.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    // Config editor component handling
    getConfigTabComponent(tab) {
      // For the refactored version, these would be separate SFC components
      // For now, we'll use inline templates
      switch(tab) {
        case 'appliances':
          return {
            props: ['config'],
            template: `
              <div class="appliances-config">
                <div v-for="app in config.appliances" :key="'edit-' + app.id" class="config-item">
                  <div class="config-item-header">
                    <h4>{{ app.name }}</h4>
                    <button @click="$emit('update', { action: 'remove', id: app.id })" class="remove-btn">Remove</button>
                  </div>
                  <div class="config-fields">
                    <div class="field">
                      <label>Name:</label>
                      <input type="text" v-model="app.name">
                    </div>
                    <!-- remaining fields... -->
                  </div>
                </div>
                <button @click="$emit('update', { action: 'add' })" class="add-btn">Add New Appliance</button>
              </div>
            `
          };
        // Add cases for 'power' and 'settings'
        default:
          return {
            template: '<div>Unknown tab</div>'
          };
      }
    },
    
    updateEditableConfig(event) {
      // Handle configuration updates from child components
      if (event.action === 'add') {
        this.addNewAppliance();
      } else if (event.action === 'remove') {
        this.removeAppliance(event.id);
      }
    },

    // Utility Methods
    safeGet(obj, path, defaultValue = null) {
      try {
        return path.split('.').reduce((o, key) => (o && o[key] !== undefined) ? o[key] : defaultValue, obj);
      } catch (e) {
        return defaultValue;
      }
    },

    updateHistoryData() {
      if (!this.rlPrediction) return;

      try {
        const timeStep = this.safeGet(this.rlPrediction, 'timestamp', 0);
        
        if (timeStep >= 0 && timeStep < 96) {
          const homeTemp = this.safeGet(this.rlPrediction, 'temperatures.home.current', null);
          if (homeTemp !== null) {
            const newHomeHistory = [...this.homeTemperatureHistory];
            newHomeHistory[timeStep] = homeTemp;
            this.homeTemperatureHistory = newHomeHistory;
          }
          
          const waterTemp = this.safeGet(this.rlPrediction, 'temperatures.water.current', null);
          if (waterTemp !== null) {
            const newWaterHistory = [...this.waterTemperatureHistory];
            newWaterHistory[timeStep] = waterTemp;
            this.waterTemperatureHistory = newWaterHistory;
          }
          
          if (this.rlPrediction.appliances && this.rlPrediction.appliances.shiftable) {
            const newAppliancePowerHistory = {};
            
            for (const key in this.appliancePowerHistory) {
              newAppliancePowerHistory[key] = [...this.appliancePowerHistory[key]];
            }
            
            // Continuing from where it was cut off...
            for (const appKey in this.rlPrediction.appliances.shiftable) {
              if (!newAppliancePowerHistory[appKey]) {
                newAppliancePowerHistory[appKey] = new Array(96).fill(0);
              }
              
              const app = this.rlPrediction.appliances.shiftable[appKey];
              if (app) {
                const power = app.active ? app.power : 0;
                newAppliancePowerHistory[appKey][timeStep] = power;
              }
            }
            
            this.appliancePowerHistory = newAppliancePowerHistory;
          }
          
          this.$emit('history-updated', {
            timeStep,
            homeTemp,
            waterTemp
          });
        }
      } catch (err) {
        console.error('Error in updateHistoryData:', err);
      }
    },

    resetCharts() {
      // Destroy existing charts
      for (const chartKey in this.charts) {
        if (this.charts[chartKey]) {
          this.charts[chartKey].destroy();
          this.charts[chartKey] = null;
        }
      }

      // Reset chartData arrays
      this.chartData = {
        solar: { primary: Array(96).fill(null), battery: Array(96).fill(null) },
        temperature: { 
          primary: Array(96).fill(null), 
          indoor: Array(96).fill(null), 
          water: Array(96).fill(null)
        },
        price: { primary: Array(96).fill(null) }
      };
      
      // Re-initialize charts
      this.$nextTick(() => {
        if (this.hourlyDataLoaded) {
          this.initCharts();
        }
      });

      // Clear history data completely
      this.homeTemperatureHistory = Array(96).fill(null);
      this.waterTemperatureHistory = Array(96).fill(null);
      this.appliancePowerHistory = {
        dishwasher: Array(96).fill(0),
        wash_machine: Array(96).fill(0),
        clothes_dryer: Array(96).fill(0)
      };

      this.comfortLineAdded = false;
    },

    async loadHourlyData() {
      try {
        const result = await readHourlyData();
        
        if (result.success && result.hourlyData && result.hourlyData.length > 0) {
          this.hourlyData = result.hourlyData;
          this.hourlyDataLoaded = true;
        
          // Initialize currentHourData with the first hour
          this.currentHourData = {
            hour: this.hourlyData[0].hour || '99:99',
            solar: this.hourlyData[0].solar || 0,
            temperature: this.hourlyData[0].temperature || 99,
            price: this.hourlyData[0].price || -0.001
          };

          return true;
        } else {
          console.error("Failed to load hourly data:", result.error);
          this.hourlyDataLoaded = true;
          return false;
        }
      } catch (error) {
        console.error("Error loading hourly data:", error);
        this.hourlyDataLoaded = true;
        return false;
      }
    },

    handleHourlyDataUpdate(data) {
      if (data.solarOutput !== undefined && this.simulationState === 'manual') {
        this.$emit('solar-output-updated', data.solarOutput);
      }
      
      if (data.energyModels) {
        this.energyModels = {
          ...this.energyModels,
          ...data.energyModels
        };
      }
    },

    updateFromSimulationStep() {
      if (!this.isRunning || !this.hourlyDataLoaded) return;
      
      const currentStep = this.simulationSteps % 96;
      const hourIndex = Math.floor(currentStep / 4) % this.hourlyData.length;
      
      if (this.hourlyData[hourIndex]) {
        this.currentHourData = {
          hour: this.hourlyData[hourIndex].hour,
          solar: this.hourlyData[hourIndex].solar,
          temperature: this.hourlyData[hourIndex].temperature,
          price: this.hourlyData[hourIndex].price
        };
        
        this.updateCharts(currentStep);
        
        if (this.simulationState === 'manual') {
          this.updateSimulationFromHourlyData(this.currentHourData);
        }
      }
    },

    initializeEnergyModels(hourData) {
      if (!hourData) return;
      
      this.energyModels.indoorTemperature = getInitialIndoorTemperature(hourData.temperature);
      this.energyModels.waterTemperature = getInitialWaterTemperature(hourData.hour);
      this.energyModels.evConnected = isEvConnected(hourData.hour);
      this.energyModels.evSoC = getInitialEvSoC();
      
      this.energyModels.hvacHistory = [];
      this.energyModels.waterHeaterHistory = [];
      this.energyModels.evHistory = [];
      
      console.log('Energy models initialized:', this.energyModels);
    },

    // Add this method to your component
    updateAppliancePower(applianceId, newPower) {
      try {
        // Find the appliance index in the array
        const appIndex = this.simulatorAppliances.findIndex(app => app.id === applianceId);
        
        if (appIndex !== -1) {
          // Format the power value
          const formattedPower = parseFloat(newPower.toFixed(1));
      
          // Emit event to parent component (App.vue)
          this.$emit('update-appliance-power', { 
            id: applianceId,
            name: this.simulatorAppliances[appIndex].name,
            type: this.simulatorAppliances[appIndex].type,
            power: formattedPower,
            active: this.simulatorAppliances[appIndex].active
          });
          
          return true;
        }
        
        console.warn(`Appliance with id ${applianceId} not found`);
        return false;
      } catch (error) {
        console.error('Error preparing appliance power update:', error);
        return false;
      }
    },

    updateEnergyModels(hourData) {
      if (!hourData) return;
      
      const hvacAppliance = this.simulatorAppliances.find(app => app.type === 'hvac');
      const waterHeaterAppliance = this.simulatorAppliances.find(app => app.type === 'water_heater');
      const evAppliance = this.simulatorAppliances.find(app => app.type === 'ev_charger');
      
      const hvacResult = calculateHvacPower(
        this.energyModels.indoorTemperature,
        hourData.temperature,
        hvacAppliance ? hvacAppliance.active : false
      );
      
      const waterHeaterResult = calculateWaterHeaterPower(
        this.energyModels.waterTemperature,
        waterHeaterAppliance ? waterHeaterAppliance.active : false,
        hourData.hour
      );
      
      this.energyModels.evConnected = isEvConnected(hourData.hour);
      
      const evResult = calculateEvChargingPower(
        this.energyModels.evSoC,
        this.energyModels.evConnected,
        evAppliance ? evAppliance.active : false,
        hourData.hour,
        hourData.price
      );
      
      this.energyModels.indoorTemperature = hvacResult.nextIndoorTemp;
      this.energyModels.waterTemperature = waterHeaterResult.nextWaterTemp;

      if (this.energyModels.evConnected) {
        this.energyModels.evSoC = evResult.nextSoC;
      }
      
      if (hvacAppliance) {
        hvacAppliance.power = parseFloat(hvacResult.power.toFixed(1));
        this.updateAppliancePower(hvacAppliance.id, hvacResult.power);

        if (hvacResult.power > 0 && !hvacAppliance.active) {
          this.toggleApplianceStatus(hvacAppliance.id);
        } else if (hvacResult.power === 0 && hvacAppliance.active) {
          this.toggleApplianceStatus(hvacAppliance.id);
        }
      }
      
      if (waterHeaterAppliance) {
        waterHeaterAppliance.power = waterHeaterResult.power.toFixed(1);
        this.updateAppliancePower(waterHeaterAppliance.id, waterHeaterResult.power);

        if (waterHeaterResult.power > 0 && !waterHeaterAppliance.active) {
          this.toggleApplianceStatus(waterHeaterAppliance.id);
        } else if (waterHeaterResult.power === 0 && waterHeaterAppliance.active) {
          this.toggleApplianceStatus(waterHeaterAppliance.id);
        }

        if (waterHeaterAppliance.active) {
          console.log("Power need: ", waterHeaterResult.power, " kW")
        }
      }
      
      if (evAppliance) {
        evAppliance.power = evResult.power.toFixed(1);
        if (this.energyModels.evConnected) {
          if (evResult.power > 0 && !evAppliance.active) {
            this.toggleApplianceStatus(evAppliance.id);
          } else if (evResult.power === 0 && evAppliance.active) {
            this.toggleApplianceStatus(evAppliance.id);
          }
        } else if (evAppliance.active) {
          this.toggleApplianceStatus(evAppliance.id);
        }
      }
      
      this.energyModels.hvacHistory.push({
        time: hourData.hour,
        power: hvacResult.power,
        temperature: this.energyModels.indoorTemperature
      });
      
      this.energyModels.waterHeaterHistory.push({
        time: hourData.hour,
        power: waterHeaterResult.power,
        temperature: this.energyModels.waterTemperature
      });
      
      this.energyModels.evHistory.push({
        time: hourData.hour,
        power: evResult.power,
        soc: this.energyModels.evSoC,
        connected: this.energyModels.evConnected
      });
      
      if (this.energyModels.hvacHistory.length > 96) {
        this.energyModels.hvacHistory.shift();
      }
      if (this.energyModels.waterHeaterHistory.length > 96) {
        this.energyModels.waterHeaterHistory.shift();
      }
      if (this.energyModels.evHistory.length > 96) {
        this.energyModels.evHistory.shift();
      }
      
      this.recalculateHouseDemand();
      
      this.$emit('energy-models-updated', {
        indoorTemperature: this.energyModels.indoorTemperature,
        waterTemperature: this.energyModels.waterTemperature,
        evSoC: this.energyModels.evSoC,
        evConnected: this.energyModels.evConnected,
        hvacPower: hvacResult.power,
        waterHeaterPower: waterHeaterResult.power,
        evPower: evResult.power
      });
    },

    recalculateHouseDemand() {
      let totalDemand = 0;
      
      this.simulatorAppliances.forEach(app => {
        if (app.active) {
          const power = parseFloat(app.power || 0);
          if (!isNaN(power)) {
            totalDemand += power;
          }
        }
      });
      
      const newHouseDemand = parseFloat(totalDemand.toFixed(1));
      
      return newHouseDemand;
    },

    updateSimulationFromHourlyData(hourData) {
      if (!hourData) return;

      if (!this.energyModels.hvacHistory.length) {
        this.initializeEnergyModels(hourData);
      }
      
      this.updateEnergyModels(hourData);
      
      this.$emit('solar-output-updated', hourData.solar);

      if (hourData.temperature) {
        const hvacAppliance = this.simulatorAppliances.find(app => app.type === 'hvac');
        
        if (hvacAppliance) {
          const isHot = hourData.temperature > 25;
          const isCold = hourData.temperature < 18;
          
          if ((isHot || isCold) && !hvacAppliance.active) {
            this.toggleApplianceStatus(hvacAppliance.id);
          } else if (!isHot && !isCold && hvacAppliance.active) {
            this.toggleApplianceStatus(hvacAppliance.id);
          }
        }
      }

      if (hourData.price) {
        const evAppliance = this.simulatorAppliances.find(app => app.type === 'ev_charger');
        
        if (evAppliance) {
          const isPriceLow = hourData.price < 0.15;
          const isPriceHigh = hourData.price > 0.20;
          
          const currentHour = parseInt(hourData.hour.split(':')[0]);
          
          const isNightTime = currentHour >= 20 || currentHour <= 7;
          
          if ((isPriceLow || isNightTime) && !evAppliance.active) {
            this.toggleApplianceStatus(evAppliance.id);
          } else if (isPriceHigh && evAppliance.active) {
            this.toggleApplianceStatus(evAppliance.id);
          }
        }
      }
    
      this.$emit('hourly-data-update', {
        solarOutput: hourData.solar,
        temperature: hourData.temperature,
        price: hourData.price,
        hour: hourData.hour,
        simulationStep: this.simulationSteps,
        energyModels: {
          indoorTemperature: this.energyModels.indoorTemperature,
          waterTemperature: this.energyModels.waterTemperature,
          evSoC: this.energyModels.evSoC,
          evConnected: this.energyModels.evConnected
        }
      });

      this.simulationSteps++;
    },

    handleVisibilityChange() {
      if (document.visibilityState === 'visible' && this.isRunning) {
        console.log('Page became visible, refreshing energy flows and power values');
        
        this.updateConnectionsStatus();
      }
    },

    handleBatteryCritical() {
      const BATTERY_MIN_LEVEL = 10.0;
      
      if (this.batteryLevel <= BATTERY_MIN_LEVEL && 
         (this.batteryStatus === 'discharging' || this.batteryPower < 0)) {
        console.log(`Battery protection: Level ${this.batteryLevel.toFixed(1)}% reached minimum threshold`);
        
        this.batteryStatus = 'empty';
        this.batteryPower = 0;
        
        this.$emit('battery-critical', {
          batteryStatus: 'empty',
          batteryPower: 0,
          batteryLevel: this.batteryLevel
        });
        
        this.updateConnectionsStatus();
      }
    }
  },
  
  watch: {
    appliances: {
      handler(newAppliances) {
        if (newAppliances && newAppliances.length > 0) {
          let hasChanges = false;
          
          this.simulatorAppliances = this.simulatorAppliances.map(app => {
            const newApp = newAppliances.find(a => a.id === app.id);
            if (newApp && (newApp.active !== app.active || newApp.power !== app.power)) {
              hasChanges = true;
              return {
                ...app,
                active: newApp.active,
                power: parseFloat(newApp.power || app.power)
              };
            }
            return app;
          });
          
          if (hasChanges) {
            this.updateConnectionsStatus();
          }
        }
      },
      deep: true
    },
    
    powerData: {
      handler() {
        this.updatePowerSourcesFromProps();
        this.updateConnectionsStatus();
      },
      deep: true
    },

    rlPrediction(newValue) {
      if (newValue && typeof newValue.timestamp === 'number') {
        this.updateHistoryData();
      }
    },

    simulationFormattedTime(newTime) {
      this.simulationTime = newTime;

      const [hours, minutes] = newTime.split(':').map(Number);
      const totalMinutes = hours * 60 + minutes;
      
      if (this.isRunning) {
        this.elapsedMinutes = totalMinutes + this.localSimulationDay * 24 * 60;
      }
    },
    
    simulationDay(newDay) {
      this.localSimulationDay = newDay;
    },

    simulationElapsedMinutes: function(newValue) {
      if (!this.isRunning) return;
      
      const calculatedSteps = Math.floor(newValue / 15);
      
      if (Math.abs(calculatedSteps - this.simulationSteps) > 1) {
        this.simulationSteps = calculatedSteps;
        
        this.updateFromSimulationStep();
      }
    },

    simulationSteps(newValue, oldValue) {
      if (newValue !== oldValue && this.isRunning) {
        const currentStep = newValue % 96;
        this.updateCharts(currentStep);
      }
    },

    batteryLevel: {
      handler(newValue, oldValue) {
        // Only update if value actually changed and simulation is running
        if (newValue !== oldValue && this.isRunning && this.hourlyDataLoaded) {
          const currentStep = this.simulationSteps % 96;
          this.updateBatteryHistory(currentStep, newValue);
        }
      }
    }
  },

  mounted() {
    if (typeof window.ApexCharts === 'undefined') {
      console.warn('ApexCharts not found, attempting to load from CDN');
      
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/apexcharts';
      script.async = true;
      script.onload = () => {
        console.log('ApexCharts loaded from CDN');
        this.$nextTick(() => {
          // if (this.hourlyDataLoaded) {
            this.initCharts();
          // }
        });
      };
      script.onerror = (err) => {
        console.error('Failed to load ApexCharts from CDN:', err);
      };
      
      document.head.appendChild(script);
    } else {
      this.initCharts();
    }

    document.documentElement.style.setProperty(
      '--animation-duration', 
      `${1/this.selectedSpeed}s`
    );

    this.$nextTick(() => {
      const speedOption = this.speedOptions.find(option => option.value === this.selectedSpeed);
      if (speedOption) {
        this.updateAnimationSpeed(speedOption.value);
      }
    });

    this.$watch('simulationSteps', (newVal, oldVal) => {
      console.log(`Simulation step changed: ${oldVal} -> ${newVal}`);
      if (this.isRunning && newVal !== oldVal) {
        this.updateCharts(newVal % 96);
      }
    });

    this.$watch('hourlyDataLoaded', (newVal) => {
      if (newVal && !this.charts.solar) {
        console.log('Hourly data loaded, initializing charts');
        this.$nextTick(() => {
          this.initCharts();
        });
      }
    });

    this.$watch('isRunning', (newVal) => {
      console.log(`Simulation running state changed to: ${newVal}`);
      
      if (newVal && this.hourlyDataLoaded && (!this.charts.solar || !this.charts.temperature || !this.charts.price)) {
        console.log('Simulation started but charts not initialized, initializing now');
        this.$nextTick(() => {
          this.initCharts();
        });
      }
    });

    this.chartUpdateInterval = setInterval(() => {
      if (this.isRunning && this.hourlyDataLoaded && this.simulationSteps > 0) {
        const currentStep = this.simulationSteps % 96;
        
        if (Math.abs(currentStep - this.lastUpdatedStep) >= 5) {
          this.updateCharts(currentStep);
        }
      }
    }, 1000);
  
    document.addEventListener('visibilitychange', this.handleVisibilityChange);
  },

  beforeDestroy() {
    if (this.chartUpdateInterval) {
      clearInterval(this.chartUpdateInterval);
    }
    
    // Clear any pending timeouts
    if (this.chartInitTimeout) {
      clearTimeout(this.chartInitTimeout);
      this.chartInitTimeout = null;
    }

    // Clean up charts
    this.cleanupCharts();

    this.$nextTick(() => {
      const apexElements = document.querySelectorAll('.apexcharts-canvas');
      apexElements.forEach(element => {
        if (element && element.parentNode) {
          element.parentNode.removeChild(element);
        }
      });
    });
    
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
  }
};

</script>