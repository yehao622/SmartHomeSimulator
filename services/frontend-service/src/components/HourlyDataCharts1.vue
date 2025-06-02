<!-- HourlyDataCharts.vue with ApexCharts CDN version -->
<template>
  <div class="hourly-charts-container">
    <h3>Environmental Data</h3>
    
    <div v-if="!dataLoaded" class="loading-message">
      <p>{{ loadingMessage }}</p>
      <button @click="retryLoad" class="retry-btn">Retry Loading Data</button>
    </div>
    
    <div v-else class="charts-compact-grid">
      <!-- First row: Solar and Temperature -->
      <div class="chart-container">
        <h4>Solar Irradiance (kW)</h4>
        <div class="chart-wrapper" ref="solarChartContainer"></div>
        <div class="current-value" v-if="currentHourData">
          Current: {{ currentHourData.solar }} kW
        </div>
      </div>
      
      <div class="chart-container">
        <h4>Outdoor Temperature (°C)</h4>
        <div class="chart-wrapper" ref="temperatureChartContainer"></div>
        <div class="current-value" v-if="currentHourData">
          Current: {{ currentHourData.temperature }}°C
        </div>
      </div>
      
      <!-- Second row: Price and Time -->
      <div class="chart-container">
        <h4>Electricity Price ($/kWh)</h4>
        <div class="chart-wrapper" ref="priceChartContainer"></div>
        <div class="current-value" v-if="currentHourData">
          Current: ${{ currentHourData.price }}/kWh
        </div>
      </div>
      
      <div class="chart-container time-info">
        <h4>Current Simulation Time</h4>
        <div class="time-display">
          <div class="time-value">{{ formattedCurrentTime }}</div>
          <div class="time-progress">
            <div class="progress-bar" :style="{ width: `${progressPercentage}%` }"></div>
          </div>
          <div class="day-info">
            <span class="day-label">Day {{ simulationDay }}</span>
            <span class="time-label">{{ isDaytime ? 'Daytime' : 'Nighttime' }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { readHourlyData } from './csvReader.js';
// No import for ApexCharts - using window.ApexCharts from CDN

export default {
  props: {
    simulationStep: {
      type: Number,
      default: 0
    },
    isRunning: {
      type: Boolean,
      default: false
    },
    simulationState: {
      type: String,
      default: 'idle'
    }
  },
  
  data() {
    return {
      dataLoaded: false,
      hourlyData: [],
      loadingMessage: 'Loading hourly data...',
      currentHourData: {
        hour: "00:00",
        solar: 0,
        temperature: 20,
        price: 0.01
      },
      charts: {
        solar: null,
        temperature: null,
        price: null
      },
      chartOptions: {
        solar: null,
        temperature: null,
        price: null
      },
      lastUpdatedStep: -1,
      dayData: {
        current: [],
        previous: []
      },
      chartColors: {
        solar: {
          primary: '#f59e0b',
          secondary: 'rgba(245, 158, 11, 0.1)'
        },
        temperature: {
          primary: '#ef4444',
          secondary: 'rgba(239, 68, 68, 0.1)'
        },
        price: {
          primary: '#8b5cf6',
          secondary: 'rgba(139, 92, 246, 0.1)'
        }
      },
      updateLocked: false,
      chartUpdateQueue: [],
      updateTimeout: null
    };
  },

  computed: {
    formattedCurrentTime() {
      return this.formatSimulationTime(this.simulationStep % 96);
    },
    
    progressPercentage() {
      return ((this.simulationStep % 96) / 96) * 100;
    },
    
    simulationDay() {
      // Calculate day based on simulation steps
      return Math.floor(this.simulationStep / 96) + 1;
    },
    
    isDaytime() {
      const currentHour = parseInt(this.formattedCurrentTime.split(':')[0]);
      return currentHour >= 6 && currentHour < 20; // 6am to 8pm
    },
    
    // Calculate the CSV hour value based on the simulation step
    currentCsvHour() {
      // Extract hours from the formatted time
      const hourStr = this.formattedCurrentTime.split(':')[0];
      const hour = parseInt(hourStr);
      
      // In the CSV, hours are stored as 100, 200, etc.
      return hour * 100;
    }
  },

  created() {
    // Pre-load data as soon as component is created
    this.loadHourlyData();
  },
  
  mounted() {
    // Check if ApexCharts is available
    if (typeof window.ApexCharts === 'undefined') {
      console.error('ApexCharts is not loaded. Make sure the CDN script is included in your HTML.');
      this.loadingMessage = 'Chart library not available. Please check console for details.';
      return;
    }
    
    // If simulation is already running when component mounts, initialize charts
    if (this.isRunning && this.simulationState === 'manual' && this.dataLoaded) {
      this.initCharts();
      // Force an update to show current data
      this.forceChartUpdate();
    }

    // Add a global listener for visibility changes
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.isRunning && this.simulationState === 'manual') {
        // Force chart update when tab becomes visible again
        this.forceChartUpdate();
      }
    });
  },
  
  beforeDestroy() {
    this.destroyCharts();
  },
  
  methods: {
    destroyCharts() {
      // Destroy all chart instances to prevent memory leaks
      Object.values(this.charts).forEach(chart => {
        if (chart) {
          try {
            chart.destroy();
          } catch (e) {
            console.warn('Error destroying chart:', e);
          }
        }
      });
      
      // Reset chart references
      this.charts = {
        solar: null,
        temperature: null,
        price: null
      };
    },

    formatSimulationTime(step) {
      if (step === undefined || step === null) {
        return "00:00";
      }

      // Each step is 15 minutes
      const stepInDay = step % 96;
      const totalMinutes = stepInDay * 15;
      const hours = Math.floor(totalMinutes / 60);
      const minutes = totalMinutes % 60;

      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    },

    formatHourLabel(step) {
      // Safe formatter that always returns a valid string
      if (step === undefined || step === null) {
        return "00:00"; // Default value for invalid steps
      }
      
      // Create a time label for the x-axis
      return this.formatSimulationTime(step);
    },

    async loadHourlyData() {
      try {
        this.loadingMessage = 'Loading hourly data...';
        const result = await readHourlyData();
        
        if (result.success && result.hourlyData && result.hourlyData.length > 0) {
          this.hourlyData = result.hourlyData;
          this.dataLoaded = true;
          
          // Process data for the charts
          this.prepareChartData();
          
          // Set initial hour data
          this.updateCurrentHourData();
          
          // Emit the data-loaded event
          this.$emit('data-loaded', {
            success: true,
            hourlyData: [...this.hourlyData] // Create a copy
          });
          
          return true;
        } else {
          this.loadingMessage = 'Error loading hourly data. Please check the console.';
          console.error('Failed to load hourly data:', result.error);
          return false;
        }
      } catch (error) {
        this.loadingMessage = `Error: ${error.message || 'Unknown error loading data'}`;
        console.error('Error in loadHourlyData:', error);
        return false;
      }
    },
    
    retryLoad() {
      this.loadHourlyData().then(() => {
        if (this.dataLoaded) {
          this.destroyCharts();
          this.initCharts();
        }
      });
    },

    // Add this method to force update on simulation changes
    forceChartUpdate() {
      // Only proceed if we're in the right state
      if (!this.isRunning || this.simulationState !== 'manual' || !this.dataLoaded) {
        return;
      }
      
      console.log(`Forcing chart update for simulationStep: ${this.simulationStep}`);
      
      // Update current hour data first
      this.updateCurrentHourData();
      
      // Then update all charts
      if (this.charts.solar && this.charts.temperature && this.charts.price) {
        this.updateCharts();
      } else if (this.dataLoaded) {
        // If charts aren't initialized but data is loaded, initialize them
        console.log("Charts not initialized, initializing now");
        this.initCharts();
        
        // Give time for charts to render, then update with current data
        setTimeout(() => {
          this.updateCharts();
        }, 100);
      }
    },
    
    updateCurrentHourData() {
      if (!this.hourlyData || this.hourlyData.length === 0) return;
      
      try {
        // Get current step (modulo 96 to ensure it's within a day)
        const currentStep = this.simulationStep % 96;

        // Calculate the current hour based on simulation step
        const hourIndex = Math.floor((this.simulationStep % 96) / 4) % this.hourlyData.length;
        
        if (this.hourlyData[hourIndex]) {
          // Create a simple object with just the data we need
          this.currentHourData = {
            hour: this.hourlyData[hourIndex].hour,
            solar: this.hourlyData[hourIndex].solar,
            temperature: this.hourlyData[hourIndex].temperature,
            price: this.hourlyData[hourIndex].price
          };
          
          // Emit hour data update event
          this.$emit('hour-data-updated', { ...this.currentHourData });
        }
      } catch (err) {
        console.warn('Error updating current hour data:', err);
      }
    },
    
    // Create the chart configurations
    initCharts() {
      // If containers aren't ready yet, retry after a delay
      if (!this.$refs.solarChartContainer) {
        console.log('Chart containers not ready, retrying in 100ms');
        setTimeout(() => this.initCharts(), 100);
        return;
      }

      console.log('Initializing charts, ApexCharts available:', typeof window.ApexCharts !== 'undefined');
      console.log('Chart containers:', {
        solar: this.$refs.solarChartContainer,
        temperature: this.$refs.temperatureChartContainer,
        price: this.$refs.priceChartContainer
      });
      
      if (!this.dataLoaded || !this.dayData.current.length) return;
      
      // Check if ApexCharts is available
      if (typeof window.ApexCharts === 'undefined') {
        console.error('ApexCharts is not available. Make sure the CDN script is included.');
        return;
      }
      
      // Initialize chart configurations
      this.initSolarChart();
      this.initTemperatureChart();
      this.initPriceChart();
    },
    
    initSolarChart() {
      const options = this.createBaseChartOptions('Solar Output (kW)', this.chartColors.solar);
      
      // Add strict solar-specific options with explicit min/max to prevent auto-scaling issues
      options.yaxis = {
        min: 0,
        max: 6,
        tickAmount: 6,
        forceNiceScale: true,
        decimalsInFloat: 1,
        labels: {
          formatter: (value) => {
            if (value === undefined || value === null || !isFinite(value)) return '';
            return value.toFixed(1);
          }
        }
      };
      
      // Store options
      this.chartOptions.solar = options;
      
      // Create chart if container exists
      if (this.$refs.solarChartContainer) {
        this.charts.solar = new window.ApexCharts(this.$refs.solarChartContainer, options);
        this.charts.solar.render();
      }
    },
    
    initTemperatureChart() {
      const options = this.createBaseChartOptions('Temperature (°C)', this.chartColors.temperature);
      
      // Add strict temperature-specific options with explicit min/max
      options.yaxis = {
        min: 8,
        max: 30,
        tickAmount: 5,
        forceNiceScale: true,
        decimalsInFloat: 1,
        labels: {
          formatter: (value) => {
            if (value === undefined || value === null || !isFinite(value)) return '';
            return value.toFixed(1);
          }
        }
      };
      
      // Store options
      this.chartOptions.temperature = options;
      
      // Create chart if container exists
      if (this.$refs.temperatureChartContainer) {
        this.charts.temperature = new window.ApexCharts(this.$refs.temperatureChartContainer, options);
        this.charts.temperature.render();
      }
    },
    
    initPriceChart() {
      const options = this.createBaseChartOptions('Electricity Price ($/kWh)', this.chartColors.price);
      
      // Add strict price-specific options with explicit min/max
      options.yaxis = {
        min: 0,
        max: 0.10,
        tickAmount: 5,
        forceNiceScale: true,
        decimalsInFloat: 3,
        labels: {
          formatter: (value) => {
            if (value === undefined || value === null || !isFinite(value)) return '';
            return '$' + value.toFixed(3);
          }
        }
      };
      
      // Store options
      this.chartOptions.price = options;
      
      // Create chart if container exists
      if (this.$refs.priceChartContainer) {
        this.charts.price = new window.ApexCharts(this.$refs.priceChartContainer, options);
        this.charts.price.render();
      }
    },
    
    createBaseChartOptions(title, colors) {
      // Create a base chart configuration
      return {
        chart: {
          type: 'line',
          height: 200,
          fontFamily: 'Arial, sans-serif',
          animations: {
            enabled: false
          },
          toolbar: {
            show: false
          },
          zoom: {
            enabled: false
          }
        },
        series: [
          {
            name: 'Current Day',
            data: [] // Will be updated later
          },
          {
            name: 'Previous Day',
            data: [] // Will be updated later
          }
        ],
        colors: [colors.primary, 'rgba(200, 200, 200, 0.5)'],
        stroke: {
          curve: 'smooth',
          width: [2, 1]
        },
        markers: {
          size: 0,
          hover: {
            size: 5
          }
        },
        grid: {
          borderColor: '#f1f1f1',
          row: {
            colors: ['transparent', 'transparent'],
            opacity: 0.5
          }
        },
        xaxis: {
          type: 'category',
          categories: Array(96).fill(0).map((_, i) => this.formatHourLabel(i)),
          tickAmount: 6,
          labels: {
            formatter: (value) => {
              // Add null check before trying to split
              if (!value) return '';
              
              // Safely try to parse the hour
              try {
                const hour = parseInt(value.split(':')[0]);
                if (isNaN(hour)) return '';
                
                // Only show a subset of labels for readability
                if (hour % 4 === 0) return value;
                return '';
              } catch (e) {
                // If any error occurs, return empty string
                return '';
              }
            }
          },
          axisBorder: {
            show: false
          }
        },
        tooltip: {
          enabled: true,
          shared: true,
          intersect: false,
          y: {
            formatter: (value) => {
              if (value === undefined || value === null || !isFinite(value)) return '';
              return value.toFixed(2);
            }
          }
        },
        legend: {
          show: false
        },
        annotations: {
          xaxis: [
            {
              x: 0,
              strokeDashArray: 0,
              borderColor: colors.primary,
              label: {
                borderColor: colors.primary,
                style: {
                  color: '#fff',
                  background: colors.primary
                },
                text: 'Current Time'
              }
            }
          ]
        }
      };
    },

    prepareChartData() {
      if (!this.hourlyData || this.hourlyData.length === 0) return;
      
      // Create data for a full day (96 steps)
      this.dayData.current = new Array(96).fill(null).map((_, index) => {
        // Find the corresponding hourly data
        // Each 4 steps represents 1 hour (15-minute intervals)
        const hourIndex = Math.floor(index / 4) % this.hourlyData.length;
        
        // Get the data values
        let solar = this.hourlyData[hourIndex].solar;
        let temperature = this.hourlyData[hourIndex].temperature;
        let price = this.hourlyData[hourIndex].price;
        
        // Validate values to prevent Infinity or NaN
        solar = isFinite(solar) ? solar : 0;
        temperature = isFinite(temperature) ? temperature : 20;
        price = isFinite(price) ? price : 0.01;
        
        return {
          step: index,
          time: this.formatHourLabel(index),
          solar: solar,
          temperature: temperature,
          price: price
        };
      });
      
      // Initialize previous day data (will be shown faded)
      this.dayData.previous = [...this.dayData.current];
    },
    
    updateCharts() {
      if (!this.isRunning) {
        console.log('Not updating charts - data not loaded or simulation not running');
        return;
      }
      
      const currentStep = this.simulationStep % 96;
      // Check if we've just started a new day
      // if (this.lastUpdatedStep > currentStep) {
      //   // New day started, shift current day data to previous day
      //   this.dayData.previous = [...this.dayData.current];
      // }
      this.lastUpdatedStep = currentStep;

      // Get hourly data for current step
      const hourIndex = Math.floor(currentStep / 4) % this.hourlyData.length;
      const hourData = this.hourlyData[hourIndex];
      
      if (!hourData) {
        console.warn(`No hourly data found for index ${hourIndex}`);
        return;
      }

      console.log('Current hour data:', hourData);

      // Update current hour data
      this.currentHourData = {
        hour: hourData.hour,
        solar: hourData.solar,
        temperature: hourData.temperature,
        price: hourData.price
      };

      // Check if charts exist before updating
      if (!this.charts.solar || !this.charts.temperature || !this.charts.price) {
        console.warn('Charts not initialized yet, initializing now');
        this.initCharts();
        return;
      }

      // Update each chart
      this.updateSingleChart('solar', currentStep, hourData.solar);
      this.updateSingleChart('temperature', currentStep, hourData.temperature);
      this.updateSingleChart('price', currentStep, hourData.price);
    },
    
    updateSingleChart(type, step, value) {
      const chart = this.charts[chartType];
      if (!chart) {
        console.warn(`Chart ${type} not initialized`);
        return;
      }
      
      try {
        console.log(`Updating ${type} chart at step ${step} with value ${value}`);
        // Create data array with visible points leading up to current step
        const data = Array(96).fill(null);

        // This creates a "drawing" effect as the simulation progresses
        for (let i = 0; i <= step; i++) {
          // Get data for this step
          const pointHourIndex = Math.floor(i / 4) % this.hourlyData.length;
          if (this.hourlyData[pointHourIndex]) {
            data[i] = this.hourlyData[pointHourIndex][type];
          }
        }
        
        // Update series data
        chart.updateSeries([{
          name: chart.options.series[0].name,
          data: data
        }]);
        
        // Update current time marker annotation
        chart.updateOptions({
          annotations: {
            xaxis: [{
              x: step,
              borderColor: chart.options.colors[0],
              label: {
                text: 'Current',
                style: {
                  color: '#fff',
                  background: chart.options.colors[0]
                }
              }
            }]
          }
        }, false, false);
      } catch (error) {
        console.warn(`Error updating ${chartType} chart:`, error);
      }
    }
  },
  
  watch: {
    simulationStep: {
      handler(newStep, oldStep) {
        // Skip if no change
        if (newStep === oldStep) return;

        // Always update the current data
        this.updateCurrentHourData();
        
        // Only update if running in manual mode
        if (this.isRunning && this.simulationState === 'manual') {
          // Make sure charts are initialized
          if (!this.charts.solar && this.dataLoaded) {
            this.initCharts();
          }
          
          // Throttle updates to avoid overwhelming the browser
          if (this.charts.solar) {
            // Set the current step explicitly to match simulation time
            this.lastUpdatedStep = newStep % 96; // Keep within a day cycle
            this.updateCharts();
          }
        }
      },
      immediate: true // React immediately to initial value
    },
    
    isRunning: {
      handler(isNowRunning) {
        console.log(`isRunning changed to: ${isNowRunning}`);
        
        if (isNowRunning && this.simulationState === 'manual') {
          // Initialize and update charts when simulation starts
          this.$nextTick(() => {
            if (this.dataLoaded && !this.charts.solar) {
              this.initCharts();
            }
            this.forceChartUpdate();
          });
        }
      },
      immediate: true
    },
    
    simulationState: {
      handler(newState) {
        console.log(`simulationState changed to: ${newState}`);
        
        if (newState === 'manual' && this.isRunning) {
          // Initialize and update charts when switching to manual mode
          this.$nextTick(() => {
            if (this.dataLoaded && !this.charts.solar) {
              this.initCharts();
            }
            this.forceChartUpdate();
          });
        }
      },
      immediate: true
    },

    currentHourData: {
      handler() {
        // This ensures the UI updates when currentHourData changes
        // even if charts aren't updating properly
        this.$forceUpdate();
      },
      deep: true
    }
  }
};
</script>

<style scoped>
.hourly-charts-container {
  margin: 20px 0;
  padding: 15px;
  background-color: #f8fafc;
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.hourly-charts-container h3 {
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.2rem;
  color: #1f2937;
}

.loading-message {
  text-align: center;
  padding: 15px;
  color: #6b7280;
}

.retry-btn {
  margin-top: 10px;
  padding: 6px 12px;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
}

/* New compact grid layout */
.charts-compact-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.chart-container {
  background-color: white;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  position: relative;
  height: 200px; /* Reduced height */
}

.chart-wrapper {
  width: 100%;
  height: 160px; /* Adjusted for smaller container */
}

.chart-container h4 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 0.9rem;
  color: #4b5563;
}

.current-value {
  position: absolute;
  top: 10px;
  right: 12px;
  padding: 3px 6px;
  background-color: #f3f4f6;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #1f2937;
}

.time-info {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.time-display {
  text-align: center;
  width: 100%;
}

.time-value {
  font-size: 2rem;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 15px;
}

.time-progress {
  width: 100%;
  height: 6px;
  background-color: #e5e7eb;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-bar {
  height: 100%;
  background-color: #3b82f6;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.day-info {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
}

.day-label {
  font-weight: 600;
  color: #4b5563;
  font-size: 0.8rem;
}

.time-label {
  color: #6b7280;
  font-size: 0.8rem;
}

/* Media queries for different screen sizes */
@media (max-width: 768px) {
  .charts-compact-grid {
    grid-template-columns: 1fr;
  }
}

/* Add these CSS classes to make the component fit better when placed next to appliances */
:deep(.energy-flow-container) {
  display: flex;
  flex-wrap: wrap;
}

:deep(.appliances) {
  flex: 1;
  min-width: 320px;
}

:deep(.hourly-charts-container) {
  flex: 1;
  min-width: 320px;
  margin-left: 15px;
}
</style>