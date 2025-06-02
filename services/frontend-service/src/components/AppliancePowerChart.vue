<!-- AppliancePowerChart.vue -->
<template>
  <div class="appliance-power-chart">
    <canvas ref="chartCanvas" width="600" height="250"></canvas>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';
import ChartManager from './chartConfig';

export default {
  props: {
    shiftableAppliances: {
      type: Object,
      required: true
    },
    timeStep: {
      type: Number,
      default: 0
    },
    powerHistory: {
      type: Object,
      default: () => ({})
    }
  },
  
  data() {
    return {
      chart: null,
      maxDataPoints: 96,
      isChartInitialized: false,
      pendingUpdate: false,
      isUpdating: false,  // Add this flag to prevent recursion
      lastUpdateTimeStep: null,  // Track the last update timestep
      dataCache: {},
      applianceColors: {
        dishwasher: {
          border: '#ef4444',
          background: 'rgba(239, 68, 68, 0.2)'
        },
        wash_machine: {
          border: '#8b5cf6',
          background: 'rgba(139, 92, 246, 0.2)'
        },
        clothes_dryer: {
          border: '#f59e0b',
          background: 'rgba(245, 158, 11, 0.2)'
        }
      }
    };
  },
  
  mounted() {
    // Wait for DOM to be ready before initializing chart
    this.$nextTick(() => {
      this.initChart();
    });
  },
  
  methods: {
    getApplianceTimepoints() {
      // This method analyzes the power history to find when appliances start or stop
      const timepoints = new Set([0, 95]); // Always include 0 and 95
      
      // Add current timeStep
      if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
        timepoints.add(this.timeStep);
      }
      
      // Check each appliance's history
      for (const appKey in this.powerHistory) {
        const history = this.powerHistory[appKey];
        if (!history) continue;
    
        // Safety check - make sure history is an array
        if (!Array.isArray(history)) {
          console.warn(`Power history for ${appKey} is not an array:`, history);
          continue;
        }
    
        if (history.length === 0) continue;
    
        // Scan for transitions (power on/off)
        for (let i = 1; i < history.length; i++) {
          const prevValue = history[i-1] || 0;
          const currValue = history[i] || 0;
      
          // If power changes from zero to non-zero (start) or non-zero to zero (stop)
          if ((prevValue === 0 && currValue > 0) || (prevValue > 0 && currValue === 0)) {
            timepoints.add(i);
          }
        }
      }
      
      return Array.from(timepoints).sort((a, b) => a - b);
    },

    initChart() {
      try {
        // Check if canvas exists
        if (!this.$refs.chartCanvas) {
          setTimeout(() => this.initChart(), 100);
          return;
        }
        
        const ctx = this.$refs.chartCanvas.getContext('2d');
        if (!ctx) {
          console.error('Failed to get canvas context');
          return;
        }
        
        // Initialize data
        this.dataCache = {};
        
        // Create datasets for each appliance
        const datasets = [];
        
        for (const appKey in this.shiftableAppliances) {
          const app = this.shiftableAppliances[appKey];
          const appName = this.formatApplianceName(appKey);
          
          // Define colors with higher transparency (0.1 opacity)
          let borderColor, backgroundColor;
          
          // Use appliance colors if available, otherwise use defaults
          if (this.applianceColors[appKey]) {
            borderColor = this.applianceColors[appKey].border;
            // Make background more transparent (0.1 opacity)
            backgroundColor = this.getTransparentColor(borderColor, 0.1);
          } else {
            // Default colors with transparency
            switch(appKey) {
              case 'dishwasher':
                borderColor = '#ef4444'; // Red
                backgroundColor = 'rgba(239, 68, 68, 0.1)';
                break;
              case 'wash_machine':
                borderColor = '#8b5cf6'; // Purple
                backgroundColor = 'rgba(139, 92, 246, 0.1)';
                break;
              case 'clothes_dryer':
                borderColor = '#f59e0b'; // Orange
                backgroundColor = 'rgba(245, 158, 11, 0.1)';
                break;
              default:
                borderColor = '#3b82f6'; // Blue
                backgroundColor = 'rgba(59, 130, 246, 0.1)';
            }
          }
          
          // Initialize data array
          const initialData = new Array(this.maxDataPoints).fill(0);
          
          // Update from history if available
          if (this.powerHistory[appKey]) {
            for (let i = 0; i < Math.min(this.powerHistory[appKey].length, this.maxDataPoints); i++) {
              initialData[i] = this.powerHistory[appKey][i] || 0;
            }
          }
          
          // Set current timestep
          if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
            initialData[this.timeStep] = app.active ? app.power : 0;
          }
          
          // Store in data cache
          this.dataCache[appKey] = [...initialData];
          
          // Create dataset with step interpolation and high transparency
          datasets.push({
            label: appName,
            data: [...initialData],
            borderColor: borderColor,
            borderWidth: 2,
            backgroundColor: backgroundColor,
            steppedLine: true,
            stepped: 'before',
            tension: 0,
            fill: true,
            pointRadius: 0,
            // Add segment support
            segment: {
              borderColor: (ctx) => {
                return ctx.p0.parsed.x <= this.timeStep ? 
                  borderColor : 
                  this.getFadedColor(borderColor, 0.3);
              },
              backgroundColor: (ctx) => {
                return ctx.p0.parsed.x <= this.timeStep ? 
                  backgroundColor : 
                  this.getFadedColor(borderColor, 0.05);
              }
            }
          });
        }
        
        // Destroy any existing chart
        if (this.chart) {
          ChartManager.unregisterChart('appliance-power');
          this.chart.destroy();
          this.chart = null;
        }
        
        // Define fixed labels for x-axis
        const fixedLabels = [0, 24, 48, 72, 95];
        
        // Create chart with segment drawing specified
        this.chart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: Array.from({ length: this.maxDataPoints }, (_, i) => i),
            datasets: datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            // Add segment drawing options for proper step function rendering
            elements: {
              line: {
                stepped: true       // Enable stepped lines globally
              }
            },
            plugins: {
              title: {
                display: true,
                text: 'Appliance Power Consumption',
                font: {
                  size: 16,
                  weight: 'bold'
                }
              },
              tooltip: {
                mode: 'index',
                intersect: false,
                callbacks: {
                  label: function(context) {
                    if (context.parsed.y !== null) {
                      return `${context.dataset.label}: ${context.parsed.y.toFixed(1)} kW`;
                    }
                    return null;
                  }
                }
              },
              legend: {
                position: 'top',
              }
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Time Step'
                },
                grid: {
                  display: false
                },
                ticks: {
                  callback: function(value) {
                    return fixedLabels.includes(value) ? value : '';
                  }
                }
              },
              y: {
                stacked: false,
                title: {
                  display: true,
                  text: 'Power (kW)'
                },
                grid: {
                  display: false
                },
                beginAtZero: true,
              }
            }
          }
        });
        
        // Register with chart manager
        ChartManager.registerChart('appliance-power', this.chart);
        
        this.isChartInitialized = true;
        
        if (this.pendingUpdate) {
          this.updateChart();
          this.pendingUpdate = false;
        }
      } catch (error) {
        console.error('Error initializing appliance power chart:', error.message);
        this.isChartInitialized = false;
        
        // Try again after a delay
        setTimeout(() => {
          this.initChart();
        }, 1000);
      }
    },
    
    // Add this utility method to convert colors to transparent versions
    getTransparentColor(color, opacity) {
      // If it's already in rgba format, just update the opacity
      if (color.startsWith('rgba')) {
        return color.replace(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/, `rgba($1, $2, $3, ${opacity})`);
      }
      
      // If it's in hex format, convert to rgba
      if (color.startsWith('#')) {
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
      }
      
      // For named colors or other formats, use a default with desired opacity
      return `rgba(59, 130, 246, ${opacity})`;
    },

    updateChart() {
      if (!this.chart || !this.isChartInitialized) {
        this.pendingUpdate = true;
        return;
      }

      try {
        // Guard against recursive updates
        if (this.isUpdating) {
          return;
        }
        
        this.isUpdating = true;
        
        // Only update if we have valid inputs
        if (!this.shiftableAppliances) {
          this.isUpdating = false;
          return;
        }
        
        let i = 0;
        for (const appKey in this.shiftableAppliances) {
          const app = this.shiftableAppliances[appKey];
          
          if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
            // Get the power value
            const value = app.active ? app.power : 0;

            if (appKey === 'clothes_dryer' && this.shiftableAppliances['wash_machine']) {
              const washMachine = this.shiftableAppliances['wash_machine'];
              // Only show clothes dryer power if washing machine has completed
              if (washMachine.progress < washMachine.total_duration) {
                value = 0; // Force dryer power to zero if washing not complete
              }
            }
            
            // Update the data cache
            if (!this.dataCache[appKey]) {
              this.dataCache[appKey] = new Array(this.maxDataPoints).fill(0);
            }
            
            // Only update if the value actually changed
            if (this.dataCache[appKey][this.timeStep] !== value) {
              // Create a new array copy
              const newData = [...this.dataCache[appKey]];
              newData[this.timeStep] = value;
              this.dataCache[appKey] = newData;
              
              // Update the chart dataset
              if (this.chart.data && 
                  this.chart.data.datasets && 
                  i < this.chart.data.datasets.length) {
                
                // Update data without triggering reactive updates
                this.chart.data.datasets[i].data[this.timeStep] = value;
                
                // Only update segment styling if needed
                if (!this.lastUpdateTimeStep || this.lastUpdateTimeStep !== this.timeStep) {
                  // Get colors for this appliance
                  const borderColor = this.getBorderColorForAppliance(appKey);
                  const backgroundColor = this.getBackgroundColorForAppliance(appKey, 0.2);
                  const fadedBackgroundColor = this.getBackgroundColorForAppliance(appKey, 0.05);
                  
                  // Create segment styling
                  this.chart.data.datasets[i].segment = {
                    borderColor: (ctx) => {
                      return ctx.p0.parsed.x <= this.timeStep ? 
                        borderColor : 
                        this.getFadedColor(borderColor, 0.3);
                    },
                    backgroundColor: (ctx) => {
                      return ctx.p0.parsed.x <= this.timeStep ? 
                        backgroundColor : 
                        fadedBackgroundColor;
                    }
                  };
                }
              }
            }
            i++;
          }
        }
        
        // Update the chart only once after all datasets have been modified
        if (this.lastUpdateTimeStep !== this.timeStep) {
          this.chart.update('none');
          this.lastUpdateTimeStep = this.timeStep;
        }
        
        this.isUpdating = false;
      } catch (error) {
        this.isUpdating = false;
        console.warn('Error updating appliance chart:', error.message);
      }
    },

    getBorderColorForAppliance(appKey) {
      // Get border color based on appliance type
      switch (appKey) {
        case 'dishwasher':
          return '#ef4444'; // Red
        case 'wash_machine':
          return '#8b5cf6'; // Purple
        case 'clothes_dryer':
          return '#f59e0b'; // Orange
        default:
          return '#3b82f6'; // Blue
      }
    },
    getBackgroundColorForAppliance(appKey, opacity) {
      // Get background color with specified opacity
      const borderColor = this.getBorderColorForAppliance(appKey);
      return this.getTransparentColor(borderColor, opacity);
    },
    getFadedColor(color, opacity) {
      // Get faded version of a color
      return this.getTransparentColor(color, opacity);
    },

    updateFromHistory() {
      if (!this.chart || !this.isChartInitialized) {
        this.pendingUpdate = true;
        return;
      }
      
      // Avoid recursion
      if (this.isUpdating) return;
      this.isUpdating = true;
      
      try {
        // Only update with valid history data
        if (!this.powerHistory) return;
        
        let hasChanges = false;
        
        for (let datasetIndex = 0; datasetIndex < this.chart.data.datasets.length; datasetIndex++) {
          const appKeys = Object.keys(this.powerHistory);
          if (datasetIndex < appKeys.length) {
            const appKey = appKeys[datasetIndex];
            
            // Skip if no dataset exists
            if (!this.chart.data.datasets[datasetIndex]) continue;
            
            // Update with history data
            if (this.powerHistory[appKey]) {
              for (let i = 0; i < Math.min(this.powerHistory[appKey].length, this.maxDataPoints); i++) {
                const historyValue = this.powerHistory[appKey][i] || 0;

                // Special case for clothes dryer - check washing machine status at this timestamp
                if (appKey === 'clothes_dryer' && this.powerHistory['wash_machine']) {
                  // Check if washing machine was active at any point before this timestamp
                  let washingCompleted = false;
                  for (let j = 0; j <= i; j++) {
                    if (this.powerHistory['wash_machine'][j] > 0) {
                      washingCompleted = true;
                      break;
                    }
                  }
                  
                  // If washing wasn't completed yet, force dryer to be inactive
                  if (!washingCompleted) {
                    historyValue = 0;
                  }
                }
                
                // Only update if values differ
                if (!this.dataCache[appKey] || this.dataCache[appKey][i] !== historyValue) {
                  // Initialize if needed
                  if (!this.dataCache[appKey]) {
                    this.dataCache[appKey] = new Array(this.maxDataPoints).fill(0);
                  }
                  
                  // Update the data point directly
                  this.dataCache[appKey][i] = historyValue;
                  this.chart.data.datasets[datasetIndex].data[i] = historyValue;
                  hasChanges = true;
                }
              }
            }
          }
        }
        
        // Only update chart if data changed
        if (hasChanges) {
          ChartManager.safeUpdateChart('appliance-power');
        }
      } finally {
        // Always reset the flag
        this.isUpdating = false;
      }
    },    
    
    formatApplianceName(key) {
      return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase());
    }
  },
  
  watch: {
    // Replace the current watcher with this:
    shiftableAppliances: {
      handler() {
        // Skip updates during recursive calls
        if (this.isUpdating) return;
        
        // Set the recursion prevention flag
        this.isUpdating = true;
        
        try {
          // Only process if chart is initialized
          if (this.chart && 
              this.chart.data && 
              this.chart.data.datasets && 
              this.timeStep >= 0 && 
              this.timeStep < this.maxDataPoints) {
            
            let hasChanges = false;
            
            // Process each appliance
            for (let i = 0; i < this.chart.data.datasets.length; i++) {
              const appKeys = Object.keys(this.shiftableAppliances);
              if (i < appKeys.length) {
                const appKey = appKeys[i];
                const app = this.shiftableAppliances[appKey];
                const value = app.active ? app.power : 0;
                
                // Only update if value has changed
                if (!this.dataCache[appKey] || this.dataCache[appKey][this.timeStep] !== value) {
                  hasChanges = true;
                  
                  // Initialize data cache if needed
                  if (!this.dataCache[appKey]) {
                    this.dataCache[appKey] = new Array(this.maxDataPoints).fill(0);
                  }
                  
                  // Update only the specific index
                  this.dataCache[appKey][this.timeStep] = value;
                  
                  // Direct update of data point (no array recreation)
                  this.chart.data.datasets[i].data[this.timeStep] = value;
                }
              }
            }
            
            // Only update chart if necessary
            if (hasChanges) {
              this.chart.update('none');
            }
          } else if (!this.isChartInitialized) {
            // Delay the update if chart isn't ready yet
            this.pendingUpdate = true;
          }
        } finally {
          // Always reset the recursion prevention flag
          this.isUpdating = false;
        }
      },
      deep: true
    },    
    
    timeStep() {
      this.updateChart();
    },
    
    powerHistory: {
      handler() {
        //console.log('Power chart received new history data');
        if (this.powerHistory && typeof this.powerHistory === 'object' && Object.keys(this.powerHistory).length > 0) {
          this.updateFromHistory();
        }
      },
      deep: false
    }
  },
  
  beforeUnmount() {
    // Clean up chart when component is unmounted
    if (this.chart) {
      ChartManager.unregisterChart('appliance-power');
      this.chart.destroy();
      this.chart = null;
      this.isChartInitialized = false;
    }
  }
};
</script>

<style scoped>
.appliance-power-chart {
  width: 100%;
  height: 250px;
  background-color: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>