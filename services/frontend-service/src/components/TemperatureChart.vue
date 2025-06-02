<!-- TemperatureChart.vue -->
<template>
  <div class="temperature-chart">
    <canvas ref="chartCanvas" width="600" height="250"></canvas>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';
import ChartManager from './chartConfig';

export default {
  props: {
    title: {
      type: String,
      required: true
    },
    currentTemperature: {
      type: Number,
      required: true
    },
    setPoint: {
      type: Number,
      required: true
    },
    timeStep: {
      type: Number,
      default: 0
    },
    history: {
      type: Array,
      default: () => []
    }
  },
  
  data() {
    return {
      chart: null,
      temperatureData: [],
      maxDataPoints: 96, // One day's worth of data (96 15-minute intervals)
      isChartInitialized: false,
      pendingUpdate: false,
      chartOptions: null
    };
  },
  
  mounted() {
    // Wait for DOM to be ready before initializing chart
    this.$nextTick(() => {
      this.initChart();
    });
  },
  
  methods: {
    initChart() {
      try {
        // Check if canvas exists
        if (!this.$refs.chartCanvas) {
          console.log('Canvas not found for', this.title);
          setTimeout(() => this.initChart(), 100); // Retry after a short delay
          return;
        }
    
        const ctx = this.$refs.chartCanvas.getContext('2d');
        if (!ctx) {
          console.error('Failed to get canvas context');
          return;
        }
    
        // Initialize data array
        this.temperatureData = new Array(this.maxDataPoints).fill(null);
    
        // Copy history data if available
        if (this.history && this.history.length > 0) {
          for (let i = 0; i < Math.min(this.history.length, this.maxDataPoints); i++) {
            this.temperatureData[i] = this.history[i];
          }
        }
    
        // Set current point
        if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
          this.temperatureData[this.timeStep] = this.currentTemperature;
        }
    
        // Destroy existing chart
        if (this.chart) {
          ChartManager.unregisterChart(this.title);
          this.chart.destroy();
          this.chart = null;
        }
    
        // Create chart configuration
        const chartConfig = {
          type: 'line',
          data: {
            labels: Array.from({ length: this.maxDataPoints }, (_, i) => i),
            datasets: [
              {
                label: 'Current Temperature',
                data: [...this.temperatureData],
                borderColor: this.title.includes('Home') ? '#3b82f6' : '#10b981',
                backgroundColor: this.title.includes('Home') ? '#3b82f6' : '#10b981',
                tension: 0.4,
                fill: false,
                pointRadius: 0,
                // Add segment support for varying colors
                segment: {
                  borderColor: (ctx) => {
                    return ctx.p0.parsed.x <= this.timeStep ? 
                      (this.title.includes('Home') ? '#3b82f6' : '#10b981') : 
                      (this.title.includes('Home') ? 'rgba(59, 130, 246, 0.3)' : 'rgba(16, 185, 129, 0.3)');
                  }
                }
              },
              {
                label: 'Set Point',
                data: new Array(this.maxDataPoints).fill(this.setPoint),
                borderColor: '#ff7300',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                fill: false
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // Disable animations for better performance
            plugins: {
              title: {
                display: true,
                text: this.title,
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
                      return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}°C`;
                    }
                    return null;
                  }
                }
              },
              legend: {
                display: true,
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
                  display: false // Remove grid lines
                },
                ticks: {
                  callback: function(value) {
                    // Show fewer ticks for clarity
                    return value % 24 === 0 ? value : '';
                  }
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Temperature (°C)'
                },
                grid: {
                  display: false // Remove grid lines
                },
                min: this.title.includes('Home') ? 18 : 55,
                max: this.title.includes('Home') ? 25 : 65,
              }
            }
          }
        };
    
        // Create the chart
        this.chart = new Chart(ctx, chartConfig);
    
        // Register the chart with the manager
        ChartManager.registerChart(this.title, this.chart);
    
        this.isChartInitialized = true;
    
        // Apply any pending updates
        if (this.pendingUpdate) {
          this.updateChart();
          this.pendingUpdate = false;
        }
      } catch (error) {
        console.error(`Error initializing chart (${this.title}):`, error.message);
        this.isChartInitialized = false;
    
        // Try again after a delay
        setTimeout(() => this.initChart(), 500);
      }
    },
    
    updateChart() {
      if (!this.chart || !this.isChartInitialized) {
        this.pendingUpdate = true;
        return;
      }

      try {
        // Update temperature data array
        if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
          // Create a new copy of the array
          const newData = [...this.temperatureData];
          newData[this.timeStep] = this.currentTemperature;
          this.temperatureData = newData;
          
          // Set the data and update colors based on current timeStep
          if (this.chart.data && this.chart.data.datasets) {
            this.chart.data.datasets[0].data = [...this.temperatureData];
            
            // Create a segmented color array where:
            // - Current day data (0 to timeStep) is full color
            // - Previous day data (timeStep+1 to end) is faded
            const pointBackgroundColors = [];
            const pointBorderColors = [];
            const pointRadius = [];
            
            const mainColor = this.title.includes('Home') ? '#3b82f6' : '#10b981'; // Blue for home, green for water
            const fadedColor = this.title.includes('Home') ? 'rgba(59, 130, 246, 0.3)' : 'rgba(16, 185, 129, 0.3)';
            
            for (let i = 0; i < this.maxDataPoints; i++) {
              if (i <= this.timeStep) {
                // Current day data - full color
                pointBackgroundColors.push(mainColor);
                pointBorderColors.push(mainColor);
              } else {
                // Previous day data - faded
                pointBackgroundColors.push(fadedColor);
                pointBorderColors.push(fadedColor);
              }
              
              // Only show point for current timeStep
              pointRadius.push(i === this.timeStep ? 4 : 0);
            }
            
            // Update pointBackgroundColor, pointBorderColor, and pointRadius arrays
            this.chart.data.datasets[0].pointBackgroundColor = pointBackgroundColors;
            this.chart.data.datasets[0].pointBorderColor = pointBorderColors;
            this.chart.data.datasets[0].pointRadius = pointRadius;
            
            // Update line segment colors
            this.chart.data.datasets[0].segment = {
              borderColor: (ctx) => {
                return ctx.p0.parsed.x <= this.timeStep ? mainColor : fadedColor;
              }
            };
          }
          
          // Update chart
          this.chart.update('none');
        }
      } catch (error) {
        console.warn(`Error updating chart (${this.title}):`, error.message);
      }
    },
    
    updateFromHistory() {
      if (!this.chart || !this.isChartInitialized) {
        this.pendingUpdate = true;
        return;
      }
      
      try {
        // Create a new array with the current data
        const newData = [...this.temperatureData];
        
        // Update with history data
        if (this.history && this.history.length > 0) {
          for (let i = 0; i < Math.min(this.history.length, this.maxDataPoints); i++) {
            if (this.history[i] !== null && this.history[i] !== undefined) {
              newData[i] = this.history[i];
            }
          }
        }
        
        // Ensure current timeStep is correct
        if (this.timeStep >= 0 && this.timeStep < this.maxDataPoints) {
          newData[this.timeStep] = this.currentTemperature;
        }
        
        // Update data
        this.temperatureData = newData;
        
        // Update chart data
        if (this.chart.data && this.chart.data.datasets) {
          this.chart.data.datasets[0].data = [...this.temperatureData];
        }
        
        // Safe update
        ChartManager.safeUpdateChart(this.title);
      } catch (error) {
        console.warn(`Error updating chart from history (${this.title}):`, error.message);
      }
    },

    resetChartData() {
      if (!this.chart || !this.isChartInitialized) return;
      
      try {
        // Create fresh empty data array
        this.temperatureData = new Array(this.maxDataPoints).fill(null);
        
        // Update chart dataset
        if (this.chart.data && this.chart.data.datasets && this.chart.data.datasets.length > 0) {
          this.chart.data.datasets[0].data = [...this.temperatureData];
          this.chart.data.datasets[1].data = new Array(this.maxDataPoints).fill(this.setPoint);
          this.chart.update('none');
        }
      } catch (error) {
        console.warn(`Error resetting chart data:`, error.message);
      }
    }
  },
  
  watch: {
    currentTemperature() {
      this.updateChart();
    },
    
    timeStep(newValue) {
      // Special case: if timeStep is reset to 0, we might be doing a full reset
      if (newValue === 0 && this.currentTemperature !== undefined) {
        // Check if all data is empty - likely a reset
        const allEmpty = this.temperatureData.every(val => val === null);
        if (allEmpty) {
          this.resetChartData();
        }
      }
      
      this.updateChart();
    },
    
    setPoint() {
      // If setpoint changes, update the chart
      if (this.chart && this.chart.data && this.chart.data.datasets && this.chart.data.datasets.length > 1) {
        try {
          this.chart.data.datasets[1].data = new Array(this.maxDataPoints).fill(this.setPoint);
          this.chart.update('none');
        } catch (error) {
          console.error(`Error updating setpoint (${this.title}):`, error);
        }
      }
    },
    
    history: {
      handler(newHistory) {
        // Only update if the new history is valid and different
        if (newHistory && Array.isArray(newHistory) && newHistory.length > 0) {
          this.updateFromHistory();
        }
      },
      deep: false
    },
    
    title() {
      // If title changes, reinitialize the chart
      this.isChartInitialized = false;
      this.$nextTick(() => {
        this.initChart();
      });
    }
  },
  
  beforeUnmount() {
    if (this.chart) {
      ChartManager.unregisterChart(this.title);
      this.chart.destroy();
      this.chart = null;
      this.isChartInitialized = false;
    }
  }
};
</script>

<style scoped>
.temperature-chart {
  width: 100%;
  height: 250px;
  background-color: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>