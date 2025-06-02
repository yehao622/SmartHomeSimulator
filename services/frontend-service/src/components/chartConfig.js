// Create a new file called chartConfig.js in your project

// This file will help manage chart instances and prevent errors

// Singleton for managing chart instances and state
const ChartManager = {
  // Store chart instances
  chartInstances: {},
  
  // Flag to prevent updates during initialization
  updatesEnabled: true,
  
  // Register a chart
  registerChart(id, chart) {
    this.chartInstances[id] = chart;
    //console.log(`Chart registered: ${id}`);
  },
  
  // Unregister a chart
  unregisterChart(id) {
    if (this.chartInstances[id]) {
      delete this.chartInstances[id];
      //console.log(`Chart unregistered: ${id}`);
    }
  },
  
  // Get a chart
  getChart(id) {
    return this.chartInstances[id];
  },
  
  // Check if updates are allowed
  canUpdate() {
    return this.updatesEnabled;
  },
  
  // Pause updates
  pauseUpdates() {
    this.updatesEnabled = false;
  },
  
  // Resume updates
  resumeUpdates() {
    this.updatesEnabled = true;
  },
  
  // Safe update method
  safeUpdateChart(id) {
    if (!this.canUpdate()) return false;
    
    const chart = this.getChart(id);
    if (!chart) return false;
    
    try {
      chart.update('none');
      return true;
    } catch (error) {
      console.warn(`Error updating chart ${id}:`, error.message);
      return false;
    }
  },

  resetAllCharts() {
    for (const id in this.chartInstances) {
      try {
        const chart = this.chartInstances[id];
        if (chart && chart.data && chart.data.datasets) {
          // Clear each dataset appropriately
          chart.data.datasets.forEach(dataset => {
            if (Array.isArray(dataset.data)) {
              // Clear based on chart type
              if (id.toLowerCase().includes('temperature')) {
                dataset.data.fill(null);
              } else {
                dataset.data.fill(0);
              }
            }
          });
          chart.update('none');
        }
      } catch (error) {
        console.warn(`Error resetting chart ${id}:`, error.message);
      }
    }
    return true;
  },

  // Add memory usage optimizations
  optimizeMemoryUsage() {
    // Limit data points in each chart when in fast mode
    for (const id in this.chartInstances) {
      const chart = this.chartInstances[id];
      if (chart && chart.data && chart.data.datasets) {
        // Reduce point radius to improve rendering performance
        chart.data.datasets.forEach(dataset => {
          dataset.pointRadius = 0; // Hide all points
        });
        
        // Use simpler animations
        chart.options.animation = false;
        
        // Update with minimal impact
        try {
          chart.update('none');
        } catch (e) {
          console.warn(`Failed to optimize chart ${id}:`, e);
        }
      }
    }
  }
};

export default ChartManager;
