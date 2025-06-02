/**
 * Configuration loader service for the Smart Home Energy Simulator
 * Loads appliance and system configuration from JSON or YAML files
 */

// Function to load configuration from a file
export async function loadConfig(configPath) {
  try {
    const response = await fetch(configPath);
    if (!response.ok) {
      throw new Error(`Failed to load configuration file: ${response.statusText}`);
    }
    
    const config = await response.json();
    return config;
  } catch (error) {
    console.error('Error loading configuration:', error);
    throw error;
  }
}

// Process appliance data for the simulation
export function processApplianceConfig(configData) {
  if (!configData || !configData.appliances) {
    return [];
  }
  
  // Map the configuration format to the format expected by the simulator
  return configData.appliances.map(appliance => ({
    id: appliance.id,
    name: appliance.name,
    power: appliance.defaultPower,
    active: appliance.active || false,
    icon: appliance.icon || null,
    type: appliance.type,
    powerType: appliance.powerType,
    group: appliance.group || 3, // Default to group 3 if not specified
    connections: Array.isArray(appliance.connections) ? 
        appliance.connections.map(conn => typeof conn === 'string' ? conn : String(conn)) : [],
    minPower: appliance.minPower,
    maxPower: appliance.maxPower,
    schedule: appliance.schedule || [],
    runDuration: appliance.runDuration
  }));
}

// Process power sources for the simulation
export function processPowerSourcesConfig(configData) {
  if (!configData || !configData.powerSources) {
    return {
      solar: { capacity: 5, efficiency: 0.85 },
      battery: { capacity: 13.5, initialLevel: 50 },
      grid: { maxPower: 10, exportEnabled: true },
    };
  }
  
  const sources = {};
  
  configData.powerSources.forEach(source => {
    // Fix connections to ensure they're all strings
    if (source.connections) {
      source.connections = source.connections.map(conn => 
        typeof conn === 'string' ? conn : String(conn)
      );
    }
    sources[source.id] = { ...source };
  });
  
  return sources;
}

// Get settings from configuration
export function getSimulationSettings(configData) {
  if (!configData || !configData.settings) {
    return {
      timeStep: 300,
      location: { latitude: 37.7749, longitude: -122.4194 }
    };
  }
  
  return configData.settings;
}

// Save updated configuration back to a file (in a real app, this would be a server API call)
export async function saveConfig(configData, path) {
  // In a browser environment without Node.js file system access,
  // we would typically send this to a server API
  
  // For demonstration, we'll just log it and return success
  console.log('Would save configuration:', configData);
  return { success: true };
}

// Generate a default configuration if none is available
export function getDefaultConfig() {
  return {
    appliances: [
      // Group 1: Fixed power, must run for full duration
      { id: 1, name: 'Dishwasher', type: 'dishwasher', defaultPower: 1.8, active: false, group: 1 },
      { id: 2, name: 'Wash Machine', type: 'wash_machine', defaultPower: 0.4, active: false, group: 1 },
      { id: 3, name: 'Clothes Dryer', type: 'clothes_dryer', defaultPower: 1.2, active: false, group: 1 },
      
      // Group 2: Variable power, can be toggled during allowed time
      { id: 4, name: 'HVAC', type: 'hvac', defaultPower: 2.5, active: false, group: 2 },
      { id: 5, name: 'Water Heater', type: 'water_heater', defaultPower: 4.5, active: false, group: 2 },
      { id: 6, name: 'EV Charger', type: 'ev_charger', defaultPower: 6.0, active: false, group: 2 },
      
      // Group 3: Fixed power, on for entire duration
      { id: 7, name: 'TV', type: 'tv', defaultPower: 0.1, active: false, group: 3 },
      { id: 8, name: 'Refrigerator', type: 'refrigerator', defaultPower: 0.2, active: true, group: 3 },
      { id: 9, name: 'Lights', type: 'lights', defaultPower: 0.2, active: false, group: 3 },
      { id: 10, name: 'Vacuum Cleaner', type: 'vacuum', defaultPower: 1.2, active: false, group: 3 },
      { id: 11, name: 'Hair Dryer', type: 'hair_dryer', defaultPower: 1.0, active: false, group: 3 }
    ],
    powerSources: [
      { id: 'solar', name: 'Solar Panel', capacity: 5.0 },
      { id: 'battery', name: 'Battery', capacity: 13.5, initialLevel: 50 },
      { id: 'grid', name: 'Power Grid', maxPower: 10.0 },
      { id: 'house', name: 'Home' }
    ],
    settings: {
      timeStep: 300,
      location: { latitude: 37.7749, longitude: -122.4194 }
    }
  };
}