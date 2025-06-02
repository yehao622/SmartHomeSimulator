// waterHeaterModel.js - Simple water heater energy consumption model

/**
 * Constants for water heater simulation
 */
const WATER_HEATER_CONSTANTS = {
  // Water properties
  SPECIFIC_HEAT: 4.186,  // kJ/kg·°C - Specific heat of water
  DENSITY: 1.0,         // kg/L - Density of water
  
  // Tank properties
  TANK_VOLUME: 200,             // L - Water tank volume
  TANK_INSULATION_FACTOR: 0.05, // °C/h - Rate of heat loss
  
  // Heater properties
  EFFICIENCY: 0.92,          // Typical electric water heater efficiency
  MAX_POWER: 4.5,            // kW - Maximum heating power
  
  // Temperature setpoints
  SETPOINT: 60.0,            // °C - Target water temperature
  COMFORT_RANGE: 3.0,        // °C - Acceptable deviation from setpoint
  
  // Incoming cold water temperature
  COLD_WATER_TEMP: 15.0,     // °C - Temperature of incoming cold water
  
  // Ambient temperature around tank
  AMBIENT_TEMP: 22.0,        // °C - Ambient temperature around tank
  
  // Time step in hours (15 minutes = 0.25 hours)
  TIME_STEP: 0.25,
  
  // Water usage patterns - fraction of tank replaced with cold water
  WATER_USAGE: {
    MORNING_PEAK: 0.10,      // Morning shower/bathroom usage
    EVENING_PEAK: 0.08,      // Evening dishes/bathroom usage
    MIDDAY: 0.03,            // Midday usage
    BACKGROUND: 0.01         // Background usage throughout day
  }
};

/**
 * Determine water usage based on time of day
 * @param {string} timeOfDay - Hour in format "HH:00"
 * @returns {number} Water usage as fraction of tank
 */
function getWaterUsage(timeOfDay) {
  const hour = parseInt(timeOfDay.split(':')[0]);
  
  // Morning peak (6am-9am)
  if (hour >= 6 && hour <= 9) {
    return WATER_HEATER_CONSTANTS.WATER_USAGE.MORNING_PEAK;
  }
  
  // Evening peak (6pm-10pm)
  if (hour >= 18 && hour <= 22) {
    return WATER_HEATER_CONSTANTS.WATER_USAGE.EVENING_PEAK;
  }
  
  // Midday usage (11am-2pm)
  if (hour >= 11 && hour <= 14) {
    return WATER_HEATER_CONSTANTS.WATER_USAGE.MIDDAY;
  }
  
  // Background usage
  return WATER_HEATER_CONSTANTS.WATER_USAGE.BACKGROUND;
}

/**
 * Calculate water heater power consumption based on current water temperature
 * @param {number} waterTemp - Current water temperature (°C)
 * @param {boolean} isActive - Whether water heater is currently active
 * @param {string} timeOfDay - Current time in format "HH:00"
 * @returns {Object} Water heater power consumption and next water temperature
 */
export function calculateWaterHeaterPower(waterTemp, isActive, timeOfDay) {
  // Get water usage for current time
  const waterUsage = getWaterUsage(timeOfDay);
  
  // If water heater is not active, calculate natural temperature change
  if (!isActive) {
    // Calculate heat loss due to tank insulation
    const heatLoss = WATER_HEATER_CONSTANTS.TANK_INSULATION_FACTOR * 
                    (waterTemp - WATER_HEATER_CONSTANTS.AMBIENT_TEMP) * 
                    WATER_HEATER_CONSTANTS.TIME_STEP;
    
    // Calculate temperature drop due to cold water coming in
    const waterReplacement = (WATER_HEATER_CONSTANTS.COLD_WATER_TEMP - waterTemp) * 
                            waterUsage;
    
    // Calculate next water temperature
    const nextWaterTemp = waterTemp - heatLoss + waterReplacement;
    
    return {
      power: 0,
      nextWaterTemp: nextWaterTemp,
      waterUsage: waterUsage,
      heatLoss: heatLoss,
      waterReplacement: waterReplacement
    };
  }
  
  // Calculate temperature difference from setpoint
  const temperatureDifference = WATER_HEATER_CONSTANTS.SETPOINT - waterTemp;
  
  // Only heat if water temperature is below setpoint
  if (temperatureDifference <= 0) {
    // No heating needed, just calculate natural changes
    const heatLoss = WATER_HEATER_CONSTANTS.TANK_INSULATION_FACTOR * 
                    (waterTemp - WATER_HEATER_CONSTANTS.AMBIENT_TEMP) * 
                    WATER_HEATER_CONSTANTS.TIME_STEP;
    
    const waterReplacement = (WATER_HEATER_CONSTANTS.COLD_WATER_TEMP - waterTemp) * 
                            waterUsage;
    
    const nextWaterTemp = waterTemp - heatLoss + waterReplacement;
    
    return {
      power: 0,
      nextWaterTemp: nextWaterTemp,
      waterUsage: waterUsage,
      heatLoss: heatLoss,
      waterReplacement: waterReplacement
    };
  }
  
  // Calculate energy needed to heat water
  const energyForHeating = WATER_HEATER_CONSTANTS.SPECIFIC_HEAT * 
                          WATER_HEATER_CONSTANTS.DENSITY * 
                          WATER_HEATER_CONSTANTS.TANK_VOLUME * 
                          temperatureDifference / 3600; // Convert from kJ to kWh
  
  // Calculate required power
  let requiredPower = energyForHeating / WATER_HEATER_CONSTANTS.TIME_STEP / 
                     WATER_HEATER_CONSTANTS.EFFICIENCY;
  
  // Limit to maximum power
  requiredPower = Math.min(requiredPower, WATER_HEATER_CONSTANTS.MAX_POWER);
  
  // Calculate temperature increase from heating
  const energyProvided = requiredPower * WATER_HEATER_CONSTANTS.EFFICIENCY * 
                        WATER_HEATER_CONSTANTS.TIME_STEP;
  
  const temperatureIncrease = energyProvided * 3600 / 
                             (WATER_HEATER_CONSTANTS.SPECIFIC_HEAT * 
                             WATER_HEATER_CONSTANTS.DENSITY * 
                             WATER_HEATER_CONSTANTS.TANK_VOLUME);
  
  // Calculate heat loss due to tank insulation
  const heatLoss = WATER_HEATER_CONSTANTS.TANK_INSULATION_FACTOR * 
                  (waterTemp - WATER_HEATER_CONSTANTS.AMBIENT_TEMP) * 
                  WATER_HEATER_CONSTANTS.TIME_STEP;
  
  // Calculate temperature drop due to cold water coming in
  const waterReplacement = (WATER_HEATER_CONSTANTS.COLD_WATER_TEMP - waterTemp) * 
                          waterUsage;
  
  // Calculate next water temperature
  const nextWaterTemp = waterTemp + temperatureIncrease - heatLoss + waterReplacement;
  
  return {
    power: requiredPower,
    nextWaterTemp: nextWaterTemp,
    temperatureIncrease: temperatureIncrease,
    waterUsage: waterUsage,
    heatLoss: heatLoss,
    waterReplacement: waterReplacement
  };
}

/**
 * Generate a reasonable starting water temperature
 * @param {string} timeOfDay - Current time in format "HH:00"
 * @returns {number} Starting water temperature
 */
export function getInitialWaterTemperature(timeOfDay) {
  const hour = parseInt(timeOfDay.split(':')[0]);
  
  // Lower temperature after morning peak
  if (hour >= 9 && hour <= 11) {
    return WATER_HEATER_CONSTANTS.SETPOINT - 8;
  }
  
  // Lower temperature after evening peak
  if (hour >= 22 || hour <= 1) {
    return WATER_HEATER_CONSTANTS.SETPOINT - 6;
  }
  
  // Otherwise, temperature is close to setpoint
  return WATER_HEATER_CONSTANTS.SETPOINT - 2;
}
