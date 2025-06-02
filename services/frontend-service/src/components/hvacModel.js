// hvacModel.js - Simple HVAC energy consumption model

/**
 * Constants for HVAC simulation
 */
const HVAC_CONSTANTS = {
  // Coefficient of Performance (COP) - efficiency factor
  COP_COOLING: 1.0,  // Typical for modern AC units
  COP_HEATING: 1.0,  // Typical for heat pumps
  
  // Building thermal characteristics
  BUILDING_FACTOR: 7.27,  // kW/°C - Combined thermal mass and insulation
  THERMAL_MASS: 1.0,      // kWh/°C - Thermal capacity of the building
  INSULATION_FACTOR: 0.968, // °C·h/kW - How well insulated the building is
  
  // Temperature setpoints and comfort range
  SETPOINT_COOLING: 20.0,  // °C - Target indoor temperature for cooling
  SETPOINT_HEATING: 19.0,  // °C - Target indoor temperature for heating
  COMFORT_RANGE: 2.0,      // °C - Acceptable deviation from setpoint
  
  // Power constraints
  MAX_POWER_COOLING: 2.5,  // kW - Maximum cooling power
  MAX_POWER_HEATING: 2.5,  // kW - Maximum heating power
  
  // Time step in hours (15 minutes = 0.25 hours)
  TIME_STEP: 0.25
};

/**
 * Calculate HVAC power consumption based on indoor and outdoor temperature
 * @param {number} indoorTemp - Current indoor temperature (°C)
 * @param {number} outdoorTemp - Current outdoor temperature (°C)
 * @param {boolean} isActive - Whether HVAC is currently active
 * @returns {Object} HVAC power consumption and next indoor temperature
 */
export function calculateHvacPower(indoorTemp, outdoorTemp, isActive) {
  // If HVAC is not active, calculate natural temperature change
  if (!isActive) {
    // Natural temperature drift toward outdoor temperature
    const temperatureDrift = (outdoorTemp - indoorTemp) * (HVAC_CONSTANTS.TIME_STEP / HVAC_CONSTANTS.INSULATION_FACTOR);
    const nextIndoorTemp = indoorTemp + temperatureDrift;
    
    return {
      power: 0,
      nextIndoorTemp: nextIndoorTemp
    };
  }
  
  // Determine heating or cooling mode
  const isCooling = outdoorTemp > HVAC_CONSTANTS.SETPOINT_COOLING;
  const setpoint = isCooling ? HVAC_CONSTANTS.SETPOINT_COOLING : HVAC_CONSTANTS.SETPOINT_HEATING;
  const cop = isCooling ? HVAC_CONSTANTS.COP_COOLING : HVAC_CONSTANTS.COP_HEATING;
  const maxPower = isCooling ? HVAC_CONSTANTS.MAX_POWER_COOLING : HVAC_CONSTANTS.MAX_POWER_HEATING;
  
  // Calculate temperature difference
  const temperatureDifference = indoorTemp - setpoint;
  
  // Calculate required power (positive for cooling, negative for heating)
  let requiredPower = HVAC_CONSTANTS.BUILDING_FACTOR * Math.abs(temperatureDifference);
  
  // Adjust power based on heating or cooling
  if ((isCooling && temperatureDifference < 0) || (!isCooling && temperatureDifference > 0)) {
    // If indoor temperature is already in comfort range in the right direction, reduce power
    requiredPower = requiredPower * 0.5;
  } else if ((isCooling && temperatureDifference > HVAC_CONSTANTS.COMFORT_RANGE) || 
             (!isCooling && temperatureDifference < -HVAC_CONSTANTS.COMFORT_RANGE)) {
    // If indoor temperature is significantly out of comfort range, increase power
    requiredPower = requiredPower * 1.5;
  }
  
  // Limit to maximum power
  requiredPower = Math.min(requiredPower, maxPower);
  
  // Calculate effective power with COP
  const effectivePower = requiredPower * cop;
  
  // Calculate new indoor temperature
  let temperatureChange = (effectivePower / HVAC_CONSTANTS.THERMAL_MASS) * HVAC_CONSTANTS.TIME_STEP;
  
  // Temperature change is negative for cooling, positive for heating
  if (isCooling) temperatureChange = -temperatureChange;
  else temperatureChange *= 3.0;
  
  // Also account for natural temperature drift
  const naturalDrift = (outdoorTemp - indoorTemp) * (HVAC_CONSTANTS.TIME_STEP / HVAC_CONSTANTS.INSULATION_FACTOR) * 0.5;
  
  // Calculate next indoor temperature
  const nextIndoorTemp = indoorTemp + temperatureChange + naturalDrift;
  
  return {
    power: requiredPower,  // Electrical power consumption
    nextIndoorTemp: nextIndoorTemp,
    effectivePower: effectivePower,  // Thermal power (heating/cooling)
    isCooling: isCooling,
    temperatureChange: temperatureChange,
    naturalDrift: naturalDrift
  };
}

/**
 * Generate a reasonable starting indoor temperature based on outdoor
 * @param {number} outdoorTemp - Current outdoor temperature
 * @returns {number} Starting indoor temperature
 */
export function getInitialIndoorTemperature(outdoorTemp) {
  // Start with a temperature that's between the setpoint and the outdoor temp
  const setpoint = outdoorTemp > HVAC_CONSTANTS.SETPOINT_COOLING ? 
                   HVAC_CONSTANTS.SETPOINT_COOLING : 
                   HVAC_CONSTANTS.SETPOINT_HEATING;
  
  // Temperature is 70% influenced by setpoint, 30% by outdoor
  return setpoint * 0.7 + outdoorTemp * 0.3;
}
