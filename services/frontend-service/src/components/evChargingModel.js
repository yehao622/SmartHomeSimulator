// evChargingModel.js - Simple EV charging model

/**
 * Constants for EV charging simulation
 */
const EV_CONSTANTS = {
  // Battery properties
  BATTERY_CAPACITY: 60.0,      // kWh - Typical electric vehicle battery capacity
  MIN_SOC: 0.1,                // Minimum state of charge (10%)
  MAX_SOC: 0.95,               // Maximum state of charge (95%) - protecting battery
  INITIAL_SOC_RANGE: [0.3, 0.4], // Range for initial SoC when EV connects
  
  // Charger properties
  MAX_POWER_LEVEL_1: 1.9,      // kW - Level 1 charging (120V)
  MAX_POWER_LEVEL_2: 7.0,      // kW - Level 2 charging (240V)
  MAX_POWER_LEVEL_2_HIGH: 11.5, // kW - High-powered Level 2
  CHARGING_EFFICIENCY: 0.9,    // Typical charging efficiency
  
  // Price thresholds for smart charging
  LOW_PRICE_THRESHOLD: 0.025,   // $/kWh - Price below which to charge at max rate
  HIGH_PRICE_THRESHOLD: 0.045,  // $/kWh - Price above which to minimize charging
  
  // Time step in hours (15 minutes = 0.25 hours)
  TIME_STEP: 0.25,
  
  // EV connected hours (typical pattern)
  DEPARTURE_HOUR: 7,           // When EV typically leaves (7am)
  ARRIVAL_HOUR: 18,            // When EV typically arrives (6pm)
  
  // Target SoC for departure
  TARGET_SOC: 0.85             // Target 85% SoC for departure
};

/**
 * Determine if EV is connected based on time of day
 * @param {string} timeOfDay - Hour in format "HH:00"
 * @returns {boolean} Whether EV is connected
 */
export function isEvConnected(timeOfDay) {
  const hour = parseInt(timeOfDay.split(':')[0]);
  
  // EV is connected from evening until morning
  return hour >= EV_CONSTANTS.ARRIVAL_HOUR || hour < EV_CONSTANTS.DEPARTURE_HOUR;
}

/**
 * Calculate hours until departure
 * @param {string} timeOfDay - Hour in format "HH:00"
 * @returns {number} Hours until departure
 */
function hoursUntilDeparture(timeOfDay) {
  const hour = parseInt(timeOfDay.split(':')[0]);
  
  if (hour < EV_CONSTANTS.DEPARTURE_HOUR) {
    // Already morning, departure coming soon
    return EV_CONSTANTS.DEPARTURE_HOUR - hour;
  } else {
    // Evening or night, departure is tomorrow
    return (24 - hour) + EV_CONSTANTS.DEPARTURE_HOUR;
  }
}

/**
 * Get initial EV state of charge when first connected
 * @returns {number} Initial state of charge (0-1)
 */
export function getInitialEvSoC() {
  // Random SoC between MIN_RANGE and MAX_RANGE
  const [min, max] = EV_CONSTANTS.INITIAL_SOC_RANGE;
  return min + Math.random() * (max - min);
}

/**
 * Calculate EV charging power and next state of charge
 * @param {number} currentSoC - Current state of charge (0-1)
 * @param {boolean} isConnected - Whether EV is connected
 * @param {boolean} isCharging - Whether charging is active
 * @param {string} timeOfDay - Current time in format "HH:00"
 * @param {number} electricityPrice - Current electricity price ($/kWh)
 * @returns {Object} EV charging data including power and next SoC
 */
export function calculateEvChargingPower(currentSoC, isConnected, isCharging, timeOfDay, electricityPrice) {
  // If EV is not connected or charging is off, no power consumption
  if (!isConnected || !isCharging) {
    return {
      power: 0,
      nextSoC: currentSoC,
      isConnected: isConnected,
      hoursToDepart: isConnected ? hoursUntilDeparture(timeOfDay) : null
    };
  }
  
  // Calculate hours until departure
  const hoursToDepart = hoursUntilDeparture(timeOfDay);
  
  // Calculate energy needed to reach target
  const currentEnergy = currentSoC * EV_CONSTANTS.BATTERY_CAPACITY;
  const targetEnergy = EV_CONSTANTS.TARGET_SOC * EV_CONSTANTS.BATTERY_CAPACITY;
  const energyNeeded = targetEnergy - currentEnergy;
  
  // Calculate minimum power needed to reach target by departure time
  let requiredPower = energyNeeded / hoursToDepart;
  
  // If negative or very small, we're already charged enough
  if (requiredPower < 0.1) {
    return {
      power: 0,
      nextSoC: currentSoC,
      isConnected: true,
      hoursToDepart: hoursToDepart,
      alreadyCharged: true
    };
  }
  
  // Apply smart charging based on electricity price
  let smartChargingPower;
  
  if (electricityPrice <= EV_CONSTANTS.LOW_PRICE_THRESHOLD) {
    // Low price - charge at maximum rate
    smartChargingPower = EV_CONSTANTS.MAX_POWER_LEVEL_2;
  } else if (electricityPrice >= EV_CONSTANTS.HIGH_PRICE_THRESHOLD) {
    // High price - charge at minimum necessary rate
    smartChargingPower = requiredPower / EV_CONSTANTS.CHARGING_EFFICIENCY;
  } else {
    // Medium price - scale charging rate linearly
    const priceRange = EV_CONSTANTS.HIGH_PRICE_THRESHOLD - EV_CONSTANTS.LOW_PRICE_THRESHOLD;
    const priceRatio = (electricityPrice - EV_CONSTANTS.LOW_PRICE_THRESHOLD) / priceRange;
    
    // Scale between max power and required power
    smartChargingPower = EV_CONSTANTS.MAX_POWER_LEVEL_2 - (priceRatio * (EV_CONSTANTS.MAX_POWER_LEVEL_2 - requiredPower / EV_CONSTANTS.CHARGING_EFFICIENCY));
  }
  
  // Ensure we don't exceed max power
  let chargingPower = Math.min(smartChargingPower, EV_CONSTANTS.MAX_POWER_LEVEL_2);
  
  // Ensure we don't exceed what battery can accept (decreases as battery fills)
  const batteryAcceptanceLimit = (EV_CONSTANTS.MAX_SOC - currentSoC) * EV_CONSTANTS.BATTERY_CAPACITY / EV_CONSTANTS.TIME_STEP;
  chargingPower = Math.min(chargingPower, batteryAcceptanceLimit);
  
  // Ensure power is not negative
  chargingPower = Math.max(0, chargingPower);
  
  // Calculate energy transferred in this time step
  const energyTransferred = chargingPower * EV_CONSTANTS.CHARGING_EFFICIENCY * EV_CONSTANTS.TIME_STEP;
  
  // Calculate next state of charge
  const nextSoC = currentSoC + (energyTransferred / EV_CONSTANTS.BATTERY_CAPACITY);
  
  // Ensure SoC doesn't exceed max
  const clampedNextSoC = Math.min(nextSoC, EV_CONSTANTS.MAX_SOC);
  
  return {
    power: chargingPower,
    nextSoC: clampedNextSoC,
    isConnected: true,
    hoursToDepart: hoursToDepart,
    energyTransferred: energyTransferred,
    batteryAcceptanceLimit: batteryAcceptanceLimit,
    requiredPower: requiredPower
  };
}
