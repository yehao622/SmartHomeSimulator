// applianceScheduler.js - Generate realistic appliance schedules for smart home simulation

/**
 * Constants for appliance scheduling
 * Time is represented as 15-minute intervals (0-95) in a day:
 * - 0 = 00:00, 4 = 01:00, 8 = 02:00, etc.
 * - 32 = 08:00, 48 = 12:00, 64 = 16:00, 80 = 20:00
 */

// Appliance scheduling parameters
const APPLIANCE_PARAMS = {
  // Group 1: Fixed power appliances that must run for full duration
  DISHWASHER: {
    START_RANGE: [68, 77],   
    END_RANGE: [86, 95],     
    DURATION: 3,             
    GROUP: 1
  },
  
  WASH_MACHINE: {
    START_RANGE: [36, 48],   
    END_RANGE: [60, 72],     
    DURATION: 6,             
    GROUP: 1
  },
  
  CLOTHES_DRYER: {
    START_RANGE: null,       // Special case: starts after wash machine
    END_RANGE: null,         // Determined by start time + 8
    DURATION: 5,             // 2 hours (8 * 15min)
    GROUP: 1
  },
  
  // Group 3: Fixed power appliances that can be toggled anytime
  TV: {
    START_RANGE: [72, 80],   // 18:00 - 20:00
    DURATION_RANGE: [12, 20], // 3-5 hours (12-20 * 15min)
    GROUP: 3
  },
  
  REFRIGERATOR: {
    START_RANGE: [0, 0],     // Starts at midnight
    DURATION_RANGE: [95, 95], // Runs all day
    GROUP: 3
  },
  
  LIGHTS: {
    START_RANGE: [68, 72],   // 17:00 - 18:00 
    DURATION_RANGE: [16, 24], // 4-6 hours (16-24 * 15min)
    GROUP: 3
  },
  
  VACUUM: {
    START_RANGE: [52, 62],   // 13:00 - 15:30
    DURATION_RANGE: [1, 6],  // 1-2 hours (4-8 * 15min)
    GROUP: 3
  },
  
  HAIR_DRYER: {
    START_RANGE: [76, 86],   // 19:00 - 21:30
    DURATION_RANGE: [1, 1],  // 15-30 minutes (1-2 * 15min)
    GROUP: 3
  }
};

/**
 * Determines a random time within a given range
 * @param {Array} range - [min, max] time range
 * @returns {number} - A random time value within the range
 */
function determineTime(range) {
  if (!range || !Array.isArray(range) || range.length !== 2) {
    throw new Error('Invalid range: Must be an array with two elements [min, max]');
  }
  
  const [min, max] = range;
  
  if (min === max) return min;
  
  // Return random integer between min and max (inclusive)
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Generate schedules for all appliances
 * @returns {Object} - Schedule for all appliances
 */
function generateApplianceSchedules() {
  const schedules = {};
  let washMachineEndTime = 0;
  
  // Process each appliance
  for (const [applianceType, params] of Object.entries(APPLIANCE_PARAMS)) {
    // Handle special case for clothes dryer (depends on washing machine)
    if (applianceType === 'CLOTHES_DRYER') {
      // Skip initial generation - we'll handle it after washing machine
      continue;
    }
    
    // Handle regular Group 1 appliances
    if (params.GROUP === 1) {
      const earlyStartTime = determineTime(params.START_RANGE);
      const deadline = determineTime(params.END_RANGE);
      const lateStartTime = deadline - params.DURATION;
      
      // Ensure valid start time range
      if (lateStartTime < earlyStartTime) {
        console.warn(`Invalid schedule for ${applianceType}: deadline too early for duration`);
        continue;
      }
      
      const startTime = determineTime([earlyStartTime, lateStartTime]);
      const endTime = startTime + params.DURATION;
      
      schedules[applianceTypeToId(applianceType)] = {
        startTime: startTime,
        endTime: endTime
      };
      
      // Remember washing machine end time for clothes dryer
      if (applianceType === 'WASH_MACHINE') {
        washMachineEndTime = endTime;
      }
    }
    
    // Handle Group 3 appliances
    if (params.GROUP === 3) {
      const startTime = determineTime(params.START_RANGE);
      let duration = params.DURATION_RANGE ? 
        determineTime(params.DURATION_RANGE) : 
        params.DURATION;

      const endTime = startTime + duration;
      
      schedules[applianceTypeToId(applianceType)] = {
        startTime: startTime,
        endTime: endTime
      };
    }
  }
  
  // Now handle clothes dryer - it starts right after washing machine
  if (washMachineEndTime > 0) {
    const dryerParams = APPLIANCE_PARAMS.CLOTHES_DRYER;
    const startTime = washMachineEndTime;
    const endTime = startTime + dryerParams.DURATION;
    
    schedules[applianceTypeToId('CLOTHES_DRYER')] = {
      startTime: startTime,
      endTime: endTime
    };
  } else {
    console.warn("Clothes dryer not initialized!")
    // Default clothes dryer schedule if no washing machine
    schedules[applianceTypeToId('CLOTHES_DRYER')] = {
      startTime: 60, // 15:00
      endTime: 68,   // 17:00
    };
  }
  
  return schedules;
}


function applianceTypeToId(type) {
  const mapping = {
    'DISHWASHER': 'dishwasher',
    'WASH_MACHINE': 'wash_machine', 
    'CLOTHES_DRYER': 'clothes_dryer',
    'TV': 'tv',
    'REFRIGERATOR': 'refrigerator',
    'LIGHTS': 'lights',
    'VACUUM': 'vacuum',
    'HAIR_DRYER': 'hair_dryer'
  };
  
  return mapping[type] || type.toLowerCase();
}


// Export functions for use in the main application
export {
  generateApplianceSchedules,
  applianceTypeToId
};