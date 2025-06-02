// csvReader.js - Utility to read and process hourly CSV data

/**
 * Apply data constraints and validation to ensure all values are within reasonable ranges
 * @param {Array} hourlyData - The processed hourly data
 * @returns {Array} - The validated hourly data
 */
function validateHourlyData(hourlyData) {
  if (!hourlyData || !Array.isArray(hourlyData)) {
    return [];
  }
  
  return hourlyData.map(item => {
    // Create a new object with validated values
    return {
      // Preserve original hour
      hour: item.hour || "00:00",
      hourValue: item.hourValue || 0,
      
      // Validate solar (between 0 and 6 kW)
      solar: isFinite(item.solar) ? 
        Math.min(Math.max(parseFloat(item.solar) || 0, 0), 6) : 0,
      
      // Validate temperature (between -10 and 40°C)
      temperature: isFinite(item.temperature) ? 
        Math.min(Math.max(parseFloat(item.temperature) || 20, -10), 40) : 20,
      
      // Validate price (between 0 and 0.5 $/kWh)
      price: isFinite(item.price) ? 
        Math.min(Math.max(parseFloat(item.price) || 0.01, 0), 0.5) : 0.01,
      
      // Preserve any other raw data
      raw: item.raw || []
    };
  });
}

/**
 * Reads and processes the hourly CSV data file
 * @param {string} filePath - Path to the CSV file
 * @returns {Object} Processed hourly data
 */
export async function readHourlyData(filePath = null) {
  try {
    let csvData = null;
    
    // Try different approaches to get the CSV data
    if (!csvData) {
      try {
        // Attempt to use the fetch API (works in browsers)
        const response = await fetch('/hourly.csv');
        if (response.ok) {
          csvData = await response.text();
        }
      } catch (fetchError) {
        console.log('Fetch attempt failed:', fetchError);
      }
    }

    // If still no data, try window.fs if available
    if (!csvData && window.fs && typeof window.fs.readFile === 'function') {
      try {
        csvData = await window.fs.readFile('hourly.csv', { encoding: 'utf8' });
      } catch (fsError) {
        console.log('File system read attempt failed:', fsError);
      }
    }

    // If still no data, use generated sample data
    if (!csvData) {
      throw new Error("No sample data found for hourly simulation!");
    }
    
    console.log(`Successfully loaded hourly data`);
    
    // Split CSV by lines
    const lines = csvData.split('\n').filter(line => line.trim());
    
    // Get headers from first line
    const headers = lines[0].split(',').map(h => {
        // Clean up header: remove quotes and trim whitespace
        return h.trim().replace(/^["']|["']$/g, '');
    });
    //console.log(headers)
    
    // Find column indices for the data we need
    const hourIndex = headers.indexOf("Hour");
    const solarIndex = headers.indexOf("Solar");
    const tempIndex = headers.indexOf("T_out");
    const priceIndex = headers.indexOf("RTP");
    //console.log(hourIndex, solarIndex, tempIndex, priceIndex)
    
    // Check if we found all required columns
    if (hourIndex === -1 || solarIndex === -1 || tempIndex === -1 || priceIndex === -1) {
      throw new Error(`Required columns not found in CSV. Found: ${headers.join(', ')}`);
    }
    
    // Start from the 25th row (index 24) and process data
    const processedData = [];
    for (let i = 24; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      
      const values = lines[i].split(',').map(v => v.trim());
      
      // Process hour format
      let hour = parseInt(values[hourIndex] || '0');
      const hours = Math.floor(hour / 100);
      const formattedHour = `${hours.toString().padStart(2, '0')}:00`;
      
      // Process Solar data
      const solarValue = parseFloat(values[solarIndex] || '0');
      const solar = parseFloat(((solarValue) / 1000 * 5).toFixed(1));
      
      // Process outdoor temperature
      const temperature = parseFloat((parseFloat(values[tempIndex] || '0')).toFixed(1));
      
      // Process real-time price
      const price = parseFloat((parseFloat(values[priceIndex] || '0')).toFixed(3));
      
      processedData.push({
        hour: formattedHour,
        hourValue: hours,
        solar: solar,
        temperature: temperature,
        price: price,
        raw: values
      });
    }

    // After processing data, add validation step
    const validatedData = validateHourlyData(processedData);
    
    return {
      success: true,
      data: validatedData,
      hourlyData: validatedData,
      length: validatedData.length
    };
    
  } catch (error) {
    console.error("Error reading hourly data:", error);
    return {
      success: false,
      error: error.message,
      data: [],
      hourlyData: [],
      length: 0
    };
  }
}

// Add this function to detect and fix any invalid CSV data
function fixInvalidCSVData(csvData) {
  // If CSV data is completely invalid, return a sample dataset
  if (!csvData || typeof csvData !== 'string' || csvData.length < 10) {
    return; //generateSampleCSVData();
  }
  
  // Otherwise, try to fix common issues
  return csvData
    // Remove any null bytes or other control characters
    .replace(/[\x00-\x09\x0B-\x1F\x7F]/g, '')
    // Ensure lines end with proper line breaks
    .replace(/\r\n?/g, '\n')
    // Fix lines with too few columns (add empty values)
    .split('\n')
    .map(line => {
      const columns = line.split(',');
      if (columns.length < 3 && line.trim()) {
        // Add empty columns if needed
        return line + ','.repeat(3 - columns.length);
      }
      return line;
    })
    .join('\n');
}

/**
 * Get data for the current simulation hour
 * @param {Array} hourlyData - Processed hourly data
 * @param {number} simulationStep - Current simulation step (15min intervals)
 * @returns {Object} Data for the current hour
 */
export function getCurrentHourData(hourlyData, simulationStep) {
  // Each hour takes 4 steps (15min each)
  const hourIndex = Math.floor(simulationStep / 4) % hourlyData.length;
  return hourlyData[hourIndex] || null;
}

/**
 * Extract chart data from hourly data
 * @param {Array} hourlyData - Processed hourly data
 * @returns {Object} Extracted chart data
 */
export function extractChartData(hourlyData) {
  const hours = hourlyData.map(row => row.hour);
  const solar = hourlyData.map(row => row.solar);
  const temperature = hourlyData.map(row => row.temperature);
  const price = hourlyData.map(row => row.price);
  
  return {
    hours,
    solar,
    temperature,
    price
  };
}
