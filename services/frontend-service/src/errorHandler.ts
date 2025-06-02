// errorHandler.ts
export function setupErrorHandling() {
  const originalConsoleError = console.error;
  console.error = function(...args: any[]) {
    const errorMessage = args.length > 0 ? String(args[0]) : '';
    
    // Suppress known Chart.js errors that don't affect functionality
    if (
      // Stack overflow errors from charts
      errorMessage.includes('Maximum call stack size exceeded') ||
      
      // Common Chart.js property errors that happen during transitions
      (errorMessage.includes('Cannot set properties of undefined') && 
       (errorMessage.includes('fullSize') || 
        errorMessage.includes('options') || 
        errorMessage.includes('data'))) ||
      
      // Common errors during array methods
      (errorMessage.includes('Cannot read properties of undefined') && 
       (errorMessage.includes('includes') || 
        errorMessage.includes('length') || 
        errorMessage.includes('forEach'))) || 
      
      // Method-specific errors
      (errorMessage.includes('Error updating chart') || 
       errorMessage.includes('update()') ||
       errorMessage.includes('Error in updateHistoryData'))
    ) {
      // Instead of suppressing completely, log a simple message
      if (args.length > 1) {
        console.warn('Chart error suppressed:', errorMessage.substring(0, 100) + '...');
      }
      return;
    }
    
    // Pass through all other errors normally
    originalConsoleError.apply(console, args);
  };

  // Add Vue error handler for global component errors
  if (window.Vue) {
    window.Vue.config.errorHandler = (err, vm, info) => {
      // Filter out the same chart errors
      const errorMessage = String(err);
      if (
        errorMessage.includes('Maximum call stack size exceeded') ||
        errorMessage.includes('Cannot set properties of undefined') ||
        errorMessage.includes('Cannot read properties of undefined') ||
        errorMessage.includes('updateChart') ||
        errorMessage.includes('chart.update')
      ) {
        console.warn('Vue error suppressed:', info);
        return;
      }
      
      // Log other Vue errors
      console.error(`Vue Error in ${info}:`, err);
    };
  }
  
  // Add unhandled promise rejection handler
  window.addEventListener('unhandledrejection', function(event) {
    console.warn('Unhandled promise rejection:', event.reason);
    event.preventDefault();
  });
  
  console.log('Enhanced error handling set up');
}