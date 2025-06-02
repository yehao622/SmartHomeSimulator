import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueJsx(), // Add JSX support for Vue components
  ],
  server: {
    host: '0.0.0.0',
    port: 8000
  },
  resolve: {
    alias: {
      '@': '/src', // Easy import alias for src directory
    },
  },
})
