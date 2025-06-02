FROM node:18-alpine

WORKDIR /app

# Install curl for healthcheck
RUN apk --no-cache add curl

COPY package*.json ./

# Use npm ci instead of npm install for more reliable builds
RUN npm install --only=production --legacy-peer-deps || \
    npm install --only=production --no-optional || \
    npm install --only=production --force

COPY . .

# Expose the port the app runs on
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["node", "index.js"]
