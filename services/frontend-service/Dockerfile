FROM node:18-alpine

WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Start the development server instead of building
# This is simpler for local development
EXPOSE 8000
CMD ["npm", "run", "dev"]