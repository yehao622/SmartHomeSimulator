#!/bin/bash

cd ./services/simulation-service
npm init -y
npm install typescript @types/node ts-node express @types/express kafkajs @influxdata/influxdb-client
npx tsc --init

# Initialize device-service (Go)
cd ../device-service
go mod init device-service
go get github.com/gorilla/mux
go get github.com/go-redis/redis/v8

# Initialize frontend-service (Vue.js)
cd ../frontend-service
npm create vite@latest . -- --template vue-ts
npm install chart.js vue-chartjs @vueuse/core

# Create base docker-compose file
cd ../../
cat > docker-compose.yaml << 'EOF'
version: '3.8'

services:
  simulation-service:
    build: ./services/simulation-service
    ports:
      - "3000:3000"
    environment:
      - KAFKA_BROKERS=kafka:9092
      - INFLUXDB_URL=http://influxdb:8086
    volumes:
      - ./services/simulation-service:/app
      - /app/node_modules
    depends_on:
      - kafka
      - influxdb

  device-service:
    build: ./services/device-service
    ports:
      - "8080:8080"
    environment:
      - REDIS_ADDR=redis:6379
    volumes:
      - ./services/device-service:/app
    depends_on:
      - redis

  frontend:
    build: ./services/frontend-service
    ports:
      - "8000:8000"
    volumes:
      - ./services/frontend-service:/app
      - /app/node_modules

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  influxdb:
    image: influxdb:latest
    ports:
      - "8086:8086"
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword
      - DOCKER_INFLUXDB_INIT_ORG=simulation
      - DOCKER_INFLUXDB_INIT_BUCKET=energy_metrics
    volumes:
      - influxdb-data:/var/lib/influxdb2

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  influxdb-data:
  redis-data:
EOF

# Create simulation service Dockerfile
cat > services/simulation-service/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

CMD ["npm", "run", "dev"]
EOF

# Create device service Dockerfile
cat > services/device-service/Dockerfile << 'EOF'
FROM golang:1.21-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git

# Initialize go module
COPY go.* ./
RUN go mod init device-service || true  # Initialize if not already initialized
RUN go mod tidy

COPY . .

RUN go build -o main .

CMD ["./main"]
EOF

# Create frontend service Dockerfile
cat > services/frontend-service/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

CMD ["npm", "run", "dev"]
EOF

echo "Project structure initialized!"
