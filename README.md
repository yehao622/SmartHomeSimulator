# SmartHomeSimulator
A web UI-based Smart Home Energy Management System: A microservices-based platform for real-time energy management, optimization, and monitoring using reinforcement learning(RL) model.

This system simulates and optimizes energy consumption in smart homes through intelligent device management, real-time monitoring, and RL powered decision making. The platform integrates renewable energy sources, battery storage, and various home appliances to minimize energy costs while maintaining comfort.

🚀 Key Features

Real-time Energy Monitoring: Live visualization of energy flows between solar panels, battery storage, grid, and home appliances
AI-Powered Optimization: Reinforcement Learning (PPO) agent for intelligent energy management decisions
IoT Device Simulation: Comprehensive modeling of smart home appliances with realistic power consumption patterns
Advanced Analytics: Historical data analysis, predictive modeling, and performance metrics
Microservices Architecture: Scalable, containerized services for production deployment
Interactive Dashboard: Real-time charts and controls for system monitoring and manual operation

🏗️ Architecture
smart-home-cloud/

├── services/

│   ├── simulation-service/     # ✅ Core simulation logic (Node.js + Python RL)

│   ├── device-service/        # 🚧 Basic Go service (minimal implementation)

│   ├── optimization-service/  # 📋 Planned for future development

│   └── frontend-service/      # ✅ Vue.js frontend with real-time charts

├── k8s/                       # 🚧 Basic Kubernetes setup (expandable)

├── terraform/                 # 🚧 Basic AWS infrastructure template

└── docker-compose.yaml       # ✅ Full local development setup

### Current Implementation Status

**✅ Fully Implemented:**
- **Frontend Service**: Complete Vue.js application with real-time monitoring, charts, and controls
- **Simulation Service**: Node.js backend with Socket.IO, energy flow calculations, and RL integration
- **RL Service**: Python service with PPO reinforcement learning for energy optimization
- **Docker Compose**: Complete local development environment with all dependencies

**🚧 Basic Implementation:**
- **Device Service**: Basic Go REST API structure (ready for expansion)
- **Kubernetes**: Basic deployment manifest for simulation service
- **Terraform**: Basic AWS EKS cluster template

**📋 Planned Features:**
- **Optimization Service**: Advanced algorithms for multi-objective optimization
- **Extended K8s**: Complete microservices deployment with service mesh
- **Advanced Terraform**: Multi-environment infrastructure with monitoring

## 🎯 Current Features vs Roadmap

### ✅ Currently Working
- Real-time energy simulation with 11 different appliances
- AI-powered energy optimization using reinforcement learning
- Interactive dashboard with live charts and controls
- Energy flow visualization between solar, battery, grid, and home
- Historical data tracking and CSV export
- Smart appliance scheduling based on electricity prices
- HVAC and water heater thermal modeling
- EV charging optimization with time-of-use pricing

### 🚧 In Development
- Enhanced device management API (Go service expansion)
- Advanced Kubernetes deployment with auto-scaling
- Cloud infrastructure automation with Terraform

### 📋 Future Roadmap
- Multi-home/building management
- Advanced optimization algorithms
- Machine learning model retraining pipeline
- Advanced analytics and reporting

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+
- Python 3.10+
- Docker & Docker Compose
- Git

### Quick Start (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/yehaur/SmartHomeSimulator.git
cd SmartHomeSimulator

docker-compose up -d

### Run locally ports

Main Application: http://localhost:8000
Simulation API: http://localhost:3000
Basic Device API: http://localhost:8080
RL Service: http://localhost:5000

## System Architecture
![SystemArchitecture1]([https://github.com/user-attachments/assets/c18006a6-f561-4a56-84a0-6e53edc3f7c2](https://private-user-images.githubusercontent.com/26770675/449986434-837ee5a7-468a-496c-aed2-d1ba4d2f5f2c.png)
![SystemArchitecture2](https://github.com/user-attachments/assets/43d6c898-e6d2-4a9a-957e-bb662c4136f9)

## Running Screenshot
![Example](https://github.com/user-attachments/assets/837ee5a7-468a-496c-aed2-d1ba4d2f5f2c)

## 1-Day Simulation Example
![User](https://github.com/user-attachments/assets/80f9ceb5-30ba-43b7-a407-305f59980ef0)
![AI](https://github.com/user-attachments/assets/970a3388-a021-4ae0-9bbd-c7024b1e7eb7)
