import express from 'express';
import { Kafka } from 'kafkajs';
import { InfluxDB } from '@influxdata/influxdb-client';

const app = express();
const port = process.env.PORT || 3000;

interface SimulationState {
  devices: Device[];
  solarOutput: number;
  batteryLevel: number;
  gridDraw: number;
  timestamp: number;
}

class SimulationService {
  private kafka: Kafka;
  private influx: InfluxDB;

  constructor() {
    this.kafka = new Kafka({
      clientId: 'simulation-service',
      brokers: [process.env.KAFKA_BROKERS || 'localhost:9092']
    });

    this.influx = new InfluxDB({
      url: process.env.INFLUXDB_URL || 'http://localhost:8086',
      token: process.env.INFLUXDB_TOKEN
    });
  }

  async runSimulation() {
        // Simulate device states
        const state: SimulationState = {
            devices: [],
            solarOutput: Math.random() * 5,
            batteryLevel: Math.random() * 100,
            gridDraw: Math.random() * 3,
            timestamp: Date.now()
        };

        await this.publishState(state);
        await this.saveMetrics(state);
  }

  private async publishState(state: SimulationState) {
    const producer = this.kafka.producer();
    await producer.connect();
    await producer.send({
      topic: 'simulation.state',
      messages: [{ value: JSON.stringify(state) }]
    });
  }

  private async saveMetrics(state: SimulationState) {
    const writeApi = this.influx.getWriteApi(
      'simulation',
      'energy_metrics'
    );

    const point = new Point('energy_metrics')
            .floatField('solar_output', state.solarOutput)
            .floatField('battery_level', state.batteryLevel)
            .floatField('grid_draw', state.gridDraw);

    writeApi.writePoint(point);
    await writeApi.close();
  }
}

const simulation = new SimulationService();

app.get('/start', async (req, res) => {
    await simulation.runSimulation();
    res.json({ status: 'Simulation started' });
});

app.listen(port, () => {
    console.log(`Simulation service listening at http://localhost:${port}`);
});
EOF

echo "Sample implementation files created!"
