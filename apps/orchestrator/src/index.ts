import Fastify from 'fastify';
import chalk from 'chalk';
import cors from '@fastify/cors';
import fastifyStatic from '@fastify/static';
import fastifyWebsocket from '@fastify/websocket';
import path from 'path';

console.log(chalk.blue('🚀 Starting Chisel orchestrator...'));

const server = Fastify({
  logger: true
});

// In-memory storage for jobs (in production, use a database)
interface Job {
  id: string;
  profiler: string;
  command: string;
  gpu: string;
  flags: Record<string, any>;
  status: 'queued' | 'running' | 'completed' | 'failed';
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  logs: string[];
  result?: any;
}

const jobs: Map<string, Job> = new Map();

const setupServer = async () => {
  // Register plugins
  await server.register(cors, {
    origin: true
  });

  await server.register(fastifyWebsocket);

  // Serve static files from the ui/dist directory
  await server.register(fastifyStatic, {
    root: path.join(process.cwd(), 'ui/dist'),
    prefix: '/'
  });

  // API Routes
  server.get('/api/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  // Get all jobs
  server.get('/api/jobs', async () => {
    return Array.from(jobs.values()).sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  });

  // Get a specific job
  server.get('/api/jobs/:id', async (request: any, reply: any) => {
    const { id } = request.params as { id: string; };
    const job = jobs.get(id);
    if (!job) {
      return reply.code(404).send({ error: 'Job not found' });
    }
    return job;
  });

  // Submit a new job
  server.post('/api/jobs', async (request: any) => {
    const body = request.body as any;
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const job: Job = {
      id: jobId,
      profiler: body.profiler,
      command: body.command,
      gpu: body.gpu,
      flags: body.flags || {},
      status: 'queued',
      createdAt: new Date(),
      logs: [`Job ${jobId} created at ${new Date().toISOString()}`]
    };

    jobs.set(jobId, job);

    // Simulate job processing (in real implementation, this would dispatch to GPU workers)
    setTimeout(() => {
      job.status = 'running';
      job.startedAt = new Date();
      job.logs.push(`Job ${jobId} started at ${new Date().toISOString()}`);

      // Simulate some logs
      setTimeout(() => {
        job.logs.push(`Running ${job.profiler} profiler on ${job.gpu}...`);
        job.logs.push(`Command: ${job.command}`);

        setTimeout(() => {
          job.status = 'completed';
          job.completedAt = new Date();
          job.logs.push(`Job ${jobId} completed at ${new Date().toISOString()}`);
          job.result = {
            metrics: {
              gpuUtilization: Math.random() * 100,
              memoryUsage: Math.random() * 100,
              temperature: 60 + Math.random() * 20
            },
            profileData: {
              duration: Math.random() * 1000,
              samples: Math.floor(Math.random() * 10000)
            }
          };
        }, 3000);
      }, 1000);
    }, 1000);

    return { jobId, status: 'queued' };
  });

  // WebSocket endpoint for real-time logs
  server.get('/api/jobs/:id/logs', { websocket: true }, (connection: any, req: any) => {
    const { id } = req.params as { id: string; };
    const job = jobs.get(id);

    if (!job) {
      connection.socket.send(JSON.stringify({ error: 'Job not found' }));
      return;
    }

    // Send initial logs
    connection.socket.send(JSON.stringify({ logs: job.logs }));

    // Set up interval to send new logs
    const interval = setInterval(() => {
      const updatedJob = jobs.get(id);
      if (updatedJob && updatedJob.logs.length > job.logs.length) {
        const newLogs = updatedJob.logs.slice(job.logs.length);
        connection.socket.send(JSON.stringify({ logs: newLogs }));
        job.logs = [...updatedJob.logs];
      }
    }, 1000);

    connection.socket.on('close', () => {
      clearInterval(interval);
    });
  });

  // Catch-all route to serve the React app
  server.get('/*', async (request: any, reply: any) => {
    return reply.sendFile('index.html');
  });
};

const start = async () => {
  try {
    console.log(chalk.yellow('🔧 Attempting to start server...'));
    await setupServer();
    await server.listen({ port: 3001, host: '0.0.0.0' });
    console.log(chalk.green('✅ Orchestrator running on http://localhost:3001'));
    console.log(chalk.cyan('📡 Ready to accept job requests'));
    console.log(chalk.blue('🌐 Web UI available at http://localhost:3001'));
    console.log(chalk.gray('💡 Press Ctrl+C to stop the server'));
  } catch (err) {
    console.error(chalk.red('❌ Error starting server:'), err);
    server.log.error(err);
    process.exit(1);
  }
};

start(); 