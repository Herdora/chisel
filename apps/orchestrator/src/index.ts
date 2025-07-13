import Fastify from 'fastify';
import chalk from 'chalk';

console.log(chalk.blue('🚀 Starting Chisel orchestrator...'));

const server = Fastify();

server.get('/', async () => {
  return { hello: 'chisel orchestrator' };
});

const start = async () => {
  try {
    console.log(chalk.yellow('🔧 Attempting to start server...'));
    await server.listen({ port: 3001 });
    console.log(chalk.green('✅ Orchestrator running on http://localhost:3001'));
    console.log(chalk.cyan('📡 Ready to accept job requests'));
    console.log(chalk.gray('💡 Press Ctrl+C to stop the server'));
  } catch (err) {
    console.error(chalk.red('❌ Error starting server:'), err);
    server.log.error(err);
    process.exit(1);
  }
};

start(); 