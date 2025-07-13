import Fastify from 'fastify';

const server = Fastify();

server.get('/', async () => {
  return { hello: 'chisel orchestrator' };
});

// TODO: Implement routes for /submit-job, etc.

const start = async () => {
  try {
    await server.listen({ port: 3000 });
    console.log('Orchestrator running on http://localhost:3000');
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};

start(); 