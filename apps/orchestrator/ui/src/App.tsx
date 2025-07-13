import { useState, useEffect } from 'react';
import { Play, Clock, CheckCircle, XCircle, Activity, Cpu, Zap } from 'lucide-react';
import './App.css';

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

function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [newJob, setNewJob] = useState({
    profiler: 'ncu',
    command: '',
    gpu: 'h100',
    flags: {}
  });

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/jobs');
      const data = await response.json();
      setJobs(data);
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    }
  };

  const submitJob = async () => {
    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newJob),
      });
      const data = await response.json();
      console.log('Job submitted:', data);
      setNewJob({ profiler: 'ncu', command: '', gpu: 'h100', flags: {} });
      fetchJobs();
    } catch (error) {
      console.error('Failed to submit job:', error);
    }
  };

  const viewJob = (job: Job) => {
    setSelectedJob(job);
    setLogs(job.logs);

    // Close existing WebSocket
    if (ws) {
      ws.close();
    }

    // Connect to WebSocket for real-time logs
    const websocket = new WebSocket(`ws://${window.location.host}/api/jobs/${job.id}/logs`);
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.logs) {
        setLogs(data.logs);
      }
    };
    setWs(websocket);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'queued':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'running':
        return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued':
        return 'bg-yellow-100 text-yellow-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <Zap className="w-8 h-8 text-blue-600" />
            Chisel Orchestrator
          </h1>
          <p className="text-gray-600 mt-2">GPU Profile Management Dashboard</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Job List */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">GPU Jobs</h2>
              </div>
              <div className="p-6">
                {/* New Job Form */}
                <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-900 mb-3">Submit New Job</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Profiler
                      </label>
                      <select
                        value={newJob.profiler}
                        onChange={(e) => setNewJob({ ...newJob, profiler: e.target.value })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        <option value="ncu">NCU (NVIDIA)</option>
                        <option value="nsys">NSYS (NVIDIA)</option>
                        <option value="torchprof">TorchProf</option>
                        <option value="rocprof">ROCProf (AMD)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        GPU Model
                      </label>
                      <select
                        value={newJob.gpu}
                        onChange={(e) => setNewJob({ ...newJob, gpu: e.target.value })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        <option value="h100">H100</option>
                        <option value="mi300x">MI300X</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Command
                      </label>
                      <input
                        type="text"
                        value={newJob.command}
                        onChange={(e) => setNewJob({ ...newJob, command: e.target.value })}
                        placeholder="python train.py"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      />
                    </div>
                    <button
                      onClick={submitJob}
                      disabled={!newJob.command}
                      className="w-full bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Submit Job
                    </button>
                  </div>
                </div>

                {/* Job List */}
                <div className="space-y-2">
                  {jobs.map((job) => (
                    <div
                      key={job.id}
                      onClick={() => viewJob(job)}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${selectedJob?.id === job.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                        }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(job.status)}
                          <span className="text-sm font-medium text-gray-900">
                            {job.profiler.toUpperCase()}
                          </span>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        {job.gpu.toUpperCase()} • {new Date(job.createdAt).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Job Details */}
          <div className="lg:col-span-2">
            {selectedJob ? (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <div className="px-6 py-4 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">
                      Job: {selectedJob.id}
                    </h2>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(selectedJob.status)}
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(selectedJob.status)}`}>
                        {selectedJob.status}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-6">
                  {/* Job Info */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Profiler</label>
                      <p className="text-sm text-gray-900">{selectedJob.profiler.toUpperCase()}</p>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">GPU</label>
                      <p className="text-sm text-gray-900">{selectedJob.gpu.toUpperCase()}</p>
                    </div>
                    <div className="col-span-2">
                      <label className="block text-xs font-medium text-gray-700 mb-1">Command</label>
                      <p className="text-sm text-gray-900 font-mono bg-gray-100 p-2 rounded">{selectedJob.command}</p>
                    </div>
                  </div>

                  {/* Results */}
                  {selectedJob.result && (
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-900 mb-3">Results</h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-blue-50 p-3 rounded-lg">
                          <div className="text-xs text-blue-600 font-medium">GPU Utilization</div>
                          <div className="text-lg font-semibold text-blue-900">
                            {selectedJob.result.metrics.gpuUtilization.toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-green-50 p-3 rounded-lg">
                          <div className="text-xs text-green-600 font-medium">Memory Usage</div>
                          <div className="text-lg font-semibold text-green-900">
                            {selectedJob.result.metrics.memoryUsage.toFixed(1)}%
                          </div>
                        </div>
                        <div className="bg-orange-50 p-3 rounded-lg">
                          <div className="text-xs text-orange-600 font-medium">Temperature</div>
                          <div className="text-lg font-semibold text-orange-900">
                            {selectedJob.result.metrics.temperature.toFixed(1)}°C
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Logs */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 mb-3">Live Logs</h3>
                    <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
                      {logs.map((log, index) => (
                        <div key={index} className="mb-1">
                          <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
                <Cpu className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Job Selected</h3>
                <p className="text-gray-500">Select a job from the list to view details and logs.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 