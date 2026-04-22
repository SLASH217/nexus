# **Common Commands & Quick Reference**

## **Project Setup**

### Initialize Environment
```bash
# From project root
cd /home/slash/Coding/nexus

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (using uv)
uv sync

# Verify installation
python -c "from nexus_rl import NexusRlEnvironment; print('✓ Setup complete')"
```

---

## **Starting the Server (Local Development)**

### Option 1: UV Run (Recommended ✓ Best for development)
```bash
# Default: Development mode with auto-reload on localhost:8000
uv run python -m nexus_rl.server.app

# Custom port
uv run python -m nexus_rl.server.app --port 8001

# Server runs on: http://localhost:8000
# Web UI available at: http://localhost:8000/docs
```

### Option 2: Direct Uvicorn (Production-ready)
```bash
# Development with hot-reload
uv run uvicorn nexus_rl.server.app:app --reload --host 0.0.0.0 --port 8000

# Production with 4 workers (no auto-reload)
uv run uvicorn nexus_rl.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 3: Direct Python Module
```bash
# Without uv (if dependencies already installed)
python -m nexus_rl.server.app

# Custom port
python -m nexus_rl.server.app --port 8001
```

---

## **Docker Deployment**

### Build Docker Image
```bash
# From project root
cd /home/slash/Coding/nexus

# Build the image
docker build -t nexus-rl:latest -f nexus_rl/Dockerfile .

# With custom tag
docker build -t nexus-rl:0.1.0 -f nexus_rl/Dockerfile .
```

### Run Container
```bash
# Basic: Run on default port 8000
docker run -p 8000:8000 nexus-rl:latest

# With custom port mapping
docker run -p 8001:8000 nexus-rl:latest

# With environment variables
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true nexus-rl:latest

# Interactive mode (attach to logs)
docker run -it -p 8000:8000 nexus-rl:latest

# Detached mode (background)
docker run -d -p 8000:8000 --name nexus-server nexus-rl:latest
```

### Docker Compose (if needed)
```yaml
# docker-compose.yml
version: '3.8'
services:
  nexus-server:
    build:
      context: .
      dockerfile: nexus_rl/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENABLE_WEB_INTERFACE=true
    command: python -m nexus_rl.server.app --host 0.0.0.0 --port 8000
```

**Run with Docker Compose:**
```bash
docker-compose up
docker-compose down  # Stop services
```

---

## **Testing & Validation**

### Run Unit Tests
```bash
# All tests (using uv - no venv activation needed)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_environment.py -v

# With detailed output and short tracebacks
uv run pytest tests/ -v --tb=short

```

---

## **Environment & Debugging**

### Check Python Environment
```bash
# Show active Python
which python

# Show installed packages
pip list

# Show uv environment
uv env list
```

### View Logs
```bash
# If running in docker container
docker logs nexus-server

# Follow logs (live)
docker logs -f nexus-server
```

### Connect to Running Server
```bash
# Health check
curl http://localhost:8000/health

# Get environment schema
curl http://localhost:8000/schema

# Interactive API documentation
# Open browser: http://localhost:8000/docs
```

---

## **Visualization & Analysis**

### Generate Benchmarks & Visualizations
```bash
# Generate theoretical Pareto frontier and resource distribution plots
uv run scripts/generate_benchmarks.py

# Output: docs/pareto_frontier.png
# Also prints integrity checksum (total resources, initial utilities, etc.)
```

### Manual Plotting in Python
```python
from nexus_rl.server.nexus_rl_environment import NexusRlEnvironment

# Generate and save theoretical utility frontier
NexusRlEnvironment.generate_theoretical_optimal_utility_graph(
    output_path="docs/pareto_frontier.png"
)

# Generate trade dynamics from episode history
episodes = [...]  # List of episode transaction records
NexusRlEnvironment.generate_trade_dynamics_graph(
    episodes=episodes,
    output_path="docs/trade_dynamics.png"
)
```

---

## **Environment Variables**

### .env Configuration (nexus_rl/.env)
```bash
# Enable/disable web interface
ENABLE_WEB_INTERFACE=true

# Optional: API keys, credentials
HUGGING_FACE_TOKEN=<your_token>

# Optional: Model cache location
HF_HOME=/path/to/cache
```

---

## **Quick Troubleshooting**

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'openenv'` | Run `uv sync` to install dependencies |
| Port 8000 already in use | Use different port: `python -m nexus_rl.server.app --port 8001` |
| Docker image not found | Build first: `docker build -t nexus-rl:latest -f nexus_rl/Dockerfile .` |
| Permission denied in Docker | Run with `--privileged` flag or check volume permissions |
| Import errors in environment | Activate venv: `source .venv/bin/activate` |

---

## **Development Workflow**

### Edit → Test → Deploy Cycle
```bash
# 1. Make code changes in nexus_rl/

# 2. Run tests (using uv - no manual venv activation)
uv run pytest tests/ -v

# 3. Start server to verify changes (auto-reload active)
uv run python -m nexus_rl.server.app

# 4. Generate benchmarks to validate environment
uv run scripts/generate_benchmarks.py

# 5. Build Docker image
docker build -t nexus-rl:latest -f nexus_rl/Dockerfile .

# 6. Test in container
docker run -p 8000:8000 nexus-rl:latest
```

### Key Files
- **Environment Logic:** `nexus_rl/server/nexus_rl_environment.py`
- **Game Mechanics:** `nexus_rl/server/logic.py`
- **Data Models:** `nexus_rl/models.py`
- **Server Entry Point:** `nexus_rl/server/app.py`
- **Dependencies:** `pyproject.toml`

---

## **Accessing the Environment**

### Via Manual Agent Controller
```bash
# Terminal 1: Start the server
uv run python -m nexus_rl.server.app

# Terminal 2: Start manual agent (interactive play)
uv run python scripts/manual_agent.py

# Then type commands: PROPOSE, WAIT, RESET, EXIT
```

### Via HTTP (REST API)
```bash
# Reset environment
curl -X POST http://localhost:8000/reset

# Step with action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"PROPOSE","target_id":1,"offer_E":10,"request_C":5}'

# Get current state
curl http://localhost:8000/state
```

### Via Python Client
```python
from openenv import make_env

env = make_env("http://localhost:8000")
obs, _ = env.reset()

action = {"action_type": "PROPOSE", "target_id": 1, "offer_E": 10, "request_C": 5}
obs, reward, done, _, info = env.step(action)
```

---

## **Performance Notes**

- **Default Max Concurrent Envs:** 1 (in `app.py`, can be increased)
- **Recommended Workers:** 4 for production (adjust based on CPU cores)
- **Memory Usage:** ~500MB for base container + dependencies
- **Latency:** ~10-50ms per step (local), ~50-200ms (Docker)

---

**Last Updated:** April 22, 2026
**Project:** Nexus MARL
