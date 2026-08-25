# IoT Agentic Deployer

Artifact of the master's thesis *Agentic AI for the deployment and
configuration of IoT infrastructures* (Damiano Buzzo).

The system bridges the semantic gap between an architect's high-level intent,
expressed in natural language, and the low-level provisioning operations
required on an IoT platform. A supervising agent decomposes the activity into
sub-tasks and delegates them to specialised agents, which build a model of the
installation, select devices from a catalogue, validate the result and deploy
it onto specific platform(at the moment Thingsboard).

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- Docker
- An OpenRouter API key

## Installation

```bash
uv sync
```

Create `.env` in the project root:

```
OPENAI_API_KEY="sk-or-..."
OPENAI_MODEL="meta-llama/llama-3.3-70b-instruct"
```

## Usage

Start the target platform and the MCP server:

```bash
cd src/iot_agentic_deployer
docker compose up -d
```

ThingsBoard takes a few minutes on first boot; once it responds on
`localhost:9090`, restart the MCP server so it can authenticate:

```bash
docker compose restart mcp-server
```

Start the interface:

```bash
uv run streamlit run main.py
```

ThingsBoard credentials: `tenant@thingsboard.org` / `tenant`.

## Architecture

| Component | Role |
|---|---|
| Conversational user interface | Where the architect describes, reviews and confirms |
| Orchestration graph | Supervisor and specialised agents as nodes of a stateful graph |
| Configuration state store | Model and checkpoints, persisted across steps and sessions |
| Device catalogue | Declarative knowledge about device types and use-case profiles |
| Validation engine | Minimal topology rules and rules of the active profile |
| Platform adapters | Provisioning capabilities of each platform, as typed operations |
| Target IoT platform | ThingsBoard, adopted as reference target |

The supervisor delegates to five specialised agents: **conceptualisation**
(builds the topology model and requests missing information), **configuration**
(applies the profile and associates devices with spaces), **summary** (renders
the configuration), **validation** (evaluates the rules) and **deployment**
(derives and executes the platform operations).

The language model is used only to recognise intent and to extract structure.
Catalogue lookup, device assignment, validation and planning are deterministic
code.

## Requirements coverage

| Req. | Realised by |
|---|---|
| R1 Natural language as the entry point | `agents/conceptualization.py` |
| R2 Guided conceptualisation of the environment | `domain/validation.py`, `missing_topology_information` |
| R3 Hierarchical decomposition of the workflow | `agents/supervisor.py`, `workflow.py` |
| R4 Catalogue-driven device selection | `domain/catalog/` |
| R5 Validation before execution | `domain/validation.py`, `agents/deployment.py` |
| R6 Automated, platform-mediated deployment | `platforms/` |
| R7 Stateful, resilient and traceable orchestration | LangGraph checkpointer, `trace` in `domain/state.py` |

## Project structure

```
agents/            one module per agent
domain/            conceptual model and validation engine
domain/catalog/    devices.yaml, use_cases.yaml and their loader
platforms/         planning, adapter contract, ThingsBoard adapter
ui/                Streamlit interface
workflow.py        orchestration graph
```

Device types and use-case profiles are declared in YAML under
`domain/catalog/`: supporting new equipment does not require modifying the
source code (R4).
