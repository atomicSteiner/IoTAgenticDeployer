# IoT Agentic Deployer

Artifact of the master's thesis *Agentic AI for the deployment and
configuration of IoT infrastructures* (Damiano Buzzo).

The system bridges the semantic gap between an architect's high-level intent,
expressed in natural language, and the low-level provisioning operations
required on an IoT platform. A supervising agent decomposes the activity into
sub-tasks and delegates them to specialised agents, which build a model of the
installation, select devices from a catalogue, validate the result and deploy
it onto a specific platform (ThingsBoard, at the moment).

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
OPENAI_MODEL="google/gemini-2.5-flash-lite"
```

Two variables are optional: `ROUTER_MODEL` uses a different model for the
supervisor's routing decision alone, falling back to `OPENAI_MODEL`, and
`THINGSBOARD_MCP_URL` overrides `http://localhost:8000/sse`.

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

Describe the building, choose a use case, assign devices, ask for a validation
check, then ask to deploy. Deployment shows the operations it would perform and
stops there: it writes nothing until the architect replies in the chat to
confirm. Configurations are kept between sessions and listed in the sidebar.

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

A message that asks for more than one stage gets all of them: the supervisor
decides the whole sequence once, and the graph works through it without asking
again. Deployment always ends a sequence, because it suspends on the plan and
waits for the architect.

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

Deployment is interruptible in the LangGraph sense: the plan suspends the graph
and the architect's reply resumes it. An interrupted run picks up where it
stopped, since operations are identified by the model rather than by the run,
and what already succeeded is not attempted again (R7).

## Project structure

```
agents/            one module per agent
domain/            conceptual model and validation engine
domain/catalog/    devices.yaml, use_cases.yaml and their loader
platforms/         planning, adapter contract, ThingsBoard adapter
rendering/         tables and diagrams built from the model
ui/                Streamlit interface
workflow.py        orchestration graph
```

Device types and use-case profiles are declared in YAML under
`domain/catalog/`: supporting new equipment does not require modifying the
source code (R4).
