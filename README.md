# IoT Agentic Deployer

Artifact of the master's thesis *Agentic AI for the deployment and
configuration of IoT infrastructures* (Damiano Buzzo).

The system bridges the semantic gap between an architect's high-level intent,
expressed in natural language, and the low-level provisioning operations
required on an IoT platform. A supervising agent decomposes the activity into
sub-tasks and delegates them to specialised agents, which build a model of the
installation, select devices from a catalogue, validate the result and deploy
it onto the platform the architect chose (ThingsBoard or OpenRemote at the moment).

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

Optional, all falling back to a working default:

| Variable | Effect |
|---|---|
| `ROUTER_MODEL` | A different model for the supervisor's routing decision alone |
| `RESPONSE_MODEL` | A different model for the single reply the architect reads |
| `THINGSBOARD_MCP_URL` | Overrides `http://localhost:8000/sse` |
| `OPENREMOTE_URL` | Overrides `http://localhost:8080` |
| `OPENREMOTE_AUTH_URL` | Overrides `http://localhost:8081/auth` |
| `OPENREMOTE_REALM` | Overrides `master` |
| `OPENREMOTE_CLIENT_ID`, `OPENREMOTE_CLIENT_SECRET` | Credentials of an OpenRemote service user; required to deploy there |

The two model overrides exist because those two calls are the ones a weaker
model handles worst: routing decides who runs, and the reply is the only text
the architect actually reads.

## Usage

Start the interface, from the project root:

```bash
uv run streamlit run src/iot_agentic_deployer/main.py
```

The conversation, the model and the validation work with no platform running.
One is needed only to deploy.

### ThingsBoard

```bash
cd src/iot_agentic_deployer
docker compose up -d
```

ThingsBoard takes a few minutes on first boot; once it responds on
`localhost:9090`, restart the MCP server so it can authenticate:

```bash
docker compose restart mcp-server
```

Credentials: `tenant@thingsboard.org` / `tenant`.

### OpenRemote

```bash
cd src/iot_agentic_deployer
docker compose -f docker-compose.openremote.yaml up -d
```

The manager answers on `localhost:8080` and Keycloak on `localhost:8081`.
Create a service user in the realm and put its client id and secret in `.env`:
deployment authenticates before it does anything, so an unreachable or
unauthenticated platform stops the run before a single entity is written.

### A session

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
| Device catalogue | Declarative knowledge about device types, profiles and platform naming |
| Validation engine | Minimal topology rules and rules of the active profile |
| Platform adapters | Provisioning capabilities of each platform, as typed operations |
| Target IoT platform | ThingsBoard and OpenRemote |

The supervisor delegates to five specialised agents: **conceptualisation**
(builds the topology model and requests missing information), **configuration**
(applies the profile, associates devices with spaces and access points, and
models the ways in), **summary** (renders the configuration), **validation**
(evaluates the rules) and **deployment** (derives and executes the platform
operations).

A sixth node, **response**, is never delegated to: it runs once the turn is
over and composes everything the turn produced into the single reply the
architect reads, in their language, with the renderings the code built left
untouched.

A message that asks for more than one stage gets all of them: the supervisor
decides the whole sequence once, and the graph works through it without asking
again. Deployment always ends a sequence, because it suspends on the plan and
waits for the architect.

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

The second platform is what tests the boundary (R6). OpenRemote has neither a
device entity nor a relation entity: everything is an asset, and containment is
the child's own `parentId`. The plan above the adapter does not know and does
not change — `create_relation` simply updates the child instead. What each
platform calls what the model holds lives in `platform_mappings.yaml`, in its
own file rather than in every catalogue entry, so a third target does not mean
editing them all.

## Project structure

```
agents/            one module per agent, plus response.py
domain/            conceptual model and validation engine
domain/catalog/    devices.yaml, use_cases.yaml, platform_mappings.yaml, loader
platforms/         planning, adapter contract, ThingsBoard and OpenRemote adapters
rendering/         tables and diagrams built from the model
ui/                Streamlit interface
workflow.py        orchestration graph
```

Device types, use-case profiles and per-platform naming are declared in YAML
under `domain/catalog/`: supporting new equipment, a new profile or a new
target does not require modifying the source code (R4).
