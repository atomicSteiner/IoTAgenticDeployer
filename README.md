# IoT Agentic Deployer

Artifact of the master's thesis *Agentic AI for the deployment and
configuration of IoT infrastructures* (Damiano Buzzo).

## Scope of the project

The system bridges the semantic gap between an architect's high-level intent,
expressed in natural language, and the low-level provisioning operations
required on an IoT platform. A supervising agent decomposes the activity into
sub-tasks and delegates them to specialised agents, which build a model of the
installation, select devices from a catalogue, validate the result and deploy
it onto the platform the architect chose (ThingsBoard or OpenRemote at the
moment).

What it covers: describing a building in natural language, completing that
description by asking for what is missing, associating catalogue devices with
spaces one at a time or by space type, checking the result against the rules of
the chosen use case, and creating the corresponding entities and relations on a
target platform, with the plan shown and confirmed first.

What it does not: it configures an installation, it does not operate one. It
does not ingest telemetry, run dashboards or manage devices once they have been
provisioned, and it reads back from a platform only to show what it created.

Nothing but deployment needs a platform. The conversation, the model, the
catalogue and the validation all work with nothing running but the app itself,
which is also the quickest way to try it.

## Structure of the project

<img src="docs/architecture.png" alt="Architecture of the artifact" width="620">

The architect talks only to the conversational interface, and the supervisor is
the only agent that answers. Around the graph sit the parts that hold what is
known: the **configuration state store**, which keeps the model and its
checkpoints across steps and sessions; the **device catalogue and profiles**,
the declarative knowledge about device types, use cases and per-platform
naming; and the **validation engine**, which evaluates the minimal topology
rules and the rules of the active profile. Below them the **platform adapters**
expose what each target platform can provision as typed operations, which is
what confines ThingsBoard and OpenRemote to a single layer.

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

### Source layout

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

### Requirements coverage

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
not change, `create_relation` simply updates the child instead. What each
platform calls what the model holds lives in `platform_mappings.yaml`, in its
own file rather than in every catalogue entry, so a third target does not mean
editing them all.

## Prerequisites

To run the assistant:

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/). `uv` fetches a
  suitable Python itself if the system one is older.
- **An OpenRouter API key.** Every model call goes through OpenRouter, so one
  key covers all of them.

To deploy onto a platform, additionally:

- **Docker**, with the daemon actually running. On Windows and macOS that means
  Docker Desktop started, not merely installed.
- Roughly **2 GB of free disk** and a working connection: the platform images
  are pulled on first use, and ThingsBoard needs a few minutes to initialise
  its database before it answers.

## Installation

### 1. Get the source

```bash
git clone https://github.com/atomicSteiner/IoTAgenticDeployer.git
cd IoTAgenticDeployer
```

Every command below is run from this directory, which is what the rest of
this section calls the project root.

### 2. Dependencies

```bash
uv sync
```

### 3. Configuration

Copy `.env.example` to `.env` in the project root and fill in the key:

```
OPENAI_API_KEY="sk-or-..."
OPENAI_MODEL="anthropic/claude-haiku-4.5"
```

The variable is called `OPENAI_API_KEY` because the client is the
OpenAI-compatible one. The value is an OpenRouter key and starts with `sk-or-`.

Everything else is optional, and falls back to a working default:

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

### 4. Running it

**From the project root**, and only from there:

```bash
uv run streamlit run src/iot_agentic_deployer/main.py
```

The working directory matters. The session database (`iot_agentic.db`) and the
sidebar index (`sessions_index.json`) are created relative to it, so launching
from somewhere else gives a different and empty set of configurations rather
than an error. If earlier work seems to have disappeared, this is why.

### 5. ThingsBoard, only to deploy there

```bash
cd src/iot_agentic_deployer
docker compose up -d
```

ThingsBoard takes a few minutes on first boot; once it responds on
`localhost:9090`, restart the MCP server so it can authenticate:

```bash
docker compose restart mcp-server
```

Credentials: `tenant@thingsboard.org` / `tenant`. Go back to the project root
before starting the app.

### 6. OpenRemote, only to deploy there

```bash
cd src/iot_agentic_deployer
docker compose -f docker-compose.openremote.yaml up -d
```

The manager answers on `localhost:8080` and Keycloak on `localhost:8081`.

In the manager console, in the realm being deployed to (`master` unless
`OPENREMOTE_REALM` says otherwise), create a **service user** and put its
client id and secret in `.env`. Two things have to be right:

- it must be a service account, because the adapter authenticates with the
  client-credentials grant and never with a password;
- it must be allowed to **write** assets, not only to read them. Reading is all
  the reachability check performs, so a read-only service user passes the check
  and then fails on the very first entity, reporting `Interrupted after 0 of N
  operation(s)` with a 403. That is a permissions problem, not a deployment
  bug.

Deployment authenticates before it does anything, so an unreachable or
unauthenticated platform stops the run before a single entity is written.

### 7. A session

Describe the building, choose a use case, assign devices, ask for a validation
check, then ask to deploy. Deployment shows the operations it would perform and
stops there: it writes nothing until the architect replies in the chat to
confirm. Configurations are kept between sessions and listed in the sidebar.

### Starting over

ThingsBoard keeps its data outside the container, in `~/.mytb-data`, so
`docker compose down` leaves the tenant populated and a repeated deployment
meets the entities the previous one created. That is a legitimate scenario and
the assistant handles it, reusing an existing entity rather than duplicating
it, but for a clean demonstration stop the containers and remove
`~/.mytb-data` and `~/.mytb-logs` first. On the assistant's side, deleting
`iot_agentic.db` and `sessions_index.json` from the project root clears every
stored configuration.
