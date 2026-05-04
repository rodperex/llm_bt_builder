# llm_bt_builder

Behavior Tree generator using Large Language Models (LLMs) for ROS 2 robots. Automatically creates Behavior Trees in XML format, using custom nodes defined in YAML. Three agent architectures are available:

| Agent | File | Description |
|---|---|---|
| `normal` | `bt_agent_node.py` | Iterative generate-validate-fix loop via raw HTTP API |
| `rag` | `bt_rag_agent_node.py` | Same loop + RAG (ChromaDB + HuggingFace embeddings) to pre-filter relevant nodes |
| `agentic` | `bt_rag_agentic_node.py` | Fully agentic: the LLM autonomously calls validation tools via LangChain tool-calling |

## Installation

### Requirements
- ROS 2
- Python 3.8+
- Dependencies:
  - rclpy
  - requests
  - PyYAML
  - torch (local mode only)
  - transformers (local mode only)
  - accelerate (local mode only)
  - langchain-core
  - langchain-chroma
  - langchain-huggingface
  - langchain-google-genai
  - langchain-ollama
  - chromadb
  - sentence-transformers

### Package Installation

1. Clone the repository into your ROS 2 workspace:
   ```bash
   git clone <repo_url> src/llm_bt_builder
   ```
2. Install Python dependencies:
   ```bash
   pip install -r src/llm_bt_builder/requirements.txt
   ```
3. Build the workspace:
   ```bash
   colcon build --packages-select llm_bt_builder
   source install/setup.bash
   ```

## Usage

### Launch Server Nodes

You can run any agent directly:
```bash
ros2 run llm_bt_builder bt_agent_node.py       # standard
ros2 run llm_bt_builder bt_rag_agent_node.py   # RAG
ros2 run llm_bt_builder bt_rag_agentic_node.py   # agentic (tool-calling)
```

#### Recommended: Use the launcher

The launcher selects the agent via `agent_type` (`normal` / `rag` / `agentic`):
```bash
# RAG agent (default)
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=rag model:=gemini-2.5-flash mode:=api key:=<API_KEY>

# Standard agent
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=normal model:=gpt-4o key:=<API_KEY>

# Agentic agent (requires a tool-calling capable model)
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=agentic provider:=openai model:=gpt-4o key:=<API_KEY>
```

> **Note:** `agent_type:=agentic` requires a model with tool-calling support: Gemini 1.5+, GPT-4o, Claude 3+, DeepSeek-v2+. Ollama support depends on the model.

### Launch Client Node

The client node reads objectives from a text file and sends them to the BT generation service:

```bash
ros2 launch llm_bt_builder bt_client.launch.py
```

To specify a different objective file:
```bash
ros2 launch llm_bt_builder bt_client.launch.py objective_file:=/path/to/your/objective.txt
```

To specify a different robot capabilities YAML file:
```bash
ros2 launch llm_bt_builder bt_client.launch.py capabilities_yaml:=/path/to/your/capabilities.yaml
```

The client loads the robot capabilities from the YAML file (default: `config/social_bt_nodes.yaml`) and includes them in the service request. The generated XML file will include the objective as a comment header for traceability.

### Configuration

All server nodes accept these ROS 2 parameters:
- `llm_provider`: `gemini`, `openai`, `anthropic`, `deepseek`, or `ollama`
- `model_id`: Model ID (e.g., `gemini-2.5-flash`, `gpt-4o`, `llama3.1`)
- `api_url`: REST endpoint URL (auto-detected per provider if empty)
- `api_key`: API key (auto-detected from env vars if empty: `GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.)
- `prompt_file`: Prompt template filename in `prompts/` (e.g., `system_prompt_cot.txt`)

The `bt_agent_node.py` additionally accepts:
- `execution_mode`: `local` (Hugging Face) or `api`

The client node accepts ROS 2 parameters:
- `objective_file`: Path to the text file containing the objective (default: `objectives/explain.txt`)
- `capabilities_yaml`: Path to the YAML file with robot capabilities (default: `config/social_bt_nodes.yaml`)

You can define custom robot capabilities in the YAML files:
- `config/bt_nodes.yaml` (general capabilities)
- `config/social_bt_nodes.yaml` (social interaction capabilities)

### ROS 2 Service

Available services by node type:

| Node | `/generate_bt` (`GenerateBT`) | `/fix_bt` (`FixBT`) |
|---|---|---|
| `bt_agent_node.py` (normal) | Yes | No |
| `bt_rag_agent_node.py` (rag) | Yes | Yes |
| `bt_rag_agentic_node.py` (agentic) | Yes | Yes |
| `mcp_bt_rag_agent_node.py` (mcp-rag) | Yes (inherited) | Yes (inherited) |

`GenerateBT` request/response:
- **Request:**
  - `objective`: Objective in YAML-like natural language/structured text (string)
  - `bt_nodes_yaml`: YAML string with robot capability node definitions
- **Response:**
  - `success`: Whether generation was successful (bool)
  - `bt_xml`: Generated Behavior Tree in XML format (string)
  - `message`: Status message or model identifier (string)

`FixBT` is used by RAG/agentic nodes to regenerate XML from a broken BT plus an explicit error message.

**Direct service call example:**
```bash
ros2 service call /generate_bt llm_bt_builder/srv/GenerateBT "{objective: 'Navigate to the kitchen and pick up the bottle', bt_nodes_yaml: '$(cat $(ros2 pkg prefix llm_bt_builder)/share/llm_bt_builder/config/bt_nodes.yaml)'}"
```

**Using the client node (recommended):**
The client node simplifies the workflow by:
- Reading the objective from a text file (configurable via `objective_file` parameter)
- Loading the robot capabilities YAML (configurable via `capabilities_yaml` parameter)
- Sending the request to the service
- Saving the generated XML with the objective as a comment header

See the "Launch Client Node" section above for usage examples.

## Objective Contract (from llm_planner)

`llm_bt_builder` expects each objective to be self-contained. At minimum:
- `description`
- optional `inputs` and `outputs`
- `recovery_policy`
- `steps`

`recovery_policy` schema:

```yaml
recovery_policy:
  required: true|false
  loop_until_success: true|false
  retry_attempts: int|forever
```

Semantics enforced during validation:
- If `required=false` and `loop_until_success=false`, `RetryUntilSuccessful` is not allowed.
- If `retry_attempts` is set and retry is allowed, `RetryUntilSuccessful num_attempts` must match exactly.
- If `required=true`, condition checks must be inside an explicit branching recovery structure (`Fallback` or `ReactiveFallback`).

This contract is parsed from `objective` in:
- `bt_rag_agent_node.py`
- `bt_agent_node.py`
- `bt_rag_agentic_node.py`

If the policy and BT structure conflict, generation is rejected with actionable validation feedback.

## End-to-End With llm_planner

Pipeline summary:
1. `llm_planner` creates plan steps and writes `objective.recovery_policy`.
2. `llm_bt_builder` generates BT XML per step.
3. Semantic validation checks node/port correctness and policy consistency.
4. On failure, RAG/agentic modes retry with targeted feedback (and `fix_bt` when applicable).

Debugging tip:
- If XML shape looks wrong, inspect the source step objective first. Many "bad BT" outcomes come from inconsistent `recovery_policy` in the plan.

## Robot Capabilities YAML Format

Define your robot's capabilities in YAML format. The agent nodes use this to generate valid Behavior Trees.

### Basic Structure

Each node definition includes:
- **name**: Node identifier
- **type**: `Action`, `Condition`, `Decorator`, or `Control`
- **description**: Brief description of what the node does
- **ports**: Input and output parameters
- **return**: Return status conditions (SUCCESS, RUNNING, FAILURE)

### Example Format

```yaml
bt_nodes:
  - name: "Speak"
    type: "Action"
    description: "Synthesizes and speaks the specified text using a TTS service."
    ports:
      - name: "text"
        direction: "Input"
        type: "string"
        description: "Text to be spoken."
      - name: "service_name"
        direction: "Input"
        type: "string"
        description: "Name of the TTS service (optional, default: /tts_service)."
      - name: "timeout"
        direction: "Input"
        type: "int"
        description: "Maximum wait time in ms (optional, default: 5000)."
    return:
      SUCCESS: "The speech has been completed (service responds and estimated speech duration elapses)."
      RUNNING: "Waiting for the service response or during speech playback."
      FAILURE: "The service is unavailable, the text parameter is missing, or the service call fails."

  - name: "IsTargetDetected"
    type: "Condition"
    description: "Checks if a target is detected using TF transforms."
    ports:
      - name: "target_frame"
        direction: "Input"
        type: "string"
        description: "Target frame to search for."
      - name: "base_frame"
        direction: "Input"
        type: "string"
        description: "Base reference frame."
      - name: "timeout"
        direction: "Input"
        type: "float"
        description: "Maximum wait time in seconds."
    return:
      SUCCESS: "The target frame is found and the transform is recent (not stale)."
      FAILURE: "Required inputs are missing, the transform cannot be found, or the transform is older than the timeout threshold."

  - name: "Follow"
    type: "Action"
    description: "Makes the robot follow a person using sensors and PID control."
    ports:
      - name: "cmd_vel_topic"
        direction: "Input"
        type: "string"
        description: "Velocity command topic."
      - name: "sonar_topic"
        direction: "Input"
        type: "string"
        description: "Sonar sensor topic."
      - name: "touch_topic"
        direction: "Input"
        type: "string"
        description: "Touch sensor topic."
    return:
      SUCCESS: "Only if succeed_on_reach input is true and the target is within minimum distance."
      RUNNING: "Actively following the target or when stopped by touch sensor or obstacles."
      FAILURE: "The target frame is lost or cannot be found via TF."
```

**Note:** Condition nodes typically only have SUCCESS and FAILURE states. Some continuous Action nodes may only return RUNNING until halted by the behavior tree.

## Agent Architectures

### Normal (`bt_agent_node.py`)

Sends the full capabilities YAML to the LLM and runs a hard-coded generate → validate → fix loop (up to 25 retries). Uses raw HTTP requests without LangChain, so it works with any model accessible via a REST API.

### RAG (`bt_rag_agent_node.py`)

Same iterative loop but adds a retrieval step:
- Indexes all YAML node definitions as vector embeddings (ChromaDB + HuggingFace `all-MiniLM-L6-v2`).
- Retrieves the top-K nodes semantically closest to the objective.
- Sends only those nodes to the LLM, reducing prompt size and hallucinations.

Best choice for large skill libraries. Launch with `agent_type:=rag`.

### Agentic (`bt_rag_agentic_node.py`)

Fully agentic architecture using LangChain `bind_tools()`:
- The LLM decides **when and in what order** to call the validation tools.
- 4 tools are exposed: `validate_xml_syntax`, `validate_bt_structure`, `validate_bt_semantics`, `submit_bt_xml`.
- `submit_bt_xml` is the termination signal — the LLM calls it when it considers the XML ready.
- A programmatic safety check runs after `submit_bt_xml` as a final guard.
- Also includes RAG for node pre-filtering.

Requires a **tool-calling capable model** (Gemini 1.5+, GPT-4o, Claude 3+, DeepSeek-v2+). Launch with `agent_type:=agentic`.

**Extra dependencies (RAG and Agentic):** See `requirements.txt` for LangChain, ChromaDB, sentence-transformers, etc.

---

## Choosing an architecture

The key difference between RAG (`bt_rag_agent_node`) and Agentic (`bt_rag_agentic_node`) is **where the reasoning lives**:

| Aspect | RAG | Agentic |
|---|---|---|
| Validation logic | Python (deterministic) | LLM (autonomous) |
| Retry decision | Hard-coded loop | LLM decides when to stop |
| Error injection | Node injects errors into next prompt | LLM reads tool results and self-corrects |
| LLM task | Generate XML only | Generate XML + navigate the tool flow |
| Tool-calling required | No | Yes (hard requirement) |
| Reasoning required | Low — any capable model | High — model must reason over tool outputs |

### When to use RAG

- The model is small or medium-sized (Llama 3.1, Gemini Flash, GPT-4o-mini).
- You want predictable retry behaviour regardless of model reasoning quality.
- You are using Ollama with a model that does not support tool calling.
- Robustness matters more than autonomy.

### When to use Agentic

- You are using a strong reasoning model: GPT-4o, Claude 3.5+, Gemini 2.5 Pro.
- The model has reliable, native tool-calling support.
- You want the LLM to decide the validation strategy rather than following a fixed script.
- You are experimenting with fully autonomous BT generation pipelines.

### Failure modes with weak models in Agentic mode

Using `agent_type:=agentic` with a model that lacks sufficient reasoning or tool-calling support can cause:

- **Infinite loops** — the model calls tools randomly without convergence.
- **Premature termination** — `submit_bt_xml` is called with incorrect XML before all validations pass.
- **Schema errors** — the model generates tool call arguments that do not match the expected schema, causing `invoke` to fail.

The programmatic guard on `submit_bt_xml` catches the last case, but the first two will exhaust `MAX_STEPS` (40) without producing a valid result. In those situations, switch to `agent_type:=rag`.

## Serving Local Models with Ollama

If you want to serve models locally using Ollama's API, follow these steps:

### Install Ollama

Visit [https://ollama.com/download](https://ollama.com/download) and follow the instructions for your operating system.

### Start Ollama Server

Once installed, start the Ollama server (it runs as a background service):
```bash
ollama serve
```

### Pull a Model

You need to pull a model before using it. For example, to pull the Qwen2.5-Coder model:
```bash
ollama pull qwen2.5-coder:1.5b
```
Or for Llama 3:
```bash
ollama pull llama3
```

### Test the API

You can test the API locally with curl:
```bash
curl http://localhost:11434/api/tags
```

### Use with llm_bt_builder

Set the following parameters in your launch file or node configuration:
- `execution_mode`: `api`
- `model_id`: The model name (e.g., `qwen2.5-coder:1.5b`)
- `api_url`: `http://localhost:11434/v1/chat/completions`
- `api_key`: (leave empty, not required for local Ollama)

This will allow llm_bt_builder to use your local Ollama server for LLM-based Behavior Tree generation.

## Local Mode vs API

- **Local:** Loads the model into memory (requires VRAM/RAM, useful for Hugging Face, Ollama, LM Studio).
- **API:** Uses REST services (OpenAI, Groq, Gemini, etc.), requires API key and URL.

## Launcher Examples

```bash
# RAG agent with Gemini
ros2 launch llm_bt_builder llm_agent.launch.py \
  agent_type:=rag provider:=gemini model:=gemini-2.5-flash key:=<API_KEY>

# Agentic agent with GPT-4o
ros2 launch llm_bt_builder llm_agent.launch.py \
  agent_type:=agentic provider:=openai model:=gpt-4o key:=<API_KEY>

# RAG agent with Ollama (local)
ros2 launch llm_bt_builder llm_agent.launch.py \
  agent_type:=rag provider:=ollama model:=llama3.1 url:=http://localhost:11434

# Agentic agent with Anthropic Claude
ros2 launch llm_bt_builder llm_agent.launch.py \
  agent_type:=agentic provider:=anthropic model:=claude-3-5-sonnet-20241022 key:=<API_KEY>
```

## For BT Node Developers

If you are developing new **BT nodes** or extending the behavior tree capabilities, these resources are essential:

### Error Code Reference

**📖 [ERROR_CODES.md](ERROR_CODES.md)** — Standard error codes for structured failure reporting.

This document defines the failure codes that BT nodes should emit when encountering errors. Using structured codes (instead of text-based error messages) ensures:
- ✅ The orchestrator can reliably classify failures (doesn't break if you rephrase the error message)
- ✅ The system can make robust decisions (FixBT vs Replan) based on error types
- ✅ New developers understand what codes to use when implementing nodes

**Key codes:**
- `bt_config_error` — Configuration error (missing required input, invalid parameter type)
- `execution_error` — General execution failure (default)
- `service_unavailable` — Service/action server not available (planned)
- `target_not_detected` — Perception/detection failure (planned)
- `timeout` — Action exceeded max time (planned)

**How to use:** When your BT node fails, call the `bt_failure()` helper function with the appropriate error code:

```cpp
#include "bt_nodes/bt_failure.hpp"

// In your BT action node:
if (!my_required_input) {
    return bt_failure(
        config(),
        registrationName(),
        "missing required input 'my_input', received: '" + my_required_input + "'",
        "bt_config_error"  // ← Structured code
    );
}
```

See [ERROR_CODES.md](ERROR_CODES.md) for the full list of codes, when to use each one, and implementation examples.

## License

Apache License 2.0

## Author

Rodrigo Pérez-Rodríguez (rodrigo.perez@urjc.es)
