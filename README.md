# llm_bt_builder

Behavior Tree generator using Large Language Models (LLMs) for ROS 2 robots. Automatically creates Behavior Trees in XML format, using custom nodes defined in YAML. Three agent architectures are available:

| Agent | File | Description |
|---|---|---|
| `normal` | `bt_agent_node.py` | Iterative generate-validate-fix loop via raw HTTP API |
| `rag` | `bt_rag_agent_node.py` | Same loop + RAG (ChromaDB + HuggingFace embeddings) to pre-filter relevant nodes |
| `agentic` | `bt_agentic_node.py` | Fully agentic: the LLM autonomously calls validation tools via LangChain tool-calling |

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
ros2 run llm_bt_builder bt_agentic_node.py     # agentic (tool-calling)
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

Both agent nodes expose the `generate_bt` service:
- **Request:**
  - `objective`: Objective in natural language (string)
  - `bt_nodes_yaml`: YAML string with robot capability node definitions
- **Response:**
  - `success`: Whether generation was successful (bool)
  - `bt_xml`: Generated Behavior Tree in XML format (string)
  - `message`: Status message or model identifier (string)

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

### Agentic (`bt_agentic_node.py`)

Fully agentic architecture using LangChain `bind_tools()`:
- The LLM decides **when and in what order** to call the validation tools.
- 4 tools are exposed: `validate_xml_syntax`, `validate_bt_structure`, `validate_bt_semantics`, `submit_bt_xml`.
- `submit_bt_xml` is the termination signal — the LLM calls it when it considers the XML ready.
- A programmatic safety check runs after `submit_bt_xml` as a final guard.
- Also includes RAG for node pre-filtering.

Requires a **tool-calling capable model** (Gemini 1.5+, GPT-4o, Claude 3+, DeepSeek-v2+). Launch with `agent_type:=agentic`.

**Extra dependencies (RAG and Agentic):** See `requirements.txt` for LangChain, ChromaDB, sentence-transformers, etc.

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

## License

Apache License 2.0

## Author

Rodrigo Pérez-Rodríguez (rodrigo.perez@urjc.es)
