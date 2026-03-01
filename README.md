# llm_bt_builder

Iterative Behavior Tree generator using Large Language Models (LLMs) for ROS 2 robots. Automatically creates Behavior Trees in XML format, using custom nodes defined in YAML and iterative reasoning with LLMs (local or API).

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

You can launch the standard agent node directly:
```bash
ros2 run llm_bt_builder bt_agent_node.py
```

Or launch the RAG agent node:
```bash
ros2 run llm_bt_builder bt_rag_agent_node.py
```

#### Recommended: Use the launcher

The launcher allows you to select agent type, model, mode, API URL and key:
```bash
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=rag model:=gemini-2.5-flash mode:=api url:=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions key:=<API_KEY>
```
For the standard agent, use `agent_type:=normal`.

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

Both server nodes accept ROS 2 parameters:
- `execution_mode`: `local` (uses Hugging Face) or `api` (uses OpenAI, Groq, Ollama, LM Studio, etc.)
- `model_id`: Model ID (e.g., `Qwen/Qwen2.5-Coder-1.5B-Instruct`)
- `api_url`: REST endpoint URL (e.g., `http://localhost:11434/v1/chat/completions` for Ollama)
- `api_key`: API key (only for cloud services)

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

## RAG Mode (Retrieval-Augmented Generation)

The RAG agent (`bt_rag_agent_node.py`) uses LangChain and embeddings to select the most relevant nodes from your YAML definitions before invoking the LLM. This improves the quality and relevance of generated Behavior Trees, especially for large skill libraries.

You can launch it directly or via the launcher with `agent_type:=rag`.

**Extra dependencies:** See requirements.txt for LangChain, ChromaDB, sentence-transformers, etc.

**How it works:**
- Indexes your YAML node definitions as vector embeddings.
- Selects the top-K relevant nodes for your objective.
- Constructs a prompt for the LLM using only those nodes.
- Performs iterative validation (syntax and semantics) before returning the XML.

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

## Launcher Example

```bash
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=rag model:=gemini-2.5-flash mode:=api url:=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions key:=<API_KEY>
```
Or for Ollama local:
```bash
ros2 launch llm_bt_builder llm_agent.launch.py agent_type:=rag model:=llama3 mode:=api url:=http://localhost:11434/v1/chat/completions key:=""
```

## License

Apache License 2.0

## Author

Rodrigo Pérez-Rodríguez (rodrigo.perez@urjc.es)
