# llm_bt_builder

Iterative Behavior Tree generator using Large Language Models (LLMs) for ROS 2 robots. Automatically creates Behavior Trees in XML format, using custom nodes defined in YAML and iterative reasoning with LLMs (local or API).

## Installation

### Requirements
- ROS 2 (Galactic or newer)
- Python 3.8+
- Dependencies:
  - rclpy
  - requests
  - PyYAML
  - torch (local mode only)
  - transformers (local mode only)
  - accelerate (local mode only)

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

### Launch the Node

You can launch the node directly:
```bash
ros2 run llm_bt_builder bt_agent_node.py
```

Or from a custom launch file.

### Configuration

The node accepts ROS 2 parameters:
- `execution_mode`: `local` (uses Hugging Face) or `api` (uses OpenAI, Groq, Ollama, LM Studio, etc.)
- `model_id`: Model ID (e.g., `Qwen/Qwen2.5-Coder-1.5B-Instruct`)
- `api_url`: REST endpoint URL (e.g., `http://localhost:11434/v1/chat/completions` for Ollama)
- `api_key`: API key (only for cloud services)

You can define custom nodes in the YAML file:
- `config/bt_nodes.yaml`

### ROS 2 Service

The node exposes the `generate_bt` service:
- **Request:**
  - `objective`: Objective in natural language
  - `bt_nodes_yaml`: YAML with node definitions
- **Response:**
  - `success`: Whether it was generated successfully
  - `bt_xml`: Behavior Tree XML
  - `message`: Status message

Example service call:
```bash
ros2 service call /generate_bt llm_bt_builder/srv/GenerateBT '{objective: "Follow the person and report results", bt_nodes_yaml: "$(cat config/bt_nodes.yaml)"}'
```

## Example Node YAML

```yaml
bt_nodes:
  - name: "SpeakToPerson"
    type: "Action"
    description: "Synthesizes speech. Parameters: text (string)"
  - name: "AskConfirmation"
    type: "Action"
    description: "Asks the person for confirmation and waits for 'yes' or 'no'. Returns SUCCESS if yes, FAILURE if no or no response."
  - name: "DetectPerson"
    type: "Condition"
    description: "Checks if a person is in front of the robot. Parameters: timeout (int)"
  - name: "TrackAndMeasureGait"
    type: "Action"
    description: "Follows the person and measures speed. Parameters: distance_meters (float)"
  - name: "ReportResults"
    type: "Action"
    description: "Speaks medical results. Parameters: metric_key (string)"
```

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
- **API:** Uses REST services (OpenAI, Groq, etc.), requires API key and URL.

## License

Apache License 2.0

## Author

Rodrigo Pérez-Rodríguez (rodrigo.perez@urjc.es)
