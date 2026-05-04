#!/usr/bin/env python3
# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from llm_bt_builder.srv import GenerateBT, FixBT
import yaml
import re
import os
import time
import xml.etree.ElementTree as ET

# --- LANGCHAIN & RAG IMPORTS ---
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print("❌ ERROR: Missing libraries. Please install requirements.txt and ensure all dependencies are met.")
    raise e

class RagBTAgent(Node):
    def __init__(self):
        super().__init__('llm_bt_rag_agent')
        self.get_logger().info(f"🛠️ Starting RAG Node...")

        # 1. PARAMETERS
        self.declare_parameter('llm_provider', 'gemini')  # gemini, openai, anthropic, ollama, deepseek
        self.declare_parameter('model_id', 'gemini-2.0-flash-lite')
        self.declare_parameter('api_url', '')
        self.declare_parameter('api_key', '')
        self.declare_parameter('prompt_file', 'system_prompt.txt')
        self.declare_parameter('embeddings_device', 'cpu')

        self.llm_provider = self.get_parameter('llm_provider').value.lower()
        self.model_id = self.get_parameter('model_id').value
        self.api_url = self.get_parameter('api_url').value
        self.api_key = self.get_parameter('api_key').value
        self.embeddings_device = str(self.get_parameter('embeddings_device').value).strip().lower()

        # API key detection based on provider
        param_key = self.get_parameter('api_key').value
        if param_key and param_key != "sk-no-key-needed":
            self.api_key = param_key
        else:
            # Map provider to environment variable
            provider_to_env = {
                'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
                'openai': ['OPENAI_API_KEY'],
                'anthropic': ['ANTHROPIC_API_KEY'],
                'deepseek': ['DEEPSEEK_API_KEY'],
                'ollama': ['LLM_API_KEY'],
                'groq': ['GROQ_API_KEY'],
                'sambanova': ['SAMBANOVA_API_KEY']
            }
            
            env_vars = provider_to_env.get(self.llm_provider, ['LLM_API_KEY'])
            for env_var in env_vars:
                self.api_key = os.getenv(env_var, '')
                if self.api_key:
                    break
            if not self.api_key:
                self.api_key = 'sk-no-key-needed'

        # 2. Load BT.CPP Node Categories from YAML files
        self.bt_control_nodes_yaml = self._load_bt_nodes_yaml('btv4_control_nodes.yaml')
        self.bt_decorator_nodes_yaml = self._load_bt_nodes_yaml('btv4_decorator_nodes.yaml')
        
        # Extract node names dynamically
        self.control_nodes = self._extract_node_names(self.bt_control_nodes_yaml)
        self.decorators = self._extract_node_names(self.bt_decorator_nodes_yaml)
        
        # Special nodes that don't require validation
        self.special_nodes = ['root', 'BehaviorTree', 'AlwaysSuccess', 'AlwaysFailure', 'SubTree']
        
        # All structural nodes (for semantic validation skip)
        self.structural_nodes = set(
            self.decorators + self.control_nodes + self.special_nodes
        )
        self.structural_required_ports = self._parse_structural_required_ports(
            self.bt_decorator_nodes_yaml, self.bt_control_nodes_yaml
        )

        # 3. SETUP
        self.llm = self.setup_llm()
        self.embeddings = self.setup_embeddings()

        # 4. SERVICE
        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
        self.fix_srv = self.create_service(FixBT, 'fix_bt', self.fix_bt_callback)
        self.get_logger().info(f"✅ RAG Agent ready. Provider: {self.llm_provider}, Model: {self.model_id}")

    def setup_llm(self):
        TIMEOUT = 120 
        try:
            # Use explicit provider parameter
            if self.llm_provider == 'gemini':
                self.get_logger().info("🔵 Configuring Gemini...")
                return ChatGoogleGenerativeAI(
                    model=self.model_id,
                    google_api_key=self.api_key,
                    temperature=0.1,
                    max_retries=2
                )
            elif self.llm_provider == 'anthropic':
                self.get_logger().info("🟣 Configuring Anthropic...")
                return ChatAnthropic(
                    model=self.model_id,
                    api_key=self.api_key,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2
                )
            elif self.llm_provider == 'openai':
                self.get_logger().info("🟢 Configuring OpenAI...")
                # LangChain needs base_url with /v1
                base_url = None
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url + '/v1'
                else:
                    base_url = 'https://api.openai.com/v1'
                
                return ChatOpenAI(
                    model=self.model_id,
                    api_key=self.api_key,
                    base_url=base_url,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2
                )
            elif self.llm_provider == 'deepseek':
                self.get_logger().info(f"🔷 Configuring DeepSeek ({self.model_id})...")
                # DeepSeek uses OpenAI-compatible API
                base_url = None
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url + '/v1'
                else:
                    base_url = "https://api.deepseek.com/v1"
                
                return ChatOpenAI(
                    model=self.model_id,
                    api_key=self.api_key,
                    base_url=base_url,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2
                )
            elif self.llm_provider == 'ollama':
                self.get_logger().info(f"🦙 Configuring Ollama ({self.model_id})...")
                # Ollama base URL (without /v1)
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                else:
                    base_url = "http://localhost:11434"
                
                return ChatOllama(
                    model=self.model_id,
                    base_url=base_url,
                    temperature=0.1,
                    timeout=TIMEOUT
                )
            elif self.llm_provider == 'groq':
                self.get_logger().info(f"⚡ Configuring Groq ({self.model_id})...")
                base_url = None
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url + '/v1'
                else:
                    base_url = "https://api.groq.com/openai/v1"
                
                return ChatOpenAI(
                    model=self.model_id,
                    api_key=self.api_key,
                    base_url=base_url,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2
                )
            elif self.llm_provider == 'sambanova':
                self.get_logger().info(f"🌐 Configuring SambaNova ({self.model_id})...")
                base_url = None
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url + '/v1'
                else:
                    base_url = "https://api.sambanova.ai/v1"
                
                return ChatOpenAI(
                    model=self.model_id,
                    api_key=self.api_key,
                    base_url=base_url,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2
                )
            else:
                self.get_logger().error(f"❌ Unknown provider: {self.llm_provider}")
                return None
        except Exception as e:
            self.get_logger().error(f"❌ Error setting up LLM: {e}")
            return None

    def setup_embeddings(self):
        self.get_logger().info("📥 Loading Embeddings (HuggingFace)...")

        requested_device = self.embeddings_device if self.embeddings_device else 'cpu'
        if requested_device == 'auto':
            requested_device = 'cpu'

        try:
            self.get_logger().info(f"📥 Embeddings device: {requested_device}")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": requested_device},
            )
        except Exception as e:
            if requested_device != 'cpu':
                self.get_logger().warn(
                    f"Embeddings initialization failed on '{requested_device}' ({e}). Falling back to CPU.")
                return HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )
            raise

    def _load_bt_nodes_yaml(self, filename):
        """Load BT.CPP standard nodes from YAML file"""
        try:
            pkg_path = get_package_share_directory('llm_bt_builder')
            yaml_path = os.path.join(pkg_path, 'config', filename)
            if not os.path.exists(yaml_path):
                # Fallback to local development path
                yaml_path = os.path.join(os.getcwd(), 'src', 'llm_bt_builder', 'config', filename)
            
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    return f.read()
            else:
                self.get_logger().warn(f"⚠️ Could not find {filename}")
                return ""
        except Exception as e:
            self.get_logger().error(f"❌ Error loading {filename}: {e}")
            return ""
    
    def _extract_node_names(self, yaml_content):
        """Extract node names from a YAML string"""
        try:
            if not yaml_content:
                return []
            data = yaml.safe_load(yaml_content)
            return [node['name'] for node in data.get('bt_nodes', [])]
        except Exception as e:
            self.get_logger().error(f"❌ Error extracting node names: {e}")
            return []

    def _parse_structural_required_ports(self, *yaml_contents):
        """Extract required port names for each structural (control/decorator) node."""
        required = {}
        for content in yaml_contents:
            try:
                data = yaml.safe_load(content)
                for node in data.get('bt_nodes', []):
                    ports = node.get('ports', []) or []
                    req = [p.get('name') or p.get('key')
                           for p in ports if p.get('name') or p.get('key')]
                    if req:
                        required[node['name']] = req
            except Exception:
                pass
        return required

    def load_prompt_template(self):
        try:
            prompt_file = self.get_parameter('prompt_file').value
            self.get_logger().info(f"📄 Loading prompt template from: {prompt_file}")
            # Try to load the prompt from the installed share directory
            pkg_path = get_package_share_directory('llm_bt_builder')
            prompt_path = os.path.join(pkg_path, 'prompts', prompt_file)
            if not os.path.exists(prompt_path):
                # Fallback to local development path
                prompt_path = os.path.join(os.getcwd(), 'src', 'llm_bt_builder', 'prompts', prompt_file)

            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f: return f.read()
            return None
        except Exception as e:
            self.get_logger().error(f"❌ Error reading prompt: {e}")
            return None

    def create_vector_store(self, yaml_content):
        try:
            # Split the YAML and create a temporary vector DB
            data = yaml.safe_load(yaml_content)
            documents = []
            for node in data.get('bt_nodes', []):
                search_content = f"Tool: {node['name']} Type: {node['type']} Desc: {node['description']}"
                node_yaml = yaml.dump(node, sort_keys=False)
                documents.append(Document(page_content=search_content, metadata={"raw_yaml": node_yaml}))
            return Chroma.from_documents(documents, self.embeddings, collection_name="temp_skills")
        except Exception as e:
            self.get_logger().error(f"❌ Error in Vector Store: {e}")
            return None

    def parse_full_specs(self, yaml_content):
        # Extract ALL valid nodes from the original YAML for final validation
        specs = {}
        try:
            data = yaml.safe_load(yaml_content)
            for node in data.get('bt_nodes', []):
                raw_ports = node.get('ports', [])
                current_ports = []
                required_inputs = []
                input_ports = set()
                output_ports = set()
                return_statuses = set()
                if raw_ports:
                    for p in raw_ports:
                        p_name = p.get('key') or p.get('name')
                        if not p_name:
                            continue
                        current_ports.append(p_name)

                        # Infer required input ports from node description metadata.
                        direction = str(p.get('direction', '')).lower()
                        description = str(p.get('description', '')).lower()
                        if direction == 'input':
                            input_ports.add(p_name)
                        elif direction == 'output':
                            output_ports.add(p_name)

                        if direction == 'input' and 'required' in description:
                            required_inputs.append(p_name)

                raw_returns = node.get('return', {})
                if isinstance(raw_returns, dict):
                    for status_name in raw_returns.keys():
                        if isinstance(status_name, str):
                            return_statuses.add(status_name.strip().upper())

                specs[node['name']] = {
                    'ports': current_ports,
                    'required_inputs': required_inputs,
                    'input_ports': input_ports,
                    'output_ports': output_ports,
                    'type': str(node.get('type', '')).strip().lower(),
                    'return_statuses': return_statuses,
                }
            return specs
        except: return {}

    def _extract_known_blackboard_vars(self, objective_text):
        """Collect blackboard vars that are readable at step start."""
        known = set()
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return known

        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Only inputs and available_blackboard_vars are readable at step start.
                    # Declared outputs must be produced by this step, not assumed pre-existing.
                    if k in ('available_blackboard_vars', 'inputs') and isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.strip():
                                known.add(item.strip())
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return known

    def _extract_required_output_vars(self, objective_text):
        """Collect blackboard vars that this step must write (declared outputs)."""
        required = set()
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return required

        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'outputs' and isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.strip():
                                required.add(item.strip())
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return required

    def _extract_recovery_policy(self, objective_text):
        """Read structured recovery policy from objective YAML (no keyword heuristics)."""
        policy = {
            'required': False,
            'loop_required': False,
            'retry_attempts': None,
        }
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return policy

        if not isinstance(data, dict):
            return policy

        # Preferred explicit contract, expected at objective.recovery_policy
        # but accepted recursively to keep compatibility with future schema moves.
        recovery_blocks = []

        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'recovery_policy' and isinstance(v, dict):
                        recovery_blocks.append(v)
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)

        for block in recovery_blocks:
            if bool(block.get('required', False)):
                policy['required'] = True
            if bool(block.get('loop_until_success', False)):
                policy['loop_required'] = True

            raw_retry = block.get('retry_attempts', None)
            if raw_retry is None:
                continue

            # Ignore booleans (bool is a subclass of int in Python).
            if isinstance(raw_retry, bool):
                continue

            parsed_retry = None
            if isinstance(raw_retry, (int, float)):
                parsed_retry = int(raw_retry)
            elif isinstance(raw_retry, str):
                value = raw_retry.strip().lower()
                if value in ('', 'null', 'none'):
                    parsed_retry = None
                elif value == 'forever':
                    parsed_retry = 'forever'
                else:
                    try:
                        parsed_retry = int(value)
                    except ValueError:
                        parsed_retry = None

            if parsed_retry == 'forever':
                policy['retry_attempts'] = 'forever'
            elif parsed_retry is not None and parsed_retry > 0:
                policy['retry_attempts'] = parsed_retry

        return policy

    def generate_bt_callback(self, request, response):
        return self._run_agentic_pipeline(request, response, is_fix=False)

    def fix_bt_callback(self, request, response):
        return self._run_agentic_pipeline(request, response, is_fix=True)

    def _run_agentic_pipeline(self, request, response, is_fix):
        K = 10
        MAX_RETRIES = 25

        if is_fix:
            self.get_logger().info(f"🔧 FIX BT Request received! Error to fix: '{request.error_message}'")
            self.get_logger().info(f"🧠 Original Objective: '{request.objective}'")
        else:
            self.get_logger().info(f"🎯 NEW BT Request received! Objective: '{request.objective}'")

        # 1. DATA PREPARATION
        full_node_specs = self.parse_full_specs(request.bt_nodes_yaml)
        known_bb_vars = self._extract_known_blackboard_vars(request.objective)
        required_output_vars = self._extract_required_output_vars(request.objective)
        recovery_policy = self._extract_recovery_policy(request.objective)

        # 2. RAG (Only done once at the beginning)
        vector_db = self.create_vector_store(request.bt_nodes_yaml)
        if not vector_db:
            response.success = False; response.message = "Error indexing YAML"; return response

        results = vector_db.similarity_search(request.objective, K)

        filtered_yaml_str = "bt_nodes:\n"
        found_names = []
        for res in results:
            raw_node = res.metadata['raw_yaml']
            filtered_yaml_str += "\n".join(["  " + line for line in raw_node.split('\n')]) + "\n"
            found_names.append(raw_node.splitlines()[0])

        self.get_logger().info(f"🔎 RAG selected: {found_names}")

        # 3. PROMPT CONSTRUCTION
        raw_template = self.load_prompt_template()
        if not raw_template:
            response.success = False; response.message = "Prompt file missing"; return response
        else:
            self.get_logger().debug(f"📄 Prompt template loaded successfully: {raw_template}")
        # Prepare BT.CPP standard nodes
        bt_standard_nodes = "## Control Nodes\n" + self.bt_control_nodes_yaml + "\n"
        bt_standard_nodes += "## Decorator Nodes\n" + self.bt_decorator_nodes_yaml

        system_content = raw_template.replace("{bt_standard_nodes}", bt_standard_nodes)
        system_content = system_content.replace("{robot_capabilities}", filtered_yaml_str)
        system_content = system_content.replace("{user_objective}", "")

        # Initialize chat history
        if is_fix:
            fix_prompt = (
                f"You previously generated an XML for this objective but it failed:\n"
                f"```xml\n{request.broken_bt_xml}\n```\n\n"
                f"It failed with this error:\n{request.error_message}\n\n"
                f"Please write a NEW, fixed XML that resolves this error and satisfies the objective: {request.objective}\n"
            )
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=fix_prompt)
            ]
        else:
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=request.objective)
            ]

        # 4. RETRY LOOP 🔄
        last_semantic_error = ""
        repeated_semantic_error_count = 0
        for attempt in range(MAX_RETRIES):
            self.get_logger().info(f"Attempt {attempt + 1}/{MAX_RETRIES}...")

            try:
                ai_msg = self.llm.invoke(messages)
                raw_response = ai_msg.content

                think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
                
                if think_match:
                    thought_process = think_match.group(1).strip()
                    self.get_logger().debug(f"\n🤔 CHAIN OF THOUGHT:\n\033[93m{thought_process}\033[0m\n")
                else:
                    self.get_logger().debug("⚠️ No <think> tags found in the response.")

                xml_str = self.extract_xml(ai_msg.content)

                # Add model name comment
                comment = f"\n  <!-- Generated by model: {self.model_id} -->"
                root_idx = xml_str.find("<root")
                if root_idx != -1:
                    end_idx = xml_str.find(">", root_idx) + 1
                    xml_str = xml_str[:end_idx] + comment + xml_str[end_idx:]

                # A. Syntactic Validation
                is_valid_xml, xml_msg = self.validate_xml_syntax(xml_str)
                if not is_valid_xml:
                    self.get_logger().warn(f"⚠️ XML Syntax Error: {xml_msg}")
                    # Add to history so the LLM can self-correct
                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: Your XML syntax is invalid: {xml_msg}. Please fix tags and structure."))
                    time.sleep(5)
                    continue

                # B. BehaviorTree Structure Validation
                is_valid_structure, struct_msg = self.validate_xml_bt(xml_str)
                if not is_valid_structure:
                    self.get_logger().warn(f"⚠️ BT Structure Error: {struct_msg}")
                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: BehaviorTree structure invalid: {struct_msg}. Fix the tree structure."))
                    time.sleep(1)
                    continue

                # C. Semantic Validation
                is_valid_bt, bt_msg = self.validate_bt_semantics(
                    xml_str,
                    full_node_specs,
                    known_bb_vars,
                    required_output_vars,
                    recovery_policy,
                )
                if not is_valid_bt:
                    self.get_logger().warn(f"⚠️ BT Semantic Error: {bt_msg}")
                    # Targeted feedback helps the model repair invalid blackboard bindings.
                    repair_hint = (
                        "Use only valid nodes/ports from capabilities and return a complete XML from scratch. "
                        "Before responding, run this internal checklist: "
                        "(1) every leaf node exists in capabilities, "
                        "(2) every attribute is an allowed port for that node, "
                        "(3) each blackboard attribute uses either one token {var} or a plain literal."
                    )
                    if "malformed blackboard reference" in bt_msg or "invalid blackboard key" in bt_msg:
                        repair_hint = (
                            "Use exactly ONE blackboard variable per attribute (e.g., text=\"{full_order}\"). "
                            "Do NOT use concatenations like \"{a},{b}\" or \"{x};{y}\". "
                            "Do NOT mix literals with blackboard placeholders in one attribute "
                            "(invalid: text=\"Chef, order is {a} and {b}\"). "
                            "If you need to say multiple variables, split into multiple nodes "
                            "(e.g., one Speak literal + one Speak per variable, or Ask/Extract to build a single variable first)."
                        )
                    elif "reads unknown blackboard key" in bt_msg:
                        repair_hint = (
                            "Only read variables declared in objective inputs/available_blackboard_vars, "
                            "or variables produced earlier in the same BT step. "
                            "If you need this value, either add the correct input key or write it before reading it. "
                            "Do not invent helper variables like combined_order unless a previous node writes them."
                        )
                    elif "did not write required output" in bt_msg:
                        repair_hint = (
                            "The step objective declares mandatory outputs. "
                            "Map the corresponding output port(s) to those exact blackboard keys before step completion."
                        )
                    elif "does NOT exist in the capabilities YAML" in bt_msg:
                        repair_hint = (
                            "You used a node that is not in capabilities. "
                            "Replace every unknown node with valid capabilities-only nodes and redesign the flow without helper/invented nodes. "
                            "If data transformation is needed, use only existing node outputs and objective-declared variables."
                        )
                    elif "recoverable checks" in bt_msg or "Recovery branch" in bt_msg:
                        repair_hint = (
                            "Recovery policy is declared in objective.recovery_policy. "
                            "Use explicit branching with success and recovery paths "
                            "(Fallback/ReactiveFallback), and include loop control (RetryUntilSuccessful/Repeat) "
                            "when loop_until_success is true. "
                            "If recovery_policy.retry_attempts is provided, set RetryUntilSuccessful num_attempts to that exact value."
                        )
                    elif "must not use RetryUntilSuccessful" in bt_msg:
                        repair_hint = (
                            "This step is not a recovery-loop step. Remove RetryUntilSuccessful and keep a plain Sequence "
                            "unless objective.recovery_policy.required=true or loop_until_success=true."
                        )
                    elif "can only return RUNNING" in bt_msg and "<Sequence>" in bt_msg:
                        repair_hint = (
                            "A node that only returns RUNNING cannot appear before required later siblings "
                            "inside a plain Sequence. Reorder so it is last, or wrap with ReactiveSequence/ReactiveFallback "
                            "so downstream checks/actions keep being ticked."
                        )
                    elif "can only return RUNNING" in bt_msg and "<ReactiveSequence>" in bt_msg:
                        repair_hint = (
                            "A node that only returns RUNNING inside a ReactiveSequence blocks all following siblings. "
                            "ReactiveSequence stops at the first RUNNING child and does not advance further. "
                            "If you want to keep ticking a condition while an action runs, use "
                            "ReactiveFallback with the condition FIRST: "
                            "<ReactiveFallback> <IsCondition/> <LongRunningAction/> </ReactiveFallback>."
                        )

                    if bt_msg == last_semantic_error:
                        repeated_semantic_error_count += 1
                    else:
                        last_semantic_error = bt_msg
                        repeated_semantic_error_count = 1

                    if repeated_semantic_error_count >= 2:
                        repair_hint += (
                            " You are repeating the same semantic error. "
                            "Discard the previous invalid structure and regenerate the BT from scratch, "
                            "strictly applying the 3-point checklist before returning XML."
                        )

                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: {bt_msg}. {repair_hint}"))
                    time.sleep(1)
                    continue # Next attempt

                # --- SUCCESS ---
                response.bt_xml = xml_str
                response.success = True
                response.message = f"RAG-({self.model_id})"
                self.get_logger().info("🎉 XML generated and VALIDATED successfully.")

                # Clean memory before exiting
                vector_db.delete_collection()
                return response

            except Exception as e:
                error_str = str(e)
                self.get_logger().error(f"🔥 Error invoking LLM: {e}")
                # Respect retry_delay from 429 responses (e.g. Gemini free tier)
                import re as _re
                delay_match = _re.search(r'retry[_\s]delay[^0-9]*(\d+)', error_str, _re.IGNORECASE)
                delay = int(delay_match.group(1)) + 2 if delay_match else 5
                self.get_logger().info(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)

        # If we reach here, all attempts failed
        response.success = False
        response.message = "Max retries reached. Validation failed."
        vector_db.delete_collection()
        return response

    def extract_xml(self, text):
        # Clean the LLM response to obtain only the XML
        match = re.search(r'```xml(.*?)```', text, re.DOTALL)
        if match: return match.group(1).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '<root' in text: return text[text.find('<root'):text.rfind('</root>')+7]
        return text

    def validate_xml_syntax(self, xml_string):
        # Check that the string is valid XML
        try:
            ET.fromstring(xml_string)
            return True, "OK"
        except ET.ParseError as e:
            return False, str(e)

    def validate_xml_bt(self, xml_string):
        """Validate BehaviorTree structural rules (decorators have 1 child, control nodes have children, etc.)"""
        try:
            root = ET.fromstring(xml_string)
            
            for elem in root.iter():
                children = list(elem)
                
                # Skip text/comments
                if not isinstance(elem.tag, str):
                    continue
                
                # Root and BehaviorTree should have exactly 1 child
                if elem.tag in ['root', 'BehaviorTree']:
                    if len(children) != 1:
                        return False, f"<{elem.tag}> must have exactly 1 child, found {len(children)}"
                
                # Decorators must have exactly 1 child
                elif elem.tag in self.decorators:
                    if len(children) != 1:
                        return False, f"Decorator <{elem.tag}> must have exactly 1 child, found {len(children)}"
                
                # Control nodes must have at least 1 child
                elif elem.tag in self.control_nodes:
                    if len(children) < 1:
                        return False, f"Control node <{elem.tag}> must have at least 1 child, found {len(children)}"
                
                # AlwaysSuccess/AlwaysFailure should have 0 children
                elif elem.tag in ['AlwaysSuccess', 'AlwaysFailure']:
                    if len(children) > 0:
                        return False, f"<{elem.tag}> should not have children, found {len(children)}"

                # Check required ports for structural nodes (e.g. num_attempts on RetryUntilSuccessful)
                for req_port in self.structural_required_ports.get(elem.tag, []):
                    if req_port not in elem.attrib:
                        return False, (f"<{elem.tag}> is missing required attribute '{req_port}'. "
                                       f"Add it, e.g. {req_port}=\"1\".")
            
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def validate_bt_semantics(self, xml_string, node_specs, known_bb_vars=None, required_outputs=None, recovery_policy=None):
        # Check that custom nodes exist in YAML and ports are correct
        # Note: Structural validation is done in validate_xml_bt
        try:
            root = ET.fromstring(xml_string)
            known_bb_vars = set(known_bb_vars or [])
            required_outputs = set(required_outputs or [])
            produced_in_tree = set()
            parent_map = {child: parent for parent in root.iter() for child in list(parent)}
            recovery_policy = recovery_policy or {'required': False, 'loop_required': False, 'retry_attempts': None}
            
            for elem in root.iter():
                # Skip structural BT.CPP nodes (using set for O(1) lookup)
                if elem.tag in self.structural_nodes:
                    continue
                    
                # Validate custom/action nodes from YAML
                if elem.tag not in node_specs:
                    return False, f"Node <{elem.tag}> does NOT exist in the capabilities YAML."

                spec = node_specs[elem.tag]
                allowed_ports = spec.get('ports', [])
                required_inputs = spec.get('required_inputs', [])
                input_ports = spec.get('input_ports', set())
                output_ports = spec.get('output_ports', set())

                # Validate required input ports are explicitly wired.
                missing_required = []
                for req in required_inputs:
                    if req not in elem.attrib or str(elem.attrib.get(req, '')).strip() == '':
                        missing_required.append(req)
                if missing_required:
                    return False, (
                        f"Node <{elem.tag}> is missing required input port(s): {missing_required}. "
                        f"Provide explicit values for those attributes."
                    )
                
                # Validate ports/attributes
                for attr in elem.attrib:
                    if attr in ['name', 'ID']:  # Structural attributes
                        continue
                    if attr not in allowed_ports:
                        return False, f"Node <{elem.tag}> has an illegal port: '{attr}'. Allowed: {allowed_ports}"

                    # Blackboard references must be exactly one {var} token, not concatenations.
                    value = str(elem.attrib[attr]).strip()
                    if '{' in value or '}' in value:
                        refs = re.findall(r'\{[^{}]+\}', value)
                        if len(refs) != 1 or refs[0] != value:
                            return False, (
                                f"Node <{elem.tag}> has malformed blackboard reference in port '{attr}': '{value}'. "
                                f"Use exactly one blackboard variable like '{{my_var}}', or a plain literal."
                            )

                        # Reject invalid BB keys like '{a},{b}' or '{a};{b}' interpreted as a single missing key.
                        bb_key = value[1:-1]
                        if ',' in bb_key or ';' in bb_key:
                            return False, (
                                f"Node <{elem.tag}> port '{attr}' uses an invalid blackboard key '{bb_key}'. "
                                f"Do not concatenate multiple variables inside one {{}}. "
                                f"Write to a single combined variable first, then pass that variable."
                            )

                        # Reads must refer to values available at step start or produced earlier in this BT.
                        # Use direction metadata when available; otherwise default to read for non-output ports.
                        is_read_ref = (attr in input_ports) or (attr not in output_ports)
                        if is_read_ref and bb_key not in known_bb_vars and bb_key not in produced_in_tree:
                            return False, (
                                f"Node <{elem.tag}> reads unknown blackboard key '{bb_key}' in input port '{attr}'. "
                                f"Declare it in objective inputs/available_blackboard_vars or write it earlier in this step."
                            )

                        if attr in output_ports:
                            produced_in_tree.add(bb_key)

                    elif attr in output_ports and value:
                        # Track literal outputs too, so later inputs can consume these keys when mapped.
                        literal_key = value.strip()
                        if literal_key.startswith('{') and literal_key.endswith('}'):
                            produced_in_tree.add(literal_key[1:-1])

            # Enforce declared inter-step contract: all required outputs must be produced.
            missing_outputs = required_outputs - produced_in_tree
            if missing_outputs:
                missing = sorted(missing_outputs)
                return False, (
                    f"BT did not write required output blackboard key(s): {missing}. "
                    f"Map node output ports to these exact keys."
                )

            loop_controls = {'RetryUntilSuccessful', 'Repeat'}
            has_loop = any(
                isinstance(elem.tag, str) and elem.tag in loop_controls
                for elem in root.iter()
            )

            retry_nodes = [
                elem for elem in root.iter()
                if isinstance(elem.tag, str) and elem.tag == 'RetryUntilSuccessful'
            ]
            retry_control_allowed = (
                recovery_policy.get('required', False) or
                recovery_policy.get('loop_required', False)
            )

            if not retry_control_allowed and retry_nodes:
                return False, (
                    "Objective recovery_policy has required=false and loop_until_success=false, "
                    "so BT must not use RetryUntilSuccessful for this step."
                )

            expected_retry_attempts = recovery_policy.get('retry_attempts', None)
            if retry_control_allowed and expected_retry_attempts is not None:
                if not retry_nodes:
                    return False, (
                        "Objective recovery_policy.retry_attempts is set for a retry-enabled step, but BT has no RetryUntilSuccessful node. "
                        "Use RetryUntilSuccessful with num_attempts matching recovery_policy.retry_attempts."
                    )

                has_expected_retry = False
                for retry_node in retry_nodes:
                    raw_num = str(retry_node.attrib.get('num_attempts', '')).strip()
                    try:
                        current = int(raw_num)
                    except ValueError:
                        continue

                    if expected_retry_attempts == 'forever' and current == -1:
                        has_expected_retry = True
                        break
                    if isinstance(expected_retry_attempts, int) and current == expected_retry_attempts:
                        has_expected_retry = True
                        break

                if not has_expected_retry:
                    expected_value = '-1' if expected_retry_attempts == 'forever' else str(expected_retry_attempts)
                    return False, (
                        "Objective recovery_policy.retry_attempts requires "
                        f"RetryUntilSuccessful num_attempts=\"{expected_value}\", "
                        "but BT uses a different value."
                    )

            # Recoverable-check policy is enforced ONLY when explicitly declared
            # via objective.recovery_policy.required (no keyword heuristics).
            if recovery_policy.get('required', False):
                condition_tags = {
                    name for name, spec in node_specs.items()
                    if spec.get('type') == 'condition'
                }
                condition_nodes = [
                    elem for elem in root.iter()
                    if isinstance(elem.tag, str) and elem.tag in condition_tags
                ]

                if condition_nodes:
                    if recovery_policy.get('loop_required', False) and not has_loop:
                        return False, (
                            "Objective recovery_policy requires loop_until_success, but BT has no retry loop control. "
                            "Wrap check+recovery logic with RetryUntilSuccessful (or Repeat)."
                        )

                    branching_controls = {'Fallback', 'ReactiveFallback'}

                    def has_branching_ancestor(node):
                        parent = parent_map.get(node)
                        while parent is not None:
                            if isinstance(parent.tag, str) and parent.tag in branching_controls:
                                return True
                            parent = parent_map.get(parent)
                        return False

                    for cond in condition_nodes:
                        if not has_branching_ancestor(cond):
                            return False, (
                                f"Recovery branch missing for condition <{cond.tag}>. "
                                "For recoverable checks, place conditions under Fallback/ReactiveFallback "
                                "with an explicit failure-recovery branch."
                            )

            # Progress-safety check: an always-RUNNING leaf in ordered short-circuit controls
            # blocks later siblings from being ticked.
            # ReactiveSequence is included: it also stops advancing past a RUNNING child.
            blocking_controls = ('Sequence', 'Fallback', 'ReactiveSequence')
            for control_tag in blocking_controls:
                for control in root.iter(control_tag):
                    children = [c for c in list(control) if isinstance(c.tag, str)]
                    for child in children[:-1]:
                        # Only leaf capability nodes are checked here; control/decorator nodes
                        # require deeper control-flow analysis and are handled by prompt guidance.
                        if child.tag in self.structural_nodes:
                            continue
                        spec = node_specs.get(child.tag)
                        if not spec:
                            continue
                        statuses = set(spec.get('return_statuses', set()))
                        if statuses == {'RUNNING'}:
                            return False, (
                                f"Node <{child.tag}> can only return RUNNING and appears before other "
                                f"children inside <{control_tag}>, so later nodes are unreachable. "
                                "Move it to the end of that control node or redesign with non-blocking flow."
                            )

            return True, "OK"
        except Exception as e:
            return False, str(e)

def main(args=None):
    rclpy.init(args=args)
    node = RagBTAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # Launch may have already shut down the global context.
            pass

if __name__ == '__main__':
    main()