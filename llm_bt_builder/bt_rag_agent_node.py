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
try:
    from llm_bt_builder.bt_validation import BTValidation
except ModuleNotFoundError:
    from bt_validation import BTValidation

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

class RagBTAgent(BTValidation, Node):
    def __init__(self):
        super().__init__('llm_bt_rag_agent')
        self.get_logger().info(f"🛠️ Starting RAG Node...")

        # 1. PARAMETERS
        self.declare_parameter('llm_provider', 'gemini')  # gemini, openai, anthropic, ollama, deepseek, groq, sambanova, cerebras
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
                'sambanova': ['SAMBANOVA_API_KEY'],
                'cerebras': ['CEREBRAS_API_KEY']
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
            elif self.llm_provider == 'cerebras':
                self.get_logger().info(f"🧠 Configuring Cerebras Cloud ({self.model_id})...")
                base_url = None
                if self.api_url and self.api_url != '':
                    base_url = self.api_url.rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url + '/v1'
                else:
                    base_url = "https://api.cerebras.ai/v1"
                
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
        return super()._parse_structural_required_ports(*yaml_contents)

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
                ports = []
                for port in node.get('ports', []):
                    if not isinstance(port, dict):
                        continue
                    ports.append(
                        f"Port: {port.get('name', '')} Dir: {port.get('direction', '')} "
                        f"Type: {port.get('type', '')} Desc: {port.get('description', '')}"
                    )

                returns = []
                for status, description in node.get('return', {}).items():
                    returns.append(f"Return: {status} Desc: {description}")

                search_content = (
                    f"Tool: {node['name']} Type: {node['type']} Desc: {node['description']}\n"
                    f"Ports: {' | '.join(ports)}\n"
                    f"Returns: {' | '.join(returns)}"
                )
                node_yaml = yaml.dump(node, sort_keys=False)
                documents.append(Document(page_content=search_content, metadata={"raw_yaml": node_yaml}))
            return Chroma.from_documents(documents, self.embeddings, collection_name="temp_skills")
        except Exception as e:
            self.get_logger().error(f"❌ Error in Vector Store: {e}")
            return None

    def parse_full_specs(self, yaml_content):
        return self._parse_capability_specs(yaml_content)

    def _sanitize_rag_query(self, objective_text):
        """Remove non-step MCP runtime context from retrieval query."""
        if not isinstance(objective_text, str):
            return str(objective_text)

        # Preferred path: structured YAML key injected by MCPRagBTAgent.
        try:
            parsed = yaml.safe_load(objective_text)
            if isinstance(parsed, dict) and 'mcp_context' in parsed:
                parsed.pop('mcp_context', None)
                cleaned = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=False)
                return cleaned.strip()
        except Exception:
            pass

        # MCP context is appended as a read-only runtime block for prompting,
        # but it is not useful for capability retrieval.
        marker = "\n# MCP_CONTEXT"
        idx = objective_text.find(marker)
        if idx != -1:
            return objective_text[:idx].rstrip()
        return objective_text.strip()

    def _build_rag_queries(self, objective_text):
        """Build multiple focused retrieval queries from the structured objective."""
        sanitized = self._sanitize_rag_query(objective_text)
        queries = []

        def add_query(text):
            if not isinstance(text, str):
                return
            normalized = " ".join(text.split()).strip()
            if normalized and normalized not in queries:
                queries.append(normalized)

        add_query(sanitized)

        try:
            data = yaml.safe_load(sanitized)
        except Exception:
            return queries

        if not isinstance(data, dict):
            return queries

        objective = data.get('objective', data)
        if not isinstance(objective, dict):
            return queries

        add_query(objective.get('description', ''))

        for skill in objective.get('skills_used', []):
            add_query(str(skill))

        for entry in objective.get('steps', []):
            if isinstance(entry, dict):
                add_query(str(entry.get('step', '')))
            elif isinstance(entry, str):
                add_query(entry)

        return queries

    def _retrieve_relevant_nodes(self, vector_db, objective_text, top_k):
        queries = self._build_rag_queries(objective_text)
        selected = []
        seen_raw_yaml = set()

        for query in queries:
            for result in vector_db.similarity_search(query, top_k):
                raw_yaml = result.metadata.get('raw_yaml', '')
                if raw_yaml and raw_yaml not in seen_raw_yaml:
                    seen_raw_yaml.add(raw_yaml)
                    selected.append(result)

        return queries, selected

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

    def _extract_allow_forced_plan_fail(self, objective_text):
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return False

        found_values = []

        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'allow_forced_plan_fail':
                        found_values.append(bool(v))
                    collect(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return any(found_values)

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
        known_bb_var_types = self._extract_known_blackboard_var_types(request.objective)
        required_output_vars = self._extract_required_output_vars(request.objective)
        recovery_policy = self._extract_recovery_policy(request.objective)
        allow_forced_plan_fail = self._extract_allow_forced_plan_fail(request.objective)

        # 2. RAG (Only done once at the beginning)
        vector_db = self.create_vector_store(request.bt_nodes_yaml)
        if not vector_db:
            response.success = False; response.message = "Error indexing YAML"; return response

        rag_query = self._sanitize_rag_query(request.objective)
        removed_mcp_context = len(rag_query) < len(request.objective)
        rag_queries, results = self._retrieve_relevant_nodes(vector_db, request.objective, K)
        rag_log = (
            "\n========== RAG INPUT START =========="
            f"\nK: {K}"
            "\nRetrieval query source: aggregated objective queries"
            f"\nMCP context removed: {removed_mcp_context}"
            f"\nOriginal chars: {len(request.objective)} | Query chars: {len(rag_query)}"
            f"\nPrimary retrieval query:\n{rag_query}"
            f"\nFocused queries ({len(rag_queries)}):\n- {'\n- '.join(rag_queries)}"
            "\n=========== RAG INPUT END ==========="
        )
        self.get_logger().info(rag_log)

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
        last_structure_error = ""
        repeated_structure_error_count = 0
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
                is_valid_structure, struct_msg, struct_hint = self.validate_xml_bt(xml_str)
                if not is_valid_structure:
                    self.get_logger().warn(f"⚠️ BT Structure Error: {struct_msg}")
                    repair_hint = struct_hint
                    if struct_msg == last_structure_error:
                        repeated_structure_error_count += 1
                    else:
                        last_structure_error = struct_msg
                        repeated_structure_error_count = 1

                    if repeated_structure_error_count >= 2:
                        repair_hint += (
                            " You are repeating the same structural error. "
                            "Discard the previous tree and rebuild the entire BT skeleton first "
                            "(root -> BehaviorTree -> control flow with valid arity), then fill leaf nodes."
                        )

                    if repeated_structure_error_count >= 5:
                        self.get_logger().error(
                            "❌ Aborting early: repeated identical BT structure error 5 times."
                        )
                        response.success = False
                        response.message = (
                            "Repeated BT structure error (x5): "
                            f"{struct_msg}"
                        )
                        vector_db.delete_collection()
                        return response

                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: BehaviorTree structure invalid: {struct_msg}. {repair_hint}"))
                    time.sleep(1)
                    continue

                # C. Semantic Validation
                is_valid_bt, bt_msg, bt_hint = self.validate_bt_semantics(
                    xml_str,
                    full_node_specs,
                    known_bb_vars,
                    known_bb_var_types,
                    required_output_vars,
                    recovery_policy,
                    allow_forced_plan_fail,
                )
                if not is_valid_bt:
                    self.get_logger().warn(f"⚠️ BT Semantic Error: {bt_msg}")
                    repair_hint = bt_hint
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
        return super().validate_xml_bt(xml_string)

    def validate_bt_semantics(
        self,
        xml_string,
        node_specs,
        known_bb_vars=None,
        known_bb_var_types=None,
        required_outputs=None,
        recovery_policy=None,
        allow_forced_plan_fail=None,
    ):
        return super().validate_bt_semantics(
            xml_string,
            node_specs,
            known_bb_vars,
            known_bb_var_types,
            required_outputs,
            recovery_policy,
            allow_forced_plan_fail,
        )

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