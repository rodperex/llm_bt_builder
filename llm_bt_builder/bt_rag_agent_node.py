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
import csv
import pathlib
import uuid
import yaml
import re
import os
import time
import json
from datetime import datetime, timezone
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
        self.declare_parameter('rag_top_k', 5)
        self.declare_parameter('rag', True)
        self.declare_parameter('metrics', False)

        self.llm_provider = self.get_parameter('llm_provider').value.lower()
        self.model_id = self.get_parameter('model_id').value
        self.api_url = self.get_parameter('api_url').value
        self.api_key = self.get_parameter('api_key').value
        self.embeddings_device = str(self.get_parameter('embeddings_device').value).strip().lower()
        self.rag_top_k = int(self.get_parameter('rag_top_k').value)
        self.rag_top_k = max(1, min(self.rag_top_k, 50))
        self.rag_enabled = bool(self.get_parameter('rag').value)
        self.metrics_enabled = bool(self.get_parameter('metrics').value)
        self._generation_metrics = None

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

    def _workspace_root(self):
        return pathlib.Path(__file__).resolve().parents[3]

    def _exec_root(self):
        exec_dir = self._workspace_root() / 'exec/btgen_metrics'
        exec_dir.mkdir(parents=True, exist_ok=True)
        return exec_dir

    def _begin_generation_metrics(self, request, is_fix: bool, pipeline: str = 'rag'):
        if not self.metrics_enabled or self._generation_metrics is not None:
            return

        self._generation_metrics = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'started_perf_sec': time.perf_counter(),
            'node_name': self.get_name(),
            'node_class': self.__class__.__name__,
            'pipeline': pipeline,
            'llm_provider': self.llm_provider,
            'model_id': self.model_id,
            'execution_mode': getattr(self, 'mode', ''),
            'metrics_enabled': True,
            'is_fix_request': bool(is_fix),
            'objective_chars': len(str(getattr(request, 'objective', '') or '')),
            'bt_nodes_yaml_chars': len(str(getattr(request, 'bt_nodes_yaml', '') or '')),
            'llm_calls_total': 0,
            'llm_calls_to_success': 0,
            'rag_enabled': bool(self.rag_enabled),
            'total_catalog_nodes': 0,
            'feedback_counts': {
                'syntax': 0,
                'structure': 0,
                'semantic': 0,
                'llm_error': 0,
                'other': 0,
            },
            'validation_failures': {
                'syntax': 0,
                'structure': 0,
                'semantic': 0,
            },
            'repeat_counts': {
                'structure_max': 0,
                'semantic_max': 0,
            },
            'feedback_trace': [],
            'rag_selected_nodes': 0,
            'bt_cleanup_collapsed_count': 0,
            'bt_cleanup_collapsed_tags': [],
            'success': False,
        }

    def _metric_note_llm_call(self):
        if self._generation_metrics is None:
            return
        self._generation_metrics['llm_calls_total'] += 1

    def _metric_note_feedback(self, feedback_type: str, message: str = ''):
        if self._generation_metrics is None:
            return

        feedback_type = feedback_type if feedback_type in self._generation_metrics['feedback_counts'] else 'other'
        self._generation_metrics['feedback_counts'][feedback_type] += 1
        self._generation_metrics['feedback_trace'].append(
            {
                'index': len(self._generation_metrics['feedback_trace']) + 1,
                'type': feedback_type,
                'message': self._truncate_for_log(message, 200),
            }
        )

    def _metric_note_validation_failure(self, validation_type: str):
        if self._generation_metrics is None:
            return
        if validation_type in self._generation_metrics['validation_failures']:
            self._generation_metrics['validation_failures'][validation_type] += 1

    def _metric_note_repeat_count(self, validation_type: str, count: int):
        if self._generation_metrics is None:
            return
        key = f'{validation_type}_max'
        if key in self._generation_metrics['repeat_counts']:
            self._generation_metrics['repeat_counts'][key] = max(self._generation_metrics['repeat_counts'][key], count)

    def _metric_note_rag_selected(self, count: int):
        if self._generation_metrics is None:
            return
        self._generation_metrics['rag_selected_nodes'] = count

    def _metric_note_total_catalog_nodes(self, count: int):
        if self._generation_metrics is None:
            return
        self._generation_metrics['total_catalog_nodes'] = count

    def _metric_merge(self, values: dict):
        if self._generation_metrics is None or not isinstance(values, dict):
            return
        self._generation_metrics.update(values)

    def _postprocess_generated_bt_xml(self, bt_xml: str, is_fix: bool):
        # Hook for subclasses to normalize/clean generated XML before metrics are finalized.
        return bt_xml, {}

    def _finalize_generation_metrics(self, response):
        if self._generation_metrics is None or not self.metrics_enabled:
            self._generation_metrics = None
            return

        metrics = dict(self._generation_metrics)
        metrics['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        metrics['duration_ms'] = round((time.perf_counter() - metrics['started_perf_sec']) * 1000.0, 3)
        metrics['success'] = bool(getattr(response, 'success', False))
        metrics['response_message'] = str(getattr(response, 'message', '') or '')
        metrics['bt_xml_chars'] = len(str(getattr(response, 'bt_xml', '') or ''))
        metrics['llm_calls_to_success'] = metrics['llm_calls_total'] if metrics['success'] else 0

        metrics.pop('started_perf_sec', None)

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        safe_model = str(self.model_id).replace('/', '_').replace(':', '_')
        safe_node = str(self.get_name()).replace('/', '_')
        json_path = self._exec_root() / f"btgen_metrics_{timestamp}_{safe_node}_{safe_model}_{uuid.uuid4().hex[:8]}.json"
        csv_path = self._exec_root() / 'bt_generation_metrics_summary.csv'

        with json_path.open('w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)

        csv_row = {
            'timestamp_utc': metrics['started_at_utc'],
            'node_name': metrics['node_name'],
            'node_class': metrics['node_class'],
            'pipeline': metrics['pipeline'],
            'llm_provider': metrics['llm_provider'],
            'model_id': metrics['model_id'],
            'execution_mode': metrics['execution_mode'],
            'metrics_enabled': metrics['metrics_enabled'],
            'is_fix_request': metrics['is_fix_request'],
            'rag_enabled': metrics['rag_enabled'],
            'success': metrics['success'],
            'duration_ms': metrics['duration_ms'],
            'llm_calls_total': metrics['llm_calls_total'],
            'llm_calls_to_success': metrics['llm_calls_to_success'],
            'objective_chars': metrics['objective_chars'],
            'bt_nodes_yaml_chars': metrics['bt_nodes_yaml_chars'],
            'bt_xml_chars': metrics['bt_xml_chars'],
            'rag_selected_nodes': metrics['rag_selected_nodes'],
            'total_catalog_nodes': metrics['total_catalog_nodes'],
            'feedback_counts_json': json.dumps(metrics['feedback_counts'], ensure_ascii=False),
            'validation_failures_json': json.dumps(metrics['validation_failures'], ensure_ascii=False),
            'repeat_counts_json': json.dumps(metrics['repeat_counts'], ensure_ascii=False),
            'feedback_trace_json': json.dumps(metrics['feedback_trace'], ensure_ascii=False),
            'response_message': metrics['response_message'],
            'metrics_json_path': str(json_path),
        }
        write_header = not csv_path.exists()
        with csv_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)

        self.get_logger().info(f"📊 BT metrics stored in: {json_path}")
        self._generation_metrics = None

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

        skills = data.get('skills_used', [])
        if not skills and isinstance(objective, dict):
            skills = objective.get('skills_used', [])
        for skill in skills:
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
        K = self.rag_top_k
        MAX_RETRIES = 25

        self._begin_generation_metrics(request, is_fix, pipeline='rag')

        if is_fix:
            self.get_logger().info(f"🔧 FIX BT Request received! Error to fix: '{request.error_message}'")
            self.get_logger().info(f"🧠 Original Objective: '{request.objective}'")
        else:
            self.get_logger().info(f"🎯 NEW BT Request received! Objective: '{request.objective}'")

        # 1. DATA PREPARATION
        full_node_specs = self.parse_full_specs(request.bt_nodes_yaml)
        total_catalog_nodes = len(full_node_specs)
        self._metric_note_total_catalog_nodes(total_catalog_nodes)
        known_bb_vars = self._extract_known_blackboard_vars(request.objective)
        known_bb_var_types = self._extract_known_blackboard_var_types(request.objective)
        required_output_vars = self._extract_required_output_vars(request.objective)
        recovery_policy = self._extract_recovery_policy(request.objective)
        allow_forced_plan_fail = self._extract_allow_forced_plan_fail(request.objective)

        # 2. RAG (Only done once at the beginning)
        vector_db = None
        if self.rag_enabled:
            vector_db = self.create_vector_store(request.bt_nodes_yaml)
            if not vector_db:
                response.success = False; response.message = "Error indexing YAML"
                self._finalize_generation_metrics(response)
                return response

            rag_query = self._sanitize_rag_query(request.objective)
            removed_mcp_context = len(rag_query) < len(request.objective)
            rag_queries, results = self._retrieve_relevant_nodes(vector_db, request.objective, K)

            # Ensure generic utility nodes are included if they exist in the robot's specs
            generic_names = ('speak', 'forceplanfail', 'saytext', 'abort')
            for node_name in full_node_specs.keys():
                if node_name.lower() in generic_names:
                    is_present = False
                    for res in results:
                        try:
                            res_name = yaml.safe_load(res.metadata.get('raw_yaml', '')).get('name')
                            if res_name == node_name:
                                is_present = True
                                break
                        except Exception:
                            pass
                    if not is_present:
                        exact_matches = vector_db.similarity_search(f"Tool: {node_name}", 1)
                        if exact_matches:
                            results.append(exact_matches[0])

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
        else:
            filtered_yaml_str = request.bt_nodes_yaml or "bt_nodes:\n"
            found_names = list(full_node_specs.keys())
            self.get_logger().info(
                f"🔎 RAG disabled: using full catalog ({len(found_names)} nodes) without retrieval filtering")

        self._metric_note_rag_selected(len(found_names))

        self.get_logger().info(f"🔎 RAG selected: {found_names}")

        # 3. PROMPT CONSTRUCTION
        raw_template = self.load_prompt_template()
        if not raw_template:
            response.success = False; response.message = "Prompt file missing"
            self._finalize_generation_metrics(response)
            return response
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
                self._metric_note_llm_call()
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
                    self._metric_note_feedback('syntax', xml_msg)
                    self._metric_note_validation_failure('syntax')
                    # Add to history so the LLM can self-correct
                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: Your XML syntax is invalid: {xml_msg}. Please fix tags and structure."))
                    time.sleep(5)
                    continue

                # B. BehaviorTree Structure Validation
                is_valid_structure, struct_msg, struct_hint = self.validate_xml_bt(xml_str)
                if not is_valid_structure:
                    self.get_logger().warn(f"⚠️ BT Structure Error: {struct_msg}")
                    self._metric_note_feedback('structure', struct_msg)
                    self._metric_note_validation_failure('structure')
                    repair_hint = struct_hint
                    if struct_msg == last_structure_error:
                        repeated_structure_error_count += 1
                    else:
                        last_structure_error = struct_msg
                        repeated_structure_error_count = 1
                    self._metric_note_repeat_count('structure', repeated_structure_error_count)

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
                        if vector_db is not None:
                            vector_db.delete_collection()
                        self._finalize_generation_metrics(response)
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
                    self._metric_note_feedback('semantic', bt_msg)
                    self._metric_note_validation_failure('semantic')
                    repair_hint = bt_hint
                    if bt_msg == last_semantic_error:
                        repeated_semantic_error_count += 1
                    else:
                        last_semantic_error = bt_msg
                        repeated_semantic_error_count = 1
                    self._metric_note_repeat_count('semantic', repeated_semantic_error_count)

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
                final_bt_xml, post_metrics = self._postprocess_generated_bt_xml(xml_str, is_fix)
                if isinstance(post_metrics, dict) and post_metrics:
                    self._metric_merge(post_metrics)

                response.bt_xml = final_bt_xml
                response.success = True
                response.message = f"RAG-({self.model_id})"
                self.get_logger().info("🎉 XML generated and VALIDATED successfully.")

                # Clean memory before exiting
                if vector_db is not None:
                    vector_db.delete_collection()
                self._finalize_generation_metrics(response)
                return response

            except Exception as e:
                error_str = str(e)
                self.get_logger().error(f"🔥 Error invoking LLM: {e}")
                self._metric_note_feedback('llm_error', error_str)
                # Respect retry_delay from 429 responses (e.g. Gemini free tier)
                import re as _re
                delay_match = _re.search(r'retry[_\s]delay[^0-9]*(\d+)', error_str, _re.IGNORECASE)
                delay = int(delay_match.group(1)) + 2 if delay_match else 5
                self.get_logger().info(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)

        # If we reach here, all attempts failed
        response.success = False
        response.message = "Max retries reached. Validation failed."
        if vector_db is not None:
            vector_db.delete_collection()
        self._finalize_generation_metrics(response)
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