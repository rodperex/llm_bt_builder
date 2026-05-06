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

# ─────────────────────────────────────────────────────────────────────────────
# bt_rag_agentic_node.py
#
# Fully agentic BT builder: the LLM autonomously decides when to call each
# validation tool (validate_xml_syntax → validate_bt_structure →
# validate_bt_semantics → submit_bt_xml) through LangChain tool-calling.
#
# Architecture:
#   RAG (ChromaDB)  →  LLM + bound tools  →  agentic tool-call loop
#
# Differences from bt_rag_agent_node.py:
#   - Uses LangChain bind_tools() instead of a hard-coded retry loop.
#   - Validations are exposed as tools; the LLM decides the order and pace.
#   - submit_bt_xml acts as the termination signal.
#   - A final programmatic safety check is applied before accepting the result.
#
# NOTE: Tool-calling requires a model that supports it (Gemini 1.5+, GPT-4o,
#       Claude 3+, DeepSeek-v2+). Some Ollama models may not support it.
# ─────────────────────────────────────────────────────────────────────────────

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

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from pydantic import BaseModel
except ImportError as e:
    print("❌ ERROR: Missing libraries. Please install requirements.txt.")
    raise e


# ── Pydantic schema shared by all tools ──────────────────────────────────────

class XMLArg(BaseModel):
    xml: str


# ── ROS 2 Node ────────────────────────────────────────────────────────────────

class AgenticBTNode(BTValidation, Node):

    def __init__(self):
        super().__init__('llm_bt_agentic')
        self.get_logger().info("🛠️  Starting Agentic BT Node...")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('llm_provider', 'gemini')
        self.declare_parameter('model_id', 'gemini-2.0-flash')
        self.declare_parameter('api_url', '')
        self.declare_parameter('api_key', '')
        self.declare_parameter('prompt_file', 'system_prompt.txt')

        self.llm_provider = self.get_parameter('llm_provider').value.lower()
        self.model_id     = self.get_parameter('model_id').value
        self.api_url      = self.get_parameter('api_url').value

        # Smart API-key detection (env vars as fallback)
        param_key = self.get_parameter('api_key').value
        if param_key and param_key != 'sk-no-key-needed':
            self.api_key = param_key
        else:
            provider_to_env = {
                'gemini':    ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
                'openai':    ['OPENAI_API_KEY'],
                'anthropic': ['ANTHROPIC_API_KEY'],
                'deepseek':  ['DEEPSEEK_API_KEY'],
                'ollama':    ['LLM_API_KEY'],
            }
            for ev in provider_to_env.get(self.llm_provider, ['LLM_API_KEY']):
                self.api_key = os.getenv(ev, '')
                if self.api_key:
                    break
            if not self.api_key:
                self.api_key = 'sk-no-key-needed'

        # ── BT.CPP structural node definitions ────────────────────────────────
        self.bt_control_nodes_yaml   = self._load_bt_nodes_yaml('btv4_control_nodes.yaml')
        self.bt_decorator_nodes_yaml = self._load_bt_nodes_yaml('btv4_decorator_nodes.yaml')
        self.control_nodes  = self._extract_node_names(self.bt_control_nodes_yaml)
        self.decorators     = self._extract_node_names(self.bt_decorator_nodes_yaml)
        self.special_nodes  = ['root', 'BehaviorTree', 'AlwaysSuccess', 'AlwaysFailure', 'SubTree']
        self.structural_nodes = set(self.decorators + self.control_nodes + self.special_nodes)
        self.structural_required_ports = self._parse_structural_required_ports(
            self.bt_decorator_nodes_yaml, self.bt_control_nodes_yaml
        )

        # ── LLM & embeddings ──────────────────────────────────────────────────
        self.llm        = self._setup_llm()
        self.embeddings = self._setup_embeddings()

        # ── Service ───────────────────────────────────────────────────────────
        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
        self.fix_srv = self.create_service(FixBT, 'fix_bt', self.fix_bt_callback)
        self.get_logger().info(
            f"✅ Agentic Node ready. Provider: {self.llm_provider}, Model: {self.model_id}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_llm(self):
        TIMEOUT = 120
        try:
            if self.llm_provider == 'gemini':
                self.get_logger().info("🔵 Configuring Gemini...")
                return ChatGoogleGenerativeAI(
                    model=self.model_id,
                    google_api_key=self.api_key,
                    temperature=0.1,
                    max_retries=2,
                )
            elif self.llm_provider == 'anthropic':
                self.get_logger().info("🟣 Configuring Anthropic...")
                return ChatAnthropic(
                    model=self.model_id,
                    api_key=self.api_key,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2,
                )
            elif self.llm_provider in ('openai', 'deepseek'):
                icon = "🟢" if self.llm_provider == 'openai' else "🔷"
                self.get_logger().info(f"{icon} Configuring {self.llm_provider}...")
                default = ('https://api.openai.com/v1' if self.llm_provider == 'openai'
                           else 'https://api.deepseek.com/v1')
                base = self.api_url.rstrip('/') if self.api_url else default
                if not base.endswith('/v1'):
                    base += '/v1'
                return ChatOpenAI(
                    model=self.model_id,
                    api_key=self.api_key,
                    base_url=base,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=TIMEOUT,
                    max_retries=2,
                )
            elif self.llm_provider == 'ollama':
                self.get_logger().info(f"🦙 Configuring Ollama ({self.model_id})...")
                base = self.api_url.rstrip('/') if self.api_url else 'http://localhost:11434'
                return ChatOllama(
                    model=self.model_id,
                    base_url=base,
                    temperature=0.1,
                    timeout=TIMEOUT,
                )
            else:
                self.get_logger().error(f"❌ Unknown provider: {self.llm_provider}")
                return None
        except Exception as e:
            self.get_logger().error(f"❌ Error setting up LLM: {e}")
            return None

    def _setup_embeddings(self):
        self.get_logger().info("📥 Loading Embeddings (HuggingFace all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ─────────────────────────────────────────────────────────────────────────
    # TOOL FACTORY
    # Each call to _make_tools() creates a fresh set of closures that capture
    # the per-request node_specs so they are safe across concurrent requests.
    # ─────────────────────────────────────────────────────────────────────────

    def _make_tools(self, full_node_specs, known_bb_vars, required_output_vars, recovery_policy):

        # Mutable state shared between submit_bt_xml closure and the caller
        submit_state = {"xml": None, "done": False}

        # ── Tool 1: XML syntax ─────────────────────────────────────────────
        def validate_xml_syntax(xml: str) -> str:
            """Validates that the XML string is well-formed.
            Call this FIRST after generating your XML."""
            try:
                ET.fromstring(xml)
                return "VALID: XML syntax is correct."
            except ET.ParseError as e:
                return f"ERROR: Invalid XML syntax — {e}. Fix the tags and call this tool again."

        # ── Tool 2: BT structure ───────────────────────────────────────────
        def validate_bt_structure(xml: str) -> str:
            """Validates BehaviorTree structural rules:
            - <root> and <BehaviorTree> must have exactly 1 child.
            - Decorator nodes must have exactly 1 child.
            - Control nodes must have at least 1 child.
            - AlwaysSuccess / AlwaysFailure must have 0 children.
            Call this after validate_xml_syntax returns VALID."""
            ok, msg = self.validate_xml_bt(xml)
            if ok:
                return "VALID: BehaviorTree structure is correct."
            return f"ERROR: {msg}"

        # ── Tool 3: BT semantics ───────────────────────────────────────────
        def validate_bt_semantics(xml: str) -> str:
            """Validates that every custom node in the XML exists in the
            capabilities YAML and only uses declared ports.
            Call this after validate_bt_structure returns VALID."""
            try:
                ok, msg = self.validate_bt_semantics(
                    xml,
                    full_node_specs,
                    known_bb_vars,
                    required_output_vars,
                    recovery_policy,
                )
                if ok:
                    return "VALID: BT semantics are correct."
                return f"ERROR: {msg}"

        # ── Tool 4: submit ─────────────────────────────────────────────────
        def submit_bt_xml(xml: str) -> str:
            """Submit the final BehaviorTree XML.
            Call this ONLY when ALL three validation tools returned VALID."""
            submit_state["xml"]  = xml
            submit_state["done"] = True
            return "XML received — running final safety check..."

        tools = [
            StructuredTool.from_function(
                func=validate_xml_syntax,
                name="validate_xml_syntax",
                description=(
                    "Validates that the XML string is well-formed (syntax check). "
                    "Call this FIRST after generating XML."
                ),
                args_schema=XMLArg,
            ),
            StructuredTool.from_function(
                func=validate_bt_structure,
                name="validate_bt_structure",
                description=(
                    "Validates BehaviorTree structural rules (children counts). "
                    "Call this after validate_xml_syntax returns VALID."
                ),
                args_schema=XMLArg,
            ),
            StructuredTool.from_function(
                func=validate_bt_semantics,
                name="validate_bt_semantics",
                description=(
                    "Validates nodes and ports against the robot capabilities YAML. "
                    "Call this after validate_bt_structure returns VALID."
                ),
                args_schema=XMLArg,
            ),
            StructuredTool.from_function(
                func=submit_bt_xml,
                name="submit_bt_xml",
                description=(
                    "Submit the final XML. Call this ONLY when ALL three validations "
                    "returned VALID."
                ),
                args_schema=XMLArg,
            ),
        ]
        return tools, submit_state

    # ─────────────────────────────────────────────────────────────────────────
    # RAG HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _create_vector_store(self, yaml_content):
        try:
            data = yaml.safe_load(yaml_content)
            documents = []
            for node in data.get('bt_nodes', []):
                content  = f"Tool: {node['name']} Type: {node['type']} Desc: {node['description']}"
                node_yaml = yaml.dump(node, sort_keys=False)
                documents.append(Document(page_content=content, metadata={"raw_yaml": node_yaml}))
            return Chroma.from_documents(documents, self.embeddings,
                                         collection_name="agentic_skills")
        except Exception as e:
            self.get_logger().error(f"❌ Error in Vector Store: {e}")
            return None

    def _parse_full_specs(self, yaml_content):
        return self._parse_capability_specs(yaml_content)

    def _load_prompt_template(self):
        try:
            prompt_file = self.get_parameter('prompt_file').value
            pkg_path    = get_package_share_directory('llm_bt_builder')
            path = os.path.join(pkg_path, 'prompts', prompt_file)
            if not os.path.exists(path):
                path = os.path.join(os.getcwd(), 'src', 'llm_bt_builder', 'prompts', prompt_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read()
        except Exception as e:
            self.get_logger().error(f"❌ Error reading prompt: {e}")
        return None

    def _load_bt_nodes_yaml(self, filename):
        try:
            pkg_path = get_package_share_directory('llm_bt_builder')
            path = os.path.join(pkg_path, 'config', filename)
            if not os.path.exists(path):
                path = os.path.join(os.getcwd(), 'src', 'llm_bt_builder', 'config', filename)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read()
        except Exception as e:
            self.get_logger().error(f"❌ Error loading {filename}: {e}")
        return ""

    def _extract_node_names(self, yaml_content):
        try:
            data = yaml.safe_load(yaml_content)
            return [n['name'] for n in data.get('bt_nodes', [])]
        except Exception:
            return []

    def _parse_structural_required_ports(self, *yaml_contents):
        return super()._parse_structural_required_ports(*yaml_contents)

    def _extract_recovery_policy(self, objective_text):
        """Read optional objective.recovery_policy without keyword heuristics."""
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
            if raw_retry is None or isinstance(raw_retry, bool):
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

    def _extract_xml(self, text):
        match = re.search(r'```xml(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '<root' in text:
            return text[text.find('<root'):text.rfind('</root>') + 7]
        return text.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SAFETY CHECK (programmatic, runs after submit_bt_xml)
    # ─────────────────────────────────────────────────────────────────────────

    def _final_validate(self, xml_str, node_specs, known_bb_vars, required_output_vars, recovery_policy):
        # Phase 1 — Syntax
        try:
            ET.fromstring(xml_str)
        except ET.ParseError as e:
            return False, f"Syntax error: {e}"
        ok, err = self.validate_xml_bt(xml_str)
        if not ok:
            return False, err
        return self.validate_bt_semantics(
            xml_str,
            node_specs,
            known_bb_vars,
            required_output_vars,
            recovery_policy,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE CALLBACK — agentic loop
    # ─────────────────────────────────────────────────────────────────────────

    def generate_bt_callback(self, request, response):
        return self._run_agentic_pipeline(request, response, is_fix=False)

    def fix_bt_callback(self, request, response):
        return self._run_agentic_pipeline(request, response, is_fix=True)

    def _run_agentic_pipeline(self, request, response, is_fix):
        K         = 10
        MAX_STEPS = 40   # upper bound on total tool-call steps

        if is_fix:
            self.get_logger().info(f"🔧 FIX BT Request received! Error to fix: '{request.error_message}'")
            self.get_logger().info(f"🎯 Original Objective: '{request.objective}'")
        else:
            self.get_logger().info(f"🎯 NEW Agentic BT Request: '{request.objective}'")

        # 1. Parse full node specs for final validation
        full_node_specs = self._parse_full_specs(request.bt_nodes_yaml)
        known_bb_vars = self._extract_known_blackboard_vars(request.objective)
        required_output_vars = self._extract_required_output_vars(request.objective)
        recovery_policy = self._extract_recovery_policy(request.objective)

        # 2. RAG — retrieve top-K semantically relevant nodes
        vector_db = self._create_vector_store(request.bt_nodes_yaml)
        if not vector_db:
            response.success = False
            response.message = "Error indexing YAML"
            return response

        results       = vector_db.similarity_search(request.objective, K)
        filtered_yaml = "bt_nodes:\n"
        found_names   = []
        for res in results:
            raw = res.metadata['raw_yaml']
            filtered_yaml += "\n".join("  " + line for line in raw.split('\n')) + "\n"
            found_names.append(raw.splitlines()[0])
        self.get_logger().info(f"🔎 RAG selected: {found_names}")

        # 3. Build system prompt
        raw_template = self._load_prompt_template()
        if not raw_template:
            response.success = False
            response.message = "Prompt file missing"
            vector_db.delete_collection()
            return response

        bt_std = ("## Control Nodes\n" + self.bt_control_nodes_yaml +
                  "\n## Decorator Nodes\n" + self.bt_decorator_nodes_yaml)

        system_content = (
            raw_template
            .replace("{bt_standard_nodes}", bt_std)
            .replace("{robot_capabilities}", filtered_yaml)
            .replace("{user_objective}", "")
        )

        agentic_addendum = (
            "\n\n## HOW TO USE YOUR TOOLS\n"
            "You have 4 tools. Follow this exact pipeline:\n"
            "1. Generate the BehaviorTree XML.\n"
            "2. Call `validate_xml_syntax` with your XML.\n"
            "   → If ERROR: fix the XML and call it again.\n"
            "3. Call `validate_bt_structure` with your XML.\n"
            "   → If ERROR: fix the XML and re-validate from step 2.\n"
            "4. Call `validate_bt_semantics` with your XML.\n"
            "   → If ERROR: fix the XML and re-validate from step 2.\n"
            "5. Only when ALL THREE return VALID, call `submit_bt_xml`.\n"
            "Do NOT call `submit_bt_xml` until every validation returns VALID."
        )

        # 4. Build tools for this request and bind to LLM
        tools, submit_state = self._make_tools(
            full_node_specs,
            known_bb_vars,
            required_output_vars,
            recovery_policy,
        )
        llm_with_tools      = self.llm.bind_tools(tools)
        tool_map            = {t.name: t for t in tools}

        if is_fix:
            fix_prompt = (
                f"You previously generated an XML for this objective but it failed:\n"
                f"```xml\n{request.broken_bt_xml}\n```\n\n"
                f"It failed with this error:\n{request.error_message}\n\n"
                f"Please write a NEW, fixed XML that resolves this error and satisfies the objective: {request.objective}\n"
            )
            messages = [
                SystemMessage(content=system_content + agentic_addendum),
                HumanMessage(content=fix_prompt),
            ]
        else:
            messages = [
                SystemMessage(content=system_content + agentic_addendum),
                HumanMessage(content=request.objective),
            ]

        # 5. Agentic loop ──────────────────────────────────────────────────
        for step in range(MAX_STEPS):
            self.get_logger().info(f"⚙️  Step {step + 1}/{MAX_STEPS}")

            try:
                ai_msg = llm_with_tools.invoke(messages)
            except Exception as e:
                self.get_logger().error(f"🔥 LLM error: {e}")
                time.sleep(2)
                continue

            # Log chain-of-thought if model emits <think> tags
            if ai_msg.content:
                think = re.search(r'<think>(.*?)</think>', ai_msg.content, re.DOTALL)
                if think:
                    self.get_logger().debug(
                        f"\n🤔 CHAIN OF THOUGHT:\n\033[93m{think.group(1).strip()}\033[0m\n"
                    )

            messages.append(ai_msg)

            # ── Handle tool calls ──────────────────────────────────────────
            if ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    tool_name = tc['name']
                    tool_args = tc['args']
                    self.get_logger().info(f"🔧 Tool called: {tool_name}")

                    if tool_name not in tool_map:
                        result = f"ERROR: Unknown tool '{tool_name}'."
                    else:
                        try:
                            result = tool_map[tool_name].invoke(tool_args)
                        except Exception as e:
                            result = f"ERROR executing {tool_name}: {e}"

                    self.get_logger().info(f"   ↳ {result}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc['id']))

                    # ── submit_bt_xml intercepted ──────────────────────────
                    if submit_state["done"]:
                        xml_str  = self._extract_xml(submit_state["xml"])

                        # Add model name comment
                        comment = f"\n  <!-- Generated by model: {self.model_id} -->"
                        root_idx = xml_str.find("<root")
                        if root_idx != -1:
                            end_idx = xml_str.find(">", root_idx) + 1
                            xml_str = xml_str[:end_idx] + comment + xml_str[end_idx:]
                        ok, err  = self._final_validate(
                            xml_str,
                            full_node_specs,
                            known_bb_vars,
                            required_output_vars,
                            recovery_policy,
                        )

                        if ok:
                            self.get_logger().info("🎉 XML validated and accepted by safety check.")
                            response.bt_xml  = xml_str
                            response.success = True
                            response.message = f"Agentic-({self.model_id})"
                            vector_db.delete_collection()
                            return response
                        else:
                            # Safety check caught something — override ToolMessage
                            self.get_logger().warn(f"⚠️  Safety check failed: {err}")
                            messages[-1] = ToolMessage(
                                content=(f"Final validation FAILED: {err}. "
                                         "Fix your XML and re-validate all steps before "
                                         "calling submit_bt_xml again."),
                                tool_call_id=tc['id'],
                            )
                            submit_state["done"] = False
                            submit_state["xml"]  = None

            # ── No tool calls: agent replied with plain text ───────────────
            else:
                xml_candidate = self._extract_xml(ai_msg.content) if ai_msg.content else ""
                if xml_candidate and '<root' in xml_candidate:
                    self.get_logger().info(
                        "💬 Agent generated XML without calling tools — prompting validation."
                    )
                    messages.append(HumanMessage(
                        content=(
                            "You generated XML but did not validate it. "
                            "Please call validate_xml_syntax with your XML now."
                        )
                    ))
                else:
                    self.get_logger().info("💬 Agent sent text only — prompting to start.")
                    messages.append(HumanMessage(
                        content=(
                            "Please generate the BehaviorTree XML for the objective "
                            "and validate it using the available tools."
                        )
                    ))

        # Max steps reached
        response.success = False
        response.message = "Max steps reached without a valid XML submission."
        vector_db.delete_collection()
        return response


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AgenticBTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
