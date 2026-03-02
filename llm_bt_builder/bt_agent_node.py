#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import requests
import xml.etree.ElementTree as ET
import re
import os
import time
from ament_index_python.packages import get_package_share_directory
from llm_bt_builder.srv import GenerateBT

class BTAgentNode(Node):
    def __init__(self):
        super().__init__('llm_bt_agent')

        # === CONFIGURATION ===
        self.declare_parameter('llm_provider', 'gemini')  # gemini, openai, anthropic, ollama, deepseek
        self.declare_parameter('execution_mode', 'api') 
        self.declare_parameter('model_id', 'gemini-2.5-flash')
        self.declare_parameter('model_cache_dir', './llm_models')
        self.declare_parameter('api_url', 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
        self.declare_parameter('api_key', '')
        self.declare_parameter('prompt_file', 'system_prompt.txt')
        
        # Load parameters
        self.llm_provider = self.get_parameter('llm_provider').value.lower()
        self.mode = self.get_parameter('execution_mode').value
        self.model_id = self.get_parameter('model_id').value
        self.cache_dir = os.path.abspath(self.get_parameter('model_cache_dir').value)
        self.api_url = self.get_parameter('api_url').value
        
        # Smart API KEY management based on provider
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
                'ollama': ['LLM_API_KEY']
            }
            
            env_vars = provider_to_env.get(self.llm_provider, ['LLM_API_KEY'])
            for env_var in env_vars:
                self.api_key = os.getenv(env_var, '')
                if self.api_key:
                    break
            if not self.api_key:
                self.api_key = 'sk-no-key-needed'

        if self.mode == 'api' and not self.api_key and 'localhost' not in self.api_url:
            self.get_logger().error("❌ FATAL ERROR: No API Key found.")
            raise ValueError("API Key missing.")
        
        # Load BT.CPP Node Categories from YAML files
        self.bt_control_nodes_yaml = self._load_bt_nodes_yaml('btv4_control_nodes.yaml')
        self.bt_decorator_nodes_yaml = self._load_bt_nodes_yaml('btv4_decorator_nodes.yaml')
        
        # Extract node names dynamically
        self.control_nodes = self._extract_node_names(self.bt_control_nodes_yaml)
        self.decorators = self._extract_node_names(self.bt_decorator_nodes_yaml)
        
        # Special nodes that don't require validation
        self.special_nodes = ['root', 'BehaviorTree', 'Blackboard', 'SetBlackboard', 'Wait',
                              'AlwaysSuccess', 'AlwaysFailure', 'SubTree']
        
        # All structural nodes (for semantic validation skip)
        self.structural_nodes = set(
            self.decorators + self.control_nodes + self.special_nodes
        )

        # Local initialization (Omitted for brevity, same as before)
        if self.mode == 'local': pass 

        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
        self.get_logger().info(f"🤖 LLM Agent Node ready. Provider: {self.llm_provider}, Model: {self.model_id}")

    def load_prompt_template(self):
        try:
            prompt_file = self.get_parameter('prompt_file').value
            self.get_logger().info(f"📄 Loading prompt template from: {prompt_file}")
            pkg_share_dir = get_package_share_directory('llm_bt_builder')
            path = os.path.join(pkg_share_dir, 'prompts', prompt_file)
            with open(path, 'r', encoding='utf-8') as f: return f.read()
        except Exception: return None
    
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

    def generate_bt_callback(self, request, response):
        MAX_RETRIES = 25

        self.get_logger().info(f"🎯 Objective: '{request.objective}'")
        
        node_specs = {}
        try:
            robot_caps = yaml.safe_load(request.bt_nodes_yaml)
            for node in robot_caps['bt_nodes']:
                current_ports = []
                # Get the list of ports (or empty list if none)
                raw_ports = node.get('ports', [])
                
                if raw_ports:
                    for p in raw_ports:
                        p_name = p.get('key') or p.get('name')
                        
                        if p_name:
                            current_ports.append(p_name)
                        else:
                            # If a port has neither key nor name, ignore but warn
                            self.get_logger().warn(f"Unnamed port in node {node['name']}")

                node_specs[node['name']] = current_ports
                
        except Exception as e:
            self.get_logger().error(f"❌ Error processing YAML: {e}")
            response.success = False; response.message = f"YAML Error: {e}"; return response

        # Load template
        template = self.load_prompt_template()
        if not template: return response
        
        # Prepare BT.CPP standard nodes
        bt_standard_nodes = "## Control Nodes\n" + self.bt_control_nodes_yaml + "\n"
        bt_standard_nodes += "## Decorator Nodes\n" + self.bt_decorator_nodes_yaml
        
        # Prepare prompt with separated sections
        prompt = template.replace("{bt_standard_nodes}", bt_standard_nodes)\
                         .replace("{robot_capabilities}", request.bt_nodes_yaml)\
                         .replace("{user_objective}", request.objective)

        messages = [
            {"role": "system", "content": "You are an expert in BehaviorTree.CPP v4."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(MAX_RETRIES):
            self.get_logger().info(f"🧠 Attempt {attempt + 1}/{MAX_RETRIES}...")
            
            raw_reply = self.call_llm(messages)
            # if not raw_reply: continue
            if not raw_reply: 
                self.get_logger().warn("⚠️ Fallo en llamada LLM. Esperando 5 segundos...")
                time.sleep(5.0)
                continue
            
            # Print chain of thought if present
            think_match = re.search(r'<think>(.*?)</think>', raw_reply, re.DOTALL)
            if think_match:
                thought_process = think_match.group(1).strip()
                self.get_logger().info(f"\n🤔 CHAIN OF THOUGHT:\n\033[93m{thought_process}\033[0m\n")
            else:
                self.get_logger().info("⚠️ No <think> tags found in the response.")

            xml_str = self.extract_xml(raw_reply)
            
            # === PHASE 1: SYNTACTIC VALIDATION ===
            is_xml_valid, xml_result = self.validate_xml_syntax(xml_str)
            
            if not is_xml_valid:
                self.get_logger().warn(f"⚠️ Syntax Error: {xml_result}")
                messages.append({"role": "assistant", "content": xml_str})
                messages.append({"role": "user", "content": f"XML SYNTAX ERROR: {xml_result}. Please fix the tags."})
                continue 

            # === PHASE 2: BEHAVORTREE STRUCTURE VALIDATION ===
            is_structure_valid, structure_msg = self.validate_xml_bt(xml_result)
            
            if not is_structure_valid:
                self.get_logger().warn(f"⚠️ BT Structure Error: {structure_msg}")
                messages.append({"role": "assistant", "content": xml_str})
                messages.append({"role": "user", "content": f"BEHAVORTREE STRUCTURE ERROR: {structure_msg}. Fix the tree structure."})
                continue

            # === PHASE 3: SEMANTIC VALIDATION ===
            is_bt_valid, semantic_msg = self.validate_bt_semantics(xml_result, node_specs)

            if is_bt_valid:
                self.get_logger().info("✅ XML Validated (Syntax and Semantics).")
                response.success = True
                response.bt_xml = xml_str
                response.message = self.model_id
                self.get_logger().debug("Message to client: " + response.message)
                return response
            else:
                self.get_logger().warn(f"⚠️ Semantic Error: {semantic_msg}")
                messages.append({"role": "assistant", "content": xml_str})
                messages.append({"role": "user", "content": f"LOGICAL ERROR: {semantic_msg}. Only use the defined ports."})

        response.success = False
        response.message = "Exceeded number of retries."
        return response

    def validate_xml_syntax(self, xml_str):
        """Step 1: Check if the string is valid XML."""
        try:
            root = ET.fromstring(xml_str)
            return True, root # Return the root object for reuse
        except ET.ParseError as e:
            return False, str(e)

    def validate_xml_bt(self, root):
        """Step 2: Validate BehaviorTree structural rules (decorators have 1 child, control nodes have children, etc.)"""
        try:
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
            
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def validate_bt_semantics(self, root, node_specs):
        """Step 3: Check if custom nodes exist in YAML and ports are correct."""
        # Note: Structural validation is done in validate_xml_bt
        
        # Structural attributes allowed in any node
        ignored_attrs = ['ID', 'name', 'num_attempts', 'server_name', 'server_timeout', 'path', '_success', '_failure'] 

        for elem in root.iter():
            # Skip structural BT.CPP nodes (using set for O(1) lookup)
            if elem.tag in self.structural_nodes:
                continue
            
            # 1. Node name check - must exist in YAML capabilities
            if elem.tag not in node_specs:
                return False, f"Node NOT allowed: <{elem.tag}>. Not in your skill list."
            
            # 2. Port check (attributes)
            allowed_ports = node_specs[elem.tag]
            for attr in elem.attrib:
                if attr in ignored_attrs: continue
                
                if attr not in allowed_ports:
                    return False, f"Node <{elem.tag}> has an invented port: '{attr}'. Valid ports: {allowed_ports}"

        return True, "OK"

    def call_llm(self, messages):
        TIMEOUT_SEC = 180.0

        if self.mode == 'api':
            try:
                # =========================================================
                # BRANCH 1: GOOGLE GEMINI (NATIVE API)
                # =========================================================
                if self.llm_provider == 'gemini':
                    # Build the complete URL from base URL or default
                    base_url = self.api_url if self.api_url and self.api_url != '' else 'https://generativelanguage.googleapis.com'
                    base_url = base_url.rstrip('/')
                    url = f"{base_url}/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
                    
                    contents = []
                    for msg in messages:
                        # Gemini is strict with roles: only 'user' or 'model'.
                        # Convert 'system' to an instruction inside 'user'.
                        role = "user" if msg['role'] in ['user', 'system'] else "model"
                        text = f"[SYSTEM INSTRUCTION]: {msg['content']}" if msg['role'] == 'system' else msg['content']
                        contents.append({"role": role, "parts": [{"text": text}]})
                        
                    payload = {
                        "contents": contents,
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 2048
                        }
                    }
                    
                    resp = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
                    
                    if resp.status_code != 200:
                        self.get_logger().error(f"❌ Gemini Error (HTTP {resp.status_code}): {resp.text}")
                        return None
                    
                    data = resp.json()
                    # Extracción específica de Google (candidates -> content -> parts)
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        self.get_logger().error(f"❌ Empty Gemini response: {data}")
                        return None
                    
                # =========================================================
                # BRANCH 2: STANDARD OPENAI / ANTHROPIC / DEEPSEEK / OLLAMA
                # =========================================================
                elif self.llm_provider in ['openai', 'anthropic', 'deepseek', 'ollama']:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    # Anthropic uses a different header for API key
                    if self.llm_provider == 'anthropic':
                        headers = {
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01"
                        }
                    
                    # Standard payload for OpenAI-compatible APIs
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": 0.1
                    }
                    
                    # Add max_tokens for Anthropic
                    if self.llm_provider == 'anthropic':
                        payload["max_tokens"] = 4096
                    
                    # Determine base URL from parameter or defaults
                    if self.api_url and self.api_url != '':
                        base_url = self.api_url.rstrip('/')
                    else:
                        # Default base URLs per provider
                        if self.llm_provider == 'openai':
                            base_url = 'https://api.openai.com'
                        elif self.llm_provider == 'anthropic':
                            base_url = 'https://api.anthropic.com'
                        elif self.llm_provider == 'deepseek':
                            base_url = 'https://api.deepseek.com'
                        else:  # ollama
                            base_url = 'http://localhost:11434'
                    
                    # Build complete endpoint based on provider
                    if self.llm_provider == 'anthropic':
                        endpoint = f"{base_url}/v1/messages"
                    else:  # openai, deepseek, ollama
                        endpoint = f"{base_url}/v1/chat/completions"
                    
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SEC)
                    
                    if resp.status_code != 200:
                        self.get_logger().error(f"❌ {self.llm_provider.upper()} API Error (HTTP {resp.status_code}): {resp.text}")
                        return None
                        
                    data = resp.json()
                    
                    # Extract response based on provider format
                    if self.llm_provider == 'anthropic':
                        # Anthropic format: content -> [0] -> text
                        if 'content' in data and len(data['content']) > 0:
                            return data['content'][0]['text']
                        else:
                            self.get_logger().error(f"❌ Unexpected Anthropic response: {data}")
                            return None
                    else:
                        # Standard OpenAI format: choices -> message -> content
                        if 'choices' in data and len(data['choices']) > 0:
                            return data['choices'][0]['message']['content']
                        else:
                            self.get_logger().error(f"❌ Unexpected or empty response: {data}")
                            return None
                
                else:
                    self.get_logger().error(f"❌ Unknown provider: {self.llm_provider}")
                    return None

            except requests.exceptions.ConnectionError:
                self.get_logger().error(f"❌ Connection error. Is the server running? URL: {self.api_url}")
                return None
            except Exception as e:
                self.get_logger().error(f"❌ Unexpected API failure: {e}")
                return None
        
        # If 'local' mode with transformers was used, it would go here, but for API we return None if we reach the end
        return None

    def extract_xml(self, text):
        # Remove chain of thought
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Try to extract from ```xml ... ``` block
        match = re.search(r'```xml(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip().replace('{{', '{').replace('}}', '}')
        # If not found, try to extract from <root>...</root>
        if '<root' in text:
            start = text.find('<root')
            end = text.rfind('</root>')
            if end != -1:
                return text[start:end+7].replace('{{', '{').replace('}}', '}')
        # Fallback: return cleaned text
        return text.strip().replace('{{', '{').replace('}}', '}')

def main(args=None):
    rclpy.init(args=args)
    node = BTAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()