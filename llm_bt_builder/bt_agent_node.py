#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import requests
import xml.etree.ElementTree as ET
import re
import os
from ament_index_python.packages import get_package_share_directory
from llm_bt_builder.srv import GenerateBT

class BTAgentNode(Node):
    def __init__(self):
        super().__init__('llm_bt_agent')

        # === CONFIGURATION ===
        self.declare_parameter('execution_mode', 'api') 
        self.declare_parameter('model_id', 'gemini-2.5-flash')
        self.declare_parameter('model_cache_dir', './llm_models')
        self.declare_parameter('api_url', 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
        self.declare_parameter('api_key', '')
        
        # Load parameters
        self.mode = self.get_parameter('execution_mode').value
        self.model_id = self.get_parameter('model_id').value
        self.cache_dir = os.path.abspath(self.get_parameter('model_cache_dir').value)
        self.api_url = self.get_parameter('api_url').value
        
        # Smart API KEY management
        param_key = self.get_parameter('api_key').value
        if param_key and param_key != "sk-no-key-needed":
            self.api_key = param_key
        else:
            # Automatic detection according to provider
            url_lower = self.api_url.lower()
            if 'googleapis' in url_lower: self.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY', '')
            elif 'openai' in url_lower: self.api_key = os.getenv('OPENAI_API_KEY', '')
            elif 'deepseek' in url_lower: self.api_key = os.getenv('DEEPSEEK_API_KEY', '')
            else: self.api_key = os.getenv('LLM_API_KEY', 'sk-no-key-needed')

        if self.mode == 'api' and not self.api_key and 'localhost' not in self.api_url:
            self.get_logger().error("❌ FATAL ERROR: No API Key found.")
            raise ValueError("API Key missing.")
        
        # Standard BT.CPP control nodes always allowed
        self.control_nodes = [
            'Sequence', 'Fallback', 'ReactiveSequence', 'ReactiveFallback',
            'Inverter', 'RetryUntilSuccessful', 'ForceSuccess', 'ForceFailure',
            'Parallel', 'ParallelAll', 'ParallelOne', 'Switch', 'WhileDoElse',
            'Repeat', 'RepeatUntilFailure', 'RepeatUntilSuccess', 'SubTree', 
            'Timeout', 'Delay', 'PipelineSequence', 'IfThenElse', 'Blackboard', 
            'SetBlackboard', 'Wait', 'AlwaysSuccess', 'AlwaysFailure'
        ]

        # Local initialization (Omitted for brevity, same as before)
        if self.mode == 'local': pass 

        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
        self.get_logger().info("🤖 LLM Agent Node ready (2-phase validation).")

    def load_prompt_template(self):
        try:
            pkg_share_dir = get_package_share_directory('llm_bt_builder')
            path = os.path.join(pkg_share_dir, 'prompts', 'system_prompt.txt')
            with open(path, 'r', encoding='utf-8') as f: return f.read()
        except Exception: return None

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
        
        # Prepare prompt
        prompt = template.replace("{robot_capabilities}", request.bt_nodes_yaml)\
                         .replace("{user_objective}", request.objective)

        messages = [
            {"role": "system", "content": "You are an expert in BehaviorTree.CPP v4."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(MAX_RETRIES):
            self.get_logger().info(f"🧠 Attempt {attempt + 1}/{MAX_RETRIES}...")
            
            raw_reply = self.call_llm(messages)
            if not raw_reply: continue
            
            xml_str = self.extract_xml(raw_reply)
            
            # === PHASE 1: SYNTACTIC VALIDATION ===
            is_xml_valid, xml_result = self.validate_xml_syntax(xml_str)
            
            if not is_xml_valid:
                self.get_logger().warn(f"⚠️ Syntax Error: {xml_result}")
                messages.append({"role": "assistant", "content": xml_str})
                messages.append({"role": "user", "content": f"XML SYNTAX ERROR: {xml_result}. Please fix the tags."})
                continue 

            # === PHASE 2: SEMANTIC VALIDATION ===
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

    def validate_bt_semantics(self, root, node_specs):
        """Step 2: Check if nodes and attributes exist in the dictionary."""
        # Structural attributes allowed in any node
        ignored_attrs = ['ID', 'name', 'num_attempts', 'server_name', 'server_timeout', 'path', '_success', '_failure'] 

        for elem in root.iter():
            # Ignore standard control nodes
            if elem.tag in ['root', 'BehaviorTree'] + self.control_nodes:
                continue
            
            # 1. Node name check
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
                if 'googleapis' in self.api_url:
                    # Build the native URL ignoring the OpenAI endpoint
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
                    
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
                # BRANCH 2: STANDARD OLLAMA / OPENAI
                # =========================================================
                else:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    # Ollama and OpenAI usually accept the 'system' role directly,
                    # so we pass the messages as is (or filter if needed).
                    payload = {
                        "model": self.model_id,
                        "messages": messages, # Pasamos la lista original
                        "temperature": 0.1
                    }
                    
                    # Request to the standard endpoint (e.g. localhost:11434/v1/chat/completions)
                    resp = requests.post(self.api_url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
                    
                    if resp.status_code != 200:
                        self.get_logger().error(f"❌ Standard API Error (HTTP {resp.status_code}): {resp.text}")
                        return None
                        
                    data = resp.json()
                    
                    # Extracción estándar (choices -> message -> content)
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content']
                    else:
                        self.get_logger().error(f"❌ Unexpected or empty response: {data}")
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
        match = re.search(r'```xml(.*?)```', text, re.DOTALL)
        xml_str = match.group(1).strip() if match else text.strip()
        return xml_str.replace('{{', '{').replace('}}', '}')

def main(args=None):
    rclpy.init(args=args)
    node = BTAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()