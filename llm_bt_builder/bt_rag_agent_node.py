#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from llm_bt_builder.srv import GenerateBT
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

        self.llm_provider = self.get_parameter('llm_provider').value.lower()
        self.model_id = self.get_parameter('model_id').value
        self.api_url = self.get_parameter('api_url').value
        self.api_key = self.get_parameter('api_key').value

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
                'ollama': ['LLM_API_KEY']
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

        # 3. SETUP
        self.llm = self.setup_llm()
        self.embeddings = self.setup_embeddings()

        # 4. SERVICE
        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
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
            else:
                self.get_logger().error(f"❌ Unknown provider: {self.llm_provider}")
                return None
        except Exception as e:
            self.get_logger().error(f"❌ Error setting up LLM: {e}")
            return None

    def setup_embeddings(self):
        self.get_logger().info("📥 Loading Embeddings (HuggingFace)...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
                if raw_ports:
                    for p in raw_ports:
                        p_name = p.get('key') or p.get('name')
                        if p_name: current_ports.append(p_name)
                specs[node['name']] = current_ports
            return specs
        except: return {}

    def generate_bt_callback(self, request, response):
        K = 10
        MAX_RETRIES = 25

        self.get_logger().info(f"🧠 Objective: '{request.objective}'")

        # 1. DATA PREPARATION
        full_node_specs = self.parse_full_specs(request.bt_nodes_yaml)

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
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=request.objective)
        ]

        # 4. RETRY LOOP 🔄
        for attempt in range(MAX_RETRIES):
            self.get_logger().info(f"Attempt {attempt + 1}/{MAX_RETRIES}...")

            try:
                ai_msg = self.llm.invoke(messages)
                raw_response = ai_msg.content

                think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
                
                if think_match:
                    thought_process = think_match.group(1).strip()
                    self.get_logger().info(f"\n🤔 CHAIN OF THOUGHT:\n\033[93m{thought_process}\033[0m\n")
                else:
                    self.get_logger().info("⚠️ No <think> tags found in the response.")

                xml_str = self.extract_xml(ai_msg.content)

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
                is_valid_bt, bt_msg = self.validate_bt_semantics(xml_str, full_node_specs)
                if not is_valid_bt:
                    self.get_logger().warn(f"⚠️ BT Semantic Error: {bt_msg}")
                    # Specific feedback about invented nodes
                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=f"ERROR: {bt_msg}. You MUST use ONLY the tools provided in the list above."))
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
                self.get_logger().error(f"🔥 Error invoking LLM: {e}")
                time.sleep(2) # Backoff if API fails

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
            
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def validate_bt_semantics(self, xml_string, node_specs):
        # Check that custom nodes exist in YAML and ports are correct
        # Note: Structural validation is done in validate_xml_bt
        try:
            root = ET.fromstring(xml_string)
            
            for elem in root.iter():
                # Skip structural BT.CPP nodes (using set for O(1) lookup)
                if elem.tag in self.structural_nodes:
                    continue
                    
                # Validate custom/action nodes from YAML
                if elem.tag not in node_specs:
                    return False, f"Node <{elem.tag}> does NOT exist in the capabilities YAML."
                
                # Validate ports/attributes
                allowed_ports = node_specs[elem.tag]
                for attr in elem.attrib:
                    if attr in ['name', 'ID']:  # Structural attributes
                        continue
                    if attr not in allowed_ports:
                        return False, f"Node <{elem.tag}> has an illegal port: '{attr}'. Allowed: {allowed_ports}"
            
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()