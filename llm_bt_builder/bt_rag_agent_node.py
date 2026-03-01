#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from llm_bt_builder.srv import GenerateBT
import yaml
import re
import os
import time  # NEW: For waits between retries
import xml.etree.ElementTree as ET

# --- LANGCHAIN & RAG IMPORTS ---
try:
    # NEW: Added AIMessage for chat history
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print("❌ ERROR: Missing libraries. Run: pip install langchain-chroma langchain-huggingface sentence-transformers chromadb")
    raise e

class RagBTAgent(Node):
    def __init__(self):
        super().__init__('llm_bt_rag_agent')
        self.get_logger().info(f"🛠️ Starting RAG Node...")

        # 1. PARAMETERS
        self.declare_parameter('model_id', 'gemini-2.0-flash-lite')
        self.declare_parameter('api_url', '')
        self.declare_parameter('api_key', '')
        self.declare_parameter('prompt_file', 'system_prompt.txt')

        self.model_id = self.get_parameter('model_id').value
        self.api_url = self.get_parameter('api_url').value
        self.api_key = self.get_parameter('api_key').value

        # 2. SETUP
        self.llm = self.setup_llm()
        self.embeddings = self.setup_embeddings()

        # 3. SERVICE
        self.srv = self.create_service(GenerateBT, 'generate_bt', self.generate_bt_callback)
        self.get_logger().info(f"✅ RAG Agent ready. Model: {self.model_id}")

    def setup_llm(self):
        TIMEOUT = 120 
        try:
            if 'gemini' in self.model_id.lower():
                self.get_logger().info("🔵 Configuring Gemini...")
                # return ChatGoogleGenerativeAI(
                #     model=self.model_id,
                #     google_api_key=self.api_key,
                #     temperature=0.1,
                #     max_retries=2,
                #     client_options={"timeout": TIMEOUT}
                # )
                return ChatGoogleGenerativeAI(
                        model=self.model_id,
                        google_api_key=self.api_key,
                        temperature=0.1,
                        max_retries=2
                    )
            else: # Fallback to Ollama for other models
                self.get_logger().info(f"🦙 Configuring Ollama ({self.model_id})...")
                clean_url = self.api_url.split('/v1')[0] if self.api_url else "http://localhost:11434"
                return ChatOllama(
                    model=self.model_id,
                    base_url=clean_url,
                    temperature=0.1,
                    timeout=TIMEOUT
                )
        except Exception as e:
            self.get_logger().error(f"❌ Error setting up LLM: {e}")
            return None

    def setup_embeddings(self):
        self.get_logger().info("📥 Loading Embeddings (HuggingFace)...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
        K = 15
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

        system_content = raw_template.replace("{robot_capabilities}", filtered_yaml_str)
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

                # B. Semantic Validation
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

    def validate_bt_semantics(self, xml_string, node_specs):
        # Check that the used nodes exist in the original YAML and that the ports are correct
        try:
            root = ET.fromstring(xml_string)
            control_nodes = [
                'Sequence', 'Fallback', 'ReactiveSequence', 'ReactiveFallback',
                'RetryUntilSuccessful', 'Inverter', 'ForceSuccess', 'ForceFailure',
                'KeepRunningUntilFailure', 'BehaviorTree', 'root', 'AlwaysSuccess', 'AlwaysFailure',
                'Parallel', 'Delay'
            ]
            for elem in root.iter():
                if elem.tag in control_nodes: continue
                if elem.tag not in node_specs:
                    return False, f"Node <{elem.tag}> does NOT exist in the capabilities YAML."
                allowed_ports = node_specs[elem.tag]
                for attr in elem.attrib:
                    if attr in ['name', 'ID']: continue
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