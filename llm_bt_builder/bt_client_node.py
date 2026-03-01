#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import datetime
import re
from ament_index_python.packages import get_package_share_directory

from llm_bt_builder.srv import GenerateBT

class BTClientNode(Node):
    def __init__(self):
        super().__init__('bt_client_node')
        
        # Get relative path to the installed package
        try:
            pkg_share_dir = get_package_share_directory('llm_bt_builder')
            default_objective_file = os.path.join(pkg_share_dir, 'objectives', 'explain.txt')
        except Exception:
            default_objective_file = ''
        
        self.declare_parameter('objective_file', default_objective_file)
        # Capability YAML file parameter
        default_yaml_file = os.path.join(pkg_share_dir, 'config', 'social_bt_nodes.yaml')
        self.declare_parameter('capabilities_yaml', default_yaml_file)
        
        self.client = self.create_client(GenerateBT, 'generate_bt')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for LLM service "generate_bt" to be available...')
            
        self.req = GenerateBT.Request()

    def send_request(self, objective, yaml_content):
        self.req.objective = objective
        self.req.bt_nodes_yaml = yaml_content
        
        self.get_logger().info('Sending request to LLM. Please wait...')
        self.future = self.client.call_async(self.req)
        
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def run(self):
        """Executes the complete BT generation workflow"""
        # Read the objective file
        objective_file = self.get_parameter('objective_file').value
        try:
            with open(objective_file, 'r', encoding='utf-8') as f:
                objective = f.read().strip()
        except Exception as e:
            self.get_logger().error(f"❌ Error reading objective file at {objective_file}: {e}")
            return
        
        self.get_logger().info(f"📄 Objective loaded from: {objective_file}")
        
        # Prepare output directory
        xml_dir = self.get_src_xml_path()
        os.makedirs(xml_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the YAML capabilities content
        yaml_path = self.get_parameter('capabilities_yaml').value
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                self.get_logger().info(f"📄 Capabilities YAML loaded from: {yaml_path}")
                self.get_logger().debug(f"YAML Content:\n{yaml_content}")
        except Exception as e:
            self.get_logger().error(f"❌ Error reading YAML file at {yaml_path}: {e}")
            return
        
        # Send request to service
        self.get_logger().info(f"🎯 OBJECTIVE sent: '{objective}'")
        response = self.send_request(objective, yaml_content)
        
        # Process the response
        if response.success:
            self.get_logger().info("✅ Behavior Tree successfully generated and validated!")
            
            # Sanitize the message for the filename
            def sanitize_filename(s):
                s = s.strip().replace(' ', '_')
                s = re.sub(r'[^\w\-_\.]+', '', s)
                return s[:40]
            
            message_part = sanitize_filename(response.message) if response.message else ""
            file_name = f"llm_gen_bt_{timestamp}"
            if message_part:
                file_name += f"_{message_part}"
            file_name += ".xml"
            out_file = os.path.join(xml_dir, file_name)
            
            try:
                with open(out_file, 'w', encoding='utf-8') as f:
                    # Add the objective as a comment at the beginning of the XML
                    f.write("<!--\n")
                    f.write("OBJECTIVE:\n")
                    for line in objective.split('\n'):
                        f.write(f"  {line}\n")
                    f.write("-->\n\n")
                    f.write(response.bt_xml)
                self.get_logger().info(f"💾 Successfully saved at: {out_file}")
            except Exception as e:
                self.get_logger().error(f"❌ Error saving XML file: {e}")
                print("\n--- XML CONTENT ---\n")
                print(response.bt_xml)
        else:
            self.get_logger().error(f"❌ Generation failed:\n{response.message}")
    
    def get_src_xml_path(self):
        """Finds the path to the src folder of the package to save the XML file."""
        try:
            share_dir = get_package_share_directory('llm_bt_builder')
            
            if 'install' in share_dir:
                base_path = share_dir.split('/install/')[0]
                src_path = os.path.join(base_path, 'src', 'llm_bt_builder', 'xml')
            else:
                src_path = os.path.join(os.getcwd(), 'xml')
            
            return src_path
        except Exception:
            return os.path.join(os.getcwd(), 'xml')

def main(args=None):
    rclpy.init(args=args)
    
    client_node = BTClientNode()
    
    try:
        client_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()