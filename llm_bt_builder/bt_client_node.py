#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
from ament_index_python.packages import get_package_share_directory

# Importamos el mensaje del servicio
from llm_bt_builder.srv import GenerateBT

class BTClientNode(Node):
    def __init__(self):
        super().__init__('bt_client_node')
        self.client = self.create_client(GenerateBT, 'generate_bt')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando a que el servicio LLM "generate_bt" esté disponible...')
            
        self.req = GenerateBT.Request()

    def send_request(self, objective, yaml_content):
        self.req.objective = objective
        self.req.bt_nodes_yaml = yaml_content
        
        self.get_logger().info('Enviando petición al LLM. Espera por favor...')
        self.future = self.client.call_async(self.req)
        
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def get_src_xml_path():
    """Finds the path to the src folder of the package to save the XML file."""
    # Intentamos obtener la ruta de instalación del paquete
    try:
        share_dir = get_package_share_directory('llm_bt_builder')
          
        if 'install' in share_dir:
            # Reemplazamos la ruta de install por la de src para desarrollo
            base_path = share_dir.split('/install/')[0]
            src_path = os.path.join(base_path, 'src', 'llm_bt_builder', 'xml')
        else:
            # Fallback a directorio local si no se detecta estructura de workspace
            src_path = os.path.join(os.getcwd(), 'xml')
            
        return src_path
    except Exception:
        return os.path.join(os.getcwd(), 'xml')

def main(args=None):
    rclpy.init(args=args)
    
    OBJECTIVE = ("Tienes que explicarle a la persona cómo es el test de velocidad de marcha, que consiste en que la persona tiene que caminar a su velocidad normal hasta que le digas que pare.\n"
                "No lo va a hacer aún solo queremos que lo entienda para después.\n"
                "Asegúrate de que ha entendido la explicación preguntándoselo explícitamente.\n"
                "Si no lo entiende, vuelve a explicárselo hasta que lo entienda.")
    
    # Determinamos la ruta de guardado en el paquete fuente (src)
    XML_DIR = get_src_xml_path()
    os.makedirs(XML_DIR, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_FILE = None  # Se definirá tras recibir la respuesta

    # 1. Obtain the YAML content from the package share directory
    try:
        pkg_share_dir = get_package_share_directory('llm_bt_builder')
        yaml_path = os.path.join(pkg_share_dir, 'config', 'social_bt_nodes.yaml')
    except Exception as e:
        print(f"❌ Error finding the package: {e}")
        return

    # 2. Read the YAML file content
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
    except Exception as e:
        print(f"❌ Error reading YAML file at {yaml_path}: {e}")
        return

    # 3. Instanciar el nodo y llamar al servicio
    client_node = BTClientNode()
    client_node.get_logger().info(f"🎯 OBJETIVO enviado: '{OBJECTIVE}'")
    
    response = client_node.send_request(OBJECTIVE, yaml_content)

    # 4. Process the response
    if response.success:
        client_node.get_logger().info("✅ Behavior Tree successfully generated and validated!")
        # Sanitize the message field to use in the filename
        import re
        def sanitize_filename(s):
            s = s.strip().replace(' ', '_')
            s = re.sub(r'[^\w\-_\.]+', '', s)
            return s[:40]  # Limit length to avoid excessive filenames

        message_part = sanitize_filename(response.message) if response.message else ""
        file_name = f"llm_gen_bt_{timestamp}"
        if message_part:
            file_name += f"_{message_part}"
        file_name += ".xml"
        OUT_FILE = os.path.join(XML_DIR, file_name)
        try:
            with open(OUT_FILE, 'w', encoding='utf-8') as f:
                f.write(response.bt_xml)
            client_node.get_logger().info(f"💾 Successfully saved at: {OUT_FILE}")
        except Exception as e:
            client_node.get_logger().error(f"❌ Error saving XML file: {e}")
            print("\n--- XML CONTENT ---\n")
            print(response.bt_xml)
    else:
        client_node.get_logger().error(f"❌ Generation failed:\n{response.message}")
    # Cleanup
    client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()