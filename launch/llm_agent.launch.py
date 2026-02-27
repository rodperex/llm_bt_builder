from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Declarar los argumentos (con valores por defecto seguros)
    
    # Modelo por defecto (Gemini Flash es el más rápido/barato)
    model_arg = DeclareLaunchArgument(
        'model',
        # default_value='gemini-2.5-flash',
        # default_value='qwen2.5-coder:1.5b', # ollama
        # default_value='llama3.1', # ollama (powerful GPU required)
        # default_value='qwen2.5-coder:7b', # ollama (powerful GPU required)
        default_value ='deepseek-r1:8b', # ollama
        description='ID del modelo a usar (ej: gemini-2.5-flash, llama3.1, qwen2.5-coder:1.5b)'
    )

    # Modo de ejecución (api vs local)
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='api',
        description='Modo de ejecución: "api" o "local"'
    )

    # URL de la API (Por defecto la de Google OpenAI, pero cambiable para Ollama)
    url_arg = DeclareLaunchArgument(
        'url',
        # default_value='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        default_value='http://localhost:11434/v1/chat/completions',
        description='Endpoint de la API (ej: http://localhost:11434/v1/chat/completions)'
    )

    # API Key (Intenta leer la variable de entorno por defecto)
    # Si la pasas por comando, sobrescribe la variable de entorno.
    key_arg = DeclareLaunchArgument(
        'key',
        default_value=EnvironmentVariable('GEMINI_API_KEY', default_value=''),
        description='API Key (opcional si ya está en variables de entorno)'
    )

    # 2. Definir el nodo
    agent_node = Node(
        package='llm_bt_builder',
        executable='bt_agent_node.py',
        name='llm_bt_agent',
        output='screen',
        emulate_tty=True, # Para ver los colores en los logs
        parameters=[{
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key')
        }]
    )

    return LaunchDescription([
        model_arg,
        mode_arg,
        url_arg,
        key_arg,
        agent_node
    ])