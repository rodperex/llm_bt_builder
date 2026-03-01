from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    model_arg = DeclareLaunchArgument(
        'model',
        # default_value='gemini-2.5-flash',
        # default_value='gemini-2.0-flash-lite',
        # default_value='llama3.1', # ollama (powerful GPU required)
        # default_value='qwen2.5-coder:7b', # ollama (powerful GPU required)
        # default_value ='deepseek-r1:8b', # ollama
        default_value='qwen2.5-coder:3b',
        description='Model ID to use (e.g., gemini-2.5-flash, llama3.1, qwen2.5-coder:1.5b)'
    )

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='api',
        description='Execution mode: "api" or "local"'
    )

    url_arg = DeclareLaunchArgument(
        'url',
        # default_value='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        default_value='http://localhost:11434/v1/chat/completions',
        description='API endpoint (e.g., http://localhost:11434/v1/chat/completions)'
    )

    key_arg = DeclareLaunchArgument(
        'key',
        default_value=EnvironmentVariable('GEMINI_API_KEY', default_value=''),
        description='API Key (optional if already set in environment variables)'
    )

    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='rag',
        description='Agent type: "rag" for RAG agent, "normal" for standard agent'
    )

    # RAG node
    rag_node = Node(
        package='llm_bt_builder',
        executable='bt_rag_agent_node.py',
        name='llm_rag_bt_agent',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key')
        }],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('agent_type'), "' == 'rag'"])
        )
    )

    # Normal node
    normal_node = Node(
        package='llm_bt_builder',
        executable='bt_agent_node.py',
        name='llm_bt_agent',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key')
        }],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('agent_type'), "' == 'normal'"])
        )
    )

    return LaunchDescription([
        model_arg,
        mode_arg,
        url_arg,
        key_arg,
        agent_type_arg,
        rag_node,
        normal_node
    ])
