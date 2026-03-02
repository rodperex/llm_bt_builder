from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    provider_arg = DeclareLaunchArgument(
        'provider',
        default_value='openai',
        # Options: 'gemini', 'openai', 'anthropic', 'deepseek', 'ollama'
        description='LLM Provider: gemini, openai, anthropic, deepseek, or ollama'
    )

    model_arg = DeclareLaunchArgument(
        'model',
        # default_value='gemini-2.5-flash',
        default_value='gpt-4o',
        # default_value='gemini-2.0-flash-lite',
        # default_value='llama3.1', # ollama (powerful GPU required)
        # default_value='qwen2.5-coder:7b', # ollama (powerful GPU required)
        # default_value ='deepseek-r1:8b', # ollama
        # default_value='deepseek-chat', # deepseek cloud API
        # default_value='qwen2.5-coder:3b',
        # Options per provider:
        # Gemini: 'gemini-2.5-flash', 'gemini-2.0-flash-lite'
        # OpenAI: 'gpt-4o', 'gpt-3.5-turbo'
        # Anthropic: 'claude-2', 'claude-instant-100k'
        # DeepSeek: 'deepseek-chat'
        # Ollama: any local model you have set up (e.g., 'llama3.1', 'qwen2.5-coder:7b', 'deepseek-r
        description='Model ID to use (e.g., gemini-2.5-flash, llama3.1, qwen2.5-coder:1.5b, deepseek-chat)'
    )

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='api',
        description='Execution mode: "api" or "local"'
    )

    url_arg = DeclareLaunchArgument(
        'url',
        default_value='',
        # Options per provider (base URLs, endpoints are added by the code):
        # Gemini: 'https://generativelanguage.googleapis.com'
        # OpenAI: 'https://api.openai.com'
        # Anthropic: 'https://api.anthropic.com'
        # DeepSeek: 'https://api.deepseek.com'
        # Ollama: 'http://localhost:11434'
        description='API base URL (optional, auto-detected per provider if empty)'
    )

    key_arg = DeclareLaunchArgument(
        'key',
        default_value='',
        description='API Key (optional, will auto-detect based on provider from env vars)'
    )

    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='rag',
        description='Agent type: "rag" for RAG agent, "normal" for standard agent'
    )

    prompt_file_arg = DeclareLaunchArgument(
        'prompt_file',
        default_value='system_prompt_cot.txt',
        description='Prompt file name in prompts/ directory (e.g., system_prompt.txt)'
    )

    # RAG node
    rag_node = Node(
        package='llm_bt_builder',
        executable='bt_rag_agent_node.py',
        name='llm_rag_bt_agent',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'llm_provider': LaunchConfiguration('provider'),
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key'),
            'prompt_file': LaunchConfiguration('prompt_file')
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
            'llm_provider': LaunchConfiguration('provider'),
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key'),
            'prompt_file': LaunchConfiguration('prompt_file')
        }],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('agent_type'), "' == 'normal'"])
        )
    )

    return LaunchDescription([
        provider_arg,
        model_arg,
        mode_arg,
        url_arg,
        key_arg,
        agent_type_arg,
        prompt_file_arg,
        rag_node,
        normal_node
    ])
