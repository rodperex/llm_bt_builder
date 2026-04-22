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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    provider_arg = DeclareLaunchArgument(
        'provider',
        default_value='openai',
        # Options: 'gemini', 'openai', 'anthropic', 'deepseek', 'ollama', 'groq', 'sambanova'
        description='LLM Provider: gemini, openai, anthropic, deepseek, or ollama'
    )

    model_arg = DeclareLaunchArgument(
        'model',
        # default_value='Meta-Llama-3.3-70B-Instruct',
        # default_value='gemini-2.5-flash-lite',
        # default_value='llama-3.1-8b-instant',
        default_value='gpt-4o',
        # default_value='gemini-2.0-flash-lite',
        # default_value='qwen2.5-coder:3b', # ollama (lighter model for testing)
        # default_value='qwen2.5-coder:7b', # ollama (powerful GPU required)
        # default_value ='deepseek-r1:8b', # ollama
        # default_value='deepseek-chat', # deepseek cloud API
        # default_value='qwen2.5-coder:3b',
        # Options per provider:
        # Gemini: 'gemini-2.5-flash', 'gemini-2.0-flash-lite'
        # OpenAI: 'gpt-4o', 'gpt-3.5-turbo'
        # Anthropic: 'claude-2', 'claude-instant-100k'
        # DeepSeek: 'deepseek-chat'
        # Ollama: any local model you have set up (e.g., 'llama3.1', 'qwen2.5-coder:7b'
        # Groq: 'llama-3.3-70b-versatile'
        # Groq: 'llama-3.1-8b-instant'
        # Sambanova: 'Meta-Llama-3.3-70B-Instruct'
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
        # Leave empty to let each node auto-detect the URL based on the provider.
        # Override only when needed (e.g. custom Ollama endpoint):
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
        description='Agent type: "rag", "mcp_rag", "normal", or "agentic"'
    )

    mcp_enabled_arg = DeclareLaunchArgument(
        'mcp_enabled',
        default_value='false',
        description='Enable MCP context enrichment (used by mcp_rag).'
    )

    mcp_cmd_arg = DeclareLaunchArgument(
        'mcp_cmd',
        default_value='ros2 run mcp_context_server mcp_context_server',
        description='Command used to start the MCP server.'
    )

    mcp_timeout_arg = DeclareLaunchArgument(
        'mcp_timeout_sec',
        default_value='2.0',
        description='Timeout for MCP calls in seconds.'
    )

    mcp_fail_open_arg = DeclareLaunchArgument(
        'mcp_fail_open',
        default_value='true',
        description='If true, continue without MCP on failure.'
    )

    prompt_file_arg = DeclareLaunchArgument(
        'prompt_file',
        default_value='system_prompt.txt',
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

    # MCP-enhanced RAG node (A/B against bt_rag_agent_node)
    mcp_rag_node = Node(
        package='llm_bt_builder',
        executable='mcp_bt_rag_agent_node.py',
        name='mcp_llm_rag_bt_agent',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'llm_provider': LaunchConfiguration('provider'),
            'model_id': LaunchConfiguration('model'),
            'execution_mode': LaunchConfiguration('mode'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key'),
            'prompt_file': LaunchConfiguration('prompt_file'),
            'mcp_enabled': LaunchConfiguration('mcp_enabled'),
            'mcp_cmd': LaunchConfiguration('mcp_cmd'),
            'mcp_timeout_sec': LaunchConfiguration('mcp_timeout_sec'),
            'mcp_fail_open': LaunchConfiguration('mcp_fail_open'),
        }],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('agent_type'), "' == 'mcp_rag'"])
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

    # Agentic node (tool-calling / ReAct)
    agentic_node = Node(
        package='llm_bt_builder',
        executable='bt_rag_agentic_node.py',
        name='llm_bt_agentic',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'llm_provider': LaunchConfiguration('provider'),
            'model_id': LaunchConfiguration('model'),
            'api_url': LaunchConfiguration('url'),
            'api_key': LaunchConfiguration('key'),
            'prompt_file': LaunchConfiguration('prompt_file')
        }],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('agent_type'), "' == 'agentic'"])
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
        mcp_enabled_arg,
        mcp_cmd_arg,
        mcp_timeout_arg,
        mcp_fail_open_arg,
        rag_node,
        mcp_rag_node,
        normal_node,
        agentic_node,
    ])
