#!/usr/bin/env python3

# Copyright 2026 Rodrigo Perez-Rodriguez
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

import json
import shlex
import subprocess
import threading
import uuid
import pathlib
import sys

import rclpy

try:
    from llm_bt_builder.bt_rag_agent_node import RagBTAgent
except ModuleNotFoundError:
    script_dir = pathlib.Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from bt_rag_agent_node import RagBTAgent


class StdioMCPClient:
    # JSON-RPC client over stdio used by the MCP BT agent.
    # Like planner MCP, this launches its own MCP subprocess per node instance.
    def __init__(self, cmd: str, timeout_sec: float = 2.0):
        self.cmd = cmd
        self.timeout_sec = timeout_sec
        self.proc = None
        self.lock = threading.Lock()

    def start(self):
        if self.proc is not None:
            return
        parts = shlex.split(self.cmd)
        self.proc = subprocess.Popen(
            parts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # MCP handshake required before using tools/call.
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mcp_bt_rag_agent_node", "version": "0.0.1"}})
        self._notify("notifications/initialized", {})

    def stop(self):
        if self.proc is None:
            return
        try:
            self._rpc("shutdown", {})
            self._notify("exit", {})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None

    def _notify(self, method: str, params: dict):
        if self.proc is None or self.proc.stdin is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict):
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("MCP process is not running")

        req_id = str(uuid.uuid4())
        req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout")
            msg = json.loads(line)
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise RuntimeError(str(msg["error"]))
            return msg.get("result", {})

    def call_tool(self, name: str, arguments: dict):
        with self.lock:
            result = self._rpc("tools/call", {"name": name, "arguments": arguments})
            if isinstance(result, dict) and "structuredContent" in result:
                return result["structuredContent"]
            return result


class MCPRagBTAgent(RagBTAgent):
    # Extension pattern over legacy RagBTAgent:
    # - Keep original generation/fix pipeline.
    # - Add MCP context to objective just-in-time.
    # This preserves previous behavior when MCP is disabled or fail-open is enabled.
    def __init__(self):
        super().__init__()

        self.declare_parameter("mcp_enabled", False)
        self.declare_parameter("mcp_cmd", "")
        self.declare_parameter("mcp_timeout_sec", 2.0)
        self.declare_parameter("mcp_fail_open", True)

        self.mcp_enabled = bool(self.get_parameter("mcp_enabled").value)
        self.mcp_cmd = str(self.get_parameter("mcp_cmd").value)
        self.mcp_timeout_sec = float(self.get_parameter("mcp_timeout_sec").value)
        self.mcp_fail_open = bool(self.get_parameter("mcp_fail_open").value)

        self.mcp_client = None
        if self.mcp_enabled and self.mcp_cmd:
            try:
                self.mcp_client = StdioMCPClient(self.mcp_cmd, self.mcp_timeout_sec)
                self.mcp_client.start()
                self.get_logger().info("MCP enabled for bt_rag agent")
            except Exception as exc:
                self.get_logger().error(f"MCP init failed: {exc}")
                # fail_open=True => fall back to plain RagBTAgent behavior.
                if not self.mcp_fail_open:
                    raise

    def destroy_node(self):
        if self.mcp_client is not None:
            self.mcp_client.stop()
        return super().destroy_node()

    def _build_mcp_block(self, include_failures: bool = False) -> str:
        # Gather read-only runtime context from MCP tools.
        if not self.mcp_client:
            return ""
        try:
            capabilities = self.mcp_client.call_tool("get_capabilities", {})
            snapshot = self.mcp_client.call_tool("get_mission_snapshot", {})
            failures = self.mcp_client.call_tool("get_failure_history", {"limit": 20}) if include_failures else {"items": []}

            return (
                "\n\n# MCP_CONTEXT (read-only runtime)\n"
                f"mcp_capabilities: {json.dumps(capabilities, ensure_ascii=True)}\n"
                f"mcp_mission_snapshot: {json.dumps(snapshot, ensure_ascii=True)}\n"
                f"mcp_recent_failures: {json.dumps(failures, ensure_ascii=True)}\n"
                "# Use this as grounding only; keep using available_blackboard_vars and declared outputs strictly.\n"
            )
        except Exception as exc:
            self.get_logger().warn(f"MCP query failed: {exc}")
            if self.mcp_fail_open:
                return ""
            raise

    def _run_agentic_pipeline(self, request, response, is_fix):
        # Inject MCP context before calling the original pipeline.
        # Restore original request afterwards to avoid side effects across retries.
        original_objective = request.objective
        try:
            request.objective = request.objective + self._build_mcp_block(include_failures=is_fix)
            return super()._run_agentic_pipeline(request, response, is_fix)
        finally:
            request.objective = original_objective


def main(args=None):
    rclpy.init(args=args)
    node = MCPRagBTAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
