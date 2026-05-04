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
import difflib

import rclpy
try:
    from llm_bt_builder.bt_rag_agent_node import RagBTAgent, Document, Chroma
except ModuleNotFoundError:
    script_dir = pathlib.Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from bt_rag_agent_node import RagBTAgent, Document, Chroma


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
            stripped = line.strip()
            if not stripped.startswith('{'):
                # Non-JSON line (e.g. ros2 run startup banners, warnings).
                continue
            msg = json.loads(stripped)
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
        self.declare_parameter("use_episodic_mem", False)
        self.declare_parameter("episodic_mem_pool_size", 100)
        self.declare_parameter("episodic_mem_top_k", 5)
        self.declare_parameter("episodic_mem_near_dup_threshold", 0.92)

        self.mcp_enabled = bool(self.get_parameter("mcp_enabled").value)
        self.mcp_cmd = str(self.get_parameter("mcp_cmd").value)
        self.mcp_timeout_sec = float(self.get_parameter("mcp_timeout_sec").value)
        self.mcp_fail_open = bool(self.get_parameter("mcp_fail_open").value)
        self.use_episodic_mem = bool(self.get_parameter("use_episodic_mem").value)
        self.episodic_mem_pool_size = int(self.get_parameter("episodic_mem_pool_size").value)
        self.episodic_mem_top_k = int(self.get_parameter("episodic_mem_top_k").value)
        self.episodic_mem_near_dup_threshold = float(self.get_parameter("episodic_mem_near_dup_threshold").value)

        self.episodic_mem_pool_size = max(1, min(self.episodic_mem_pool_size, 300))
        self.episodic_mem_top_k = max(1, min(self.episodic_mem_top_k, 20))
        self.episodic_mem_near_dup_threshold = max(0.80, min(self.episodic_mem_near_dup_threshold, 0.999))

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

    def _create_episodic_vector_store(self, items, include_failure_cause: bool = False):
        # Same RAG style as node catalog: Document -> Chroma -> similarity_search.
        documents = []
        for item in items:
            if not isinstance(item, dict):
                continue

            mission_goal = str(item.get("mission_goal", ""))
            step_goal = str(item.get("step_goal", ""))
            step_id = int(item.get("step_id", -1)) if str(item.get("step_id", "")).lstrip("-").isdigit() else -1
            bt_xml = str(item.get("bt_xml", ""))
            failure_cause = str(item.get("failure_cause", item.get("reason", "")))

            search_content = f"mission_goal: {mission_goal}\nstep_goal: {step_goal}\nstep_id: {step_id}\n"
            if include_failure_cause:
                search_content += f"failure_cause: {failure_cause}\n"

            documents.append(
                Document(
                    page_content=search_content,
                    metadata={
                        "mission_goal": mission_goal,
                        "step_goal": step_goal,
                        "step_id": step_id,
                        "bt_xml": bt_xml,
                        "failure_cause": failure_cause,
                    },
                )
            )

        if not documents:
            return None

        collection = f"episodic_bt_{'failure' if include_failure_cause else 'success'}_{uuid.uuid4().hex[:8]}"
        return Chroma.from_documents(documents, self.embeddings, collection_name=collection)

    def _truncate_for_log(self, text: str, max_len: int = 90) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."

    def _normalize_for_dedup(self, text: str) -> str:
        # Normalize whitespace and case to make duplicate detection robust.
        return " ".join(str(text or "").strip().lower().split())

    def _build_dedup_signature(self, item: dict, include_failure_cause: bool) -> str:
        mission_goal = self._normalize_for_dedup(item.get("mission_goal", ""))
        step_goal = self._normalize_for_dedup(item.get("step_goal", ""))
        step_id = str(item.get("step_id", "")).strip()
        bt_xml = self._normalize_for_dedup(item.get("bt_xml", ""))
        if include_failure_cause:
            failure_cause = self._normalize_for_dedup(item.get("failure_cause", item.get("reason", "")))
            return f"mg:{mission_goal}\nsg:{step_goal}\nsid:{step_id}\nfx:{failure_cause}\nxml:{bt_xml}"
        return f"mg:{mission_goal}\nsg:{step_goal}\nsid:{step_id}\nxml:{bt_xml}"

    def _deduplicate_episodic_items(self, items, include_failure_cause: bool):
        unique_items = []
        seen_exact = set()
        seen_signatures = []
        removed_exact = 0
        removed_near = 0

        for item in items:
            if not isinstance(item, dict):
                continue

            signature = self._build_dedup_signature(item, include_failure_cause)
            if signature in seen_exact:
                removed_exact += 1
                continue

            is_near_duplicate = False
            for prev_signature in seen_signatures:
                ratio = difflib.SequenceMatcher(None, signature, prev_signature).ratio()
                if ratio >= self.episodic_mem_near_dup_threshold:
                    is_near_duplicate = True
                    break

            if is_near_duplicate:
                removed_near += 1
                continue

            seen_exact.add(signature)
            seen_signatures.append(signature)
            unique_items.append(item)

        return unique_items, removed_exact, removed_near

    def _log_selected_episodic_memories(self, objective: str, episodic: dict, include_failures: bool):
        success_examples = episodic.get("success_examples", []) if isinstance(episodic, dict) else []
        failure_examples = episodic.get("failure_examples", []) if isinstance(episodic, dict) else []

        success_step_goals = [
            str(ep.get("step_goal", "")).strip() for ep in success_examples if str(ep.get("step_goal", "")).strip()
        ]
        failure_step_goals = [
            str(ep.get("step_goal", "")).strip() for ep in failure_examples if str(ep.get("step_goal", "")).strip()
        ]

        success_step_goals_txt = " | ".join(success_step_goals) if success_step_goals else "<none>"
        failure_step_goals_txt = " | ".join(failure_step_goals) if failure_step_goals else "<none>"

        # Keep a concise summary in INFO for normal runs.
        self.get_logger().info("[episodic_mem] ============================================================")
        self.get_logger().info("[episodic_mem] INJECTED EPISODIC CONTEXT")
        self.get_logger().info(
            f"[episodic_mem] success_steps   : {self._truncate_for_log(success_step_goals_txt, 180)}")
        if include_failures:
            self.get_logger().info(
                f"[episodic_mem] failure_steps   : {self._truncate_for_log(failure_step_goals_txt, 180)}")
        else:
            self.get_logger().info("[episodic_mem] failure_steps   : <skipped>")
        self.get_logger().info(f"[episodic_mem] success_count   : {len(success_examples)}")
        self.get_logger().info(
            f"[episodic_mem] failure_count   : {len(failure_examples) if include_failures else 0}")
        self.get_logger().info("[episodic_mem] ============================================================")

        # Keep full episodic detail in DEBUG only.
        if not success_examples:
            self.get_logger().debug("[episodic_mem] success_examples: <none>")
        for idx, ep in enumerate(success_examples, start=1):
            self.get_logger().debug(f"[episodic_mem][success #{idx}] ------------------------------")
            self.get_logger().debug(f"[episodic_mem][success #{idx}] step_id       : {ep.get('step_id', -1)}")
            self.get_logger().debug(
                f"[episodic_mem][success #{idx}] mission_goal  : {self._truncate_for_log(ep.get('mission_goal', ''))}")
            self.get_logger().debug(
                f"[episodic_mem][success #{idx}] step_goal     : {self._truncate_for_log(ep.get('step_goal', ''))}")
            self.get_logger().debug(
                f"[episodic_mem][success #{idx}] bt_xml_len    : {len(str(ep.get('bt_xml', '')))}")
            self.get_logger().debug(
                f"[episodic_mem][success #{idx}] bt_xml_head   : {self._truncate_for_log(ep.get('bt_xml', ''), 100)}")

        if include_failures:
            if not failure_examples:
                self.get_logger().debug("[episodic_mem] failure_examples: <none>")
            for idx, ep in enumerate(failure_examples, start=1):
                self.get_logger().debug(f"[episodic_mem][failure #{idx}] ------------------------------")
                self.get_logger().debug(f"[episodic_mem][failure #{idx}] step_id       : {ep.get('step_id', -1)}")
                self.get_logger().debug(
                    f"[episodic_mem][failure #{idx}] mission_goal  : {self._truncate_for_log(ep.get('mission_goal', ''))}")
                self.get_logger().debug(
                    f"[episodic_mem][failure #{idx}] step_goal     : {self._truncate_for_log(ep.get('step_goal', ''))}")
                self.get_logger().debug(
                    f"[episodic_mem][failure #{idx}] cause         : {self._truncate_for_log(ep.get('failure_cause', ''))}")

    def _retrieve_bt_episodic_memories(self, objective: str, include_failures: bool):
        if not self.mcp_client:
            self.get_logger().debug("[episodic_mem][diag] retrieval skipped: mcp_client unavailable")
            return {"success_examples": [], "failure_examples": []}

        if not self.use_episodic_mem:
            self.get_logger().debug(
                "[episodic_mem][diag] retrieval skipped: use_episodic_mem=false")
            return {"success_examples": [], "failure_examples": []}

        top_success = []
        success_db = None
        try:
            # Pull a bounded pool and retrieve only top-k relevant items via vector search.
            success_mem = self.mcp_client.call_tool(
                "get_bt_success_episodic_memory",
                {"limit": min(self.episodic_mem_pool_size, 100)},
            )
            success_items = success_mem.get("items", []) if isinstance(success_mem, dict) else []
            success_total = success_mem.get("total", len(success_items)) if isinstance(success_mem, dict) else len(success_items)
            success_source = success_mem.get("source", "unknown") if isinstance(success_mem, dict) else "unknown"
            dedup_success_items, removed_exact, removed_near = self._deduplicate_episodic_items(
                success_items,
                include_failure_cause=False,
            )
            self.get_logger().info(
                f"[episodic_mem][diag] success_received: total={success_total} fetched={len(success_items)} "
                f"deduped={len(dedup_success_items)} removed_exact={removed_exact} removed_near={removed_near} "
                f"source={success_source}")

            success_db = self._create_episodic_vector_store(dedup_success_items, include_failure_cause=False)
            if success_db is not None:
                results = success_db.similarity_search(objective, self.episodic_mem_top_k)
                self.get_logger().debug(
                    f"[episodic_mem][diag] success_selected: {len(results)} (top_k={self.episodic_mem_top_k})")
                for res in results:
                    m = res.metadata if isinstance(res.metadata, dict) else {}
                    top_success.append(
                        {
                            "mission_goal": m.get("mission_goal", ""),
                            "step_goal": m.get("step_goal", ""),
                            "step_id": m.get("step_id", -1),
                            "bt_xml": m.get("bt_xml", ""),
                        }
                    )
            else:
                self.get_logger().debug("[episodic_mem][diag] success_selected: 0 (empty vector store)")
        finally:
            if success_db is not None:
                try:
                    success_db.delete_collection()
                except Exception:
                    pass

        top_failures = []
        if include_failures:
            failure_db = None
            try:
                failure_mem = self.mcp_client.call_tool(
                    "get_bt_failure_episodic_memory",
                    {"limit": self.episodic_mem_pool_size},
                )
                failure_items = failure_mem.get("items", []) if isinstance(failure_mem, dict) else []
                failure_total = failure_mem.get("total", len(failure_items)) if isinstance(failure_mem, dict) else len(failure_items)
                failure_source = failure_mem.get("source", "unknown") if isinstance(failure_mem, dict) else "unknown"
                dedup_failure_items, removed_exact, removed_near = self._deduplicate_episodic_items(
                    failure_items,
                    include_failure_cause=True,
                )
                self.get_logger().info(
                    f"[episodic_mem][diag] failure_received: total={failure_total} fetched={len(failure_items)} "
                    f"deduped={len(dedup_failure_items)} removed_exact={removed_exact} removed_near={removed_near} "
                    f"source={failure_source}")

                failure_db = self._create_episodic_vector_store(dedup_failure_items, include_failure_cause=True)
                if failure_db is not None:
                    results = failure_db.similarity_search(objective, self.episodic_mem_top_k)
                    self.get_logger().debug(
                        f"[episodic_mem][diag] failure_selected: {len(results)} (top_k={self.episodic_mem_top_k})")
                    for res in results:
                        m = res.metadata if isinstance(res.metadata, dict) else {}
                        top_failures.append(
                            {
                                "mission_goal": m.get("mission_goal", ""),
                                "step_goal": m.get("step_goal", ""),
                                "step_id": m.get("step_id", -1),
                                "failure_cause": m.get("failure_cause", ""),
                            }
                        )
                else:
                    self.get_logger().debug("[episodic_mem][diag] failure_selected: 0 (empty vector store)")
            finally:
                if failure_db is not None:
                    try:
                        failure_db.delete_collection()
                    except Exception:
                        pass
        else:
            self.get_logger().info(
                "[episodic_mem][diag] failure_received: skipped (include_failures=false)")

        return {
            "success_examples": top_success,
            "failure_examples": top_failures,
        }

    def _build_mcp_block(self, objective: str, include_failures: bool = False) -> str:
        # Gather read-only runtime context from MCP tools.
        if not self.mcp_client:
            return ""
        try:
            capabilities = self.mcp_client.call_tool("get_capabilities", {})
            snapshot = self.mcp_client.call_tool("get_mission_snapshot", {})
            failures = self.mcp_client.call_tool("get_failure_history", {"limit": 20}) if include_failures else {"items": []}
            episodic = self._retrieve_bt_episodic_memories(objective, include_failures)
            self._log_selected_episodic_memories(objective, episodic, include_failures)

            return (
                "\n\n# MCP_CONTEXT (read-only runtime)\n"
                f"mcp_capabilities: {json.dumps(capabilities, ensure_ascii=True)}\n"
                f"mcp_mission_snapshot: {json.dumps(snapshot, ensure_ascii=True)}\n"
                f"mcp_recent_failures: {json.dumps(failures, ensure_ascii=True)}\n"
                f"mcp_bt_success_episodic: {json.dumps(episodic['success_examples'], ensure_ascii=True)}\n"
                f"mcp_bt_failure_episodic: {json.dumps(episodic['failure_examples'], ensure_ascii=True)}\n"
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
            request.objective = request.objective + self._build_mcp_block(
                objective=request.objective,
                include_failures=is_fix,
            )
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
