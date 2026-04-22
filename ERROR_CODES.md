# BT Failure Error Codes Reference

This document defines the structured error codes used by BehaviorTree nodes to report failures. These codes allow the orchestrator to classify errors robustly without relying on text-message parsing.

## Code Table

| Code | Severity | Description | Category | Replan? | FixBT? |
|------|----------|-------------|----------|---------|--------|
| `bt_config_error` | 🔴 Critical | BT node configuration error (missing parameter, invalid type) | Configuration | ❌ | ✅ |
| `execution_error` | 🟠 Moderate | General execution error (default for other failures) | Execution | ✅ | ❌ |
| `service_unavailable` | 🟠 Moderate | Service/action server unavailable or unreachable | Connectivity | ✅ | ❌ |
| `target_not_detected` | 🟡 Mild | Target object/entity was not detected (perception) | Perception | ✅ | ❌ |
| `timeout` | 🟡 Mild | Action exceeded the maximum allowed time | Timing | ✅ | ❌ |

## Orchestrator Decision

The **`llm_plan_orchestrator`** uses the error code to decide between strategies:

```cpp
// In orchestrator's is_local_error():
if (last_failure_code_ == "bt_config_error") {
    // Local error -> Call FixBT (generate an alternative BT)
    return true;
}

// For other codes -> Replan (generate an alternative plan)
```

## Implemented Codes

### ✅ `bt_config_error`

**When to use it:**
- A required parameter is missing in the node configuration
- Incorrect data type in input
- Input port is not connected when it should be

**Example in dummy_bt_nodes (speak.cpp):**
```cpp
if (!text){
    return bt_failure(
        config(),
        registrationName(),
        "missing required input 'text', received: '" + text + "'",
        "bt_config_error"  // <- Structured code
    );
}
```

**When NOT to use it:**
- The parameter is configured correctly but the robot cannot execute (use `execution_error`)
- The service does not exist (use `service_unavailable`)
- The target object does not appear in the camera feed (use `target_not_detected`)

---

### ✅ `execution_error`

**When to use it:**
- Default: used when no other code is specified
- Generic execution error that does not fit other categories
- Internal node failures (unexpected exceptions)

**Example:**
```cpp
return bt_failure(
    config(),
    registrationName(),
    "Failed to execute action: " + std::string(e.what())
    // No code specified -> "execution_error" is used by default
);
```

---

## Planned Codes (TODO)

### 🔄 `service_unavailable`

**Purpose:** Distinguish service/connectivity availability failures from execution errors.

**When to use it:**
- `rclcpp::Service` or `rclcpp::Client` returns a connection error
- Action Server does not respond
- Topic does not exist or publisher/subscriber is not connected
- Timeout while waiting for a service

**Expected locations:**
- `social_bt_nodes/src/bt_nodes/*/...cpp` (nodes that call services)
- `dummy_bt_nodes/src/bt_nodes/*/...cpp` (nodes with ROS I/O)

**Future implementation example:**
```cpp
try {
    auto future = client_->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node_, future)
        != rclcpp::executor::FutureReturnCode::SUCCESS) {
        return bt_failure(config(), registrationName(),
            "Service '" + service_name_ + "' not available",
            "service_unavailable");  // <- New code
    }
} catch (const std::exception& e) {
    return bt_failure(config(), registrationName(),
        "Failed to call service: " + std::string(e.what()),
        "service_unavailable");
}
```

---

### 🔄 `target_not_detected`

**Purpose:** Perception/detection failures (camera, lidar, TF).

**When to use it:**
- Expected object does not appear in image/point-cloud
- Requested transform (TF) does not exist in the TF tree
- Face/person detection returns empty
- Depth map has no data in the target region

**Expected locations:**
- `simple_perception/src/...cpp` (visual perception)
- `social_bt_nodes/src/bt_nodes/*/...cpp` (nodes with detection)

**Future implementation example:**
```cpp
if (!perception_result||perception_result->detections.empty()){
    return bt_failure(config(), registrationName(),
        "Target object not detected in frame",
        "target_not_detected");  // <- New code
}
```

---

### 🔄 `timeout`

**Purpose:** Distinguish timeouts from other execution failures.

**When to use it:**
- Action client reaches timeout while waiting for a response
- Sleep/wait operation expires
- BT.CPP reaches max_iterations

**Expected locations:**
- `social_bt_nodes/src/bt_nodes/*/...cpp` (actions with deadlines)

**Future implementation example:**
```cpp
if (wait_for_service_ready(timeout)) {
    // OK, execute
} else {
    return bt_failure(config(), registrationName(),
        "Service '" + service_name_ + "' did not become ready within "
        + std::to_string(timeout.count()) + "ms",
        "timeout");  // <- New code
}
```

---

## Guide: Add a New Error Code

### 1. Define it in documentation (this file)

Add a row to the "Code Table" and a section under "Planned Codes".

### 2. Implement it in BT nodes

Use the new code when calling `bt_failure()`:

```cpp
#include "bt_failure.hpp"  // Already includes bt_failure()

// In your BT node:
return bt_failure(
    config(),
    registrationName(),
    "Human-readable error message",
    "your_new_code"  // <- Structured code
);
```

### 3. Update orchestrator logic (optional)

If the new code requires a special decision, update `is_local_error()` in [llm_plan_orchestrator.cpp](../behavior_architecture/src/llm_plan_orchestrator.cpp):

```cpp
if (last_failure_code_ == "your_new_code") {
    // Special decision: FixBT or Replan?
    return true;  // true = FixBT, false = Replan
}
```

### 4. Update the table in this document

Change the code status from "🔄 TODO" to "✅ Implemented".

---

## Integration with `bt_failure.hpp`

The helper function `bt_failure()` is defined in:
- 📁 `dummy_bt_nodes/include/bt_nodes/bt_failure.hpp`
- 📁 `social_bt_nodes/include/bt_nodes/bt_failure.hpp`

**Signature:**
```cpp
inline BT::NodeStatus bt_failure(
    const BT::NodeConfig & cfg,
    const std::string & node_name,
    const std::string & reason,
    const std::string & failure_code = "execution_error"
);
```

**What it does:**
1. Writes `reason` to blackboard key `"bt_last_failure"` (human-readable text)
2. Writes `failure_code` to blackboard key `"bt_last_failure_code"` (structured code)
3. Returns `BT::NodeStatus::FAILURE`

When the orchestrator executes the BT, it reads both keys after each cycle.

---

## Reference File: Orchestrator Classification

File: [behavior_architecture/src/llm_plan_orchestrator.cpp](../behavior_architecture/src/llm_plan_orchestrator.cpp)

**Method:** `is_local_error()`
- Reads `last_failure_code_` (populated by `collect_failure_reason()`)
- Checks structured codes first (robust)
- Falls back to substring matching only for internal orchestrator errors (XML parsing)

**Method:** `collect_failure_reason()`
- Reads `"bt_last_failure"` (text) from blackboard
- Reads `"bt_last_failure_code"` (code) from blackboard
- Stores it in `last_failure_code_` for later decisions
- Clears both keys after reading

---

## Design Notes

**Why codes + text?**
- **Codes**: Robust against wording changes (does not depend on exact strings)
- **Text**: Human-readable for logging, debugging, and LLM prompts

**Why NOT regex in the orchestrator?**
- Regex is fragile: if a node developer changes message wording, regex fails
- Codes are explicit: the node developer chooses the code; if it changes, it is intentional

**Why default `"execution_error"`?**
- Backward-compatible: legacy nodes calling `bt_failure()` without a code still work
- Safe: unknown errors are not classified as config errors (worse: false negative)

---

## Checklist: Before Merging New Error-Code Changes

- [ ] Does the node use a code from this table?
- [ ] Does the code accurately describe the failure cause?
- [ ] Did I update `ERROR_CODES.md` if I added a new code?
- [ ] Did I update `llm_plan_orchestrator.cpp` if decision logic changed?
- [ ] Did it compile without warnings? (`colcon build --packages-select dummy_bt_nodes social_bt_nodes`)
- [ ] Did I test that the BT fails correctly and the orchestrator classifies the error?

---

## Contact / Questions

If you are unsure which code to use, check this table or ask the team in the PR.

**Goal:** Ensure every failure is classifiable without parsing text. 🎯
