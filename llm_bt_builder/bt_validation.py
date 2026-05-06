import re
import xml.etree.ElementTree as ET

import yaml


class BTValidation:
    def _parse_structural_required_ports(self, *yaml_contents):
        required = {}
        for content in yaml_contents:
            try:
                data = yaml.safe_load(content)
                for node in data.get('bt_nodes', []):
                    ports = node.get('ports', []) or []
                    req = [
                        p.get('name') or p.get('key')
                        for p in ports
                        if p.get('name') or p.get('key')
                    ]
                    if req:
                        required[node['name']] = req
            except Exception:
                pass
        return required

    def _parse_capability_specs(self, yaml_content):
        specs = {}
        try:
            data = yaml.safe_load(yaml_content)
            for node in data.get('bt_nodes', []):
                raw_ports = node.get('ports', []) or []
                current_ports = []
                required_inputs = []
                input_ports = set()
                output_ports = set()
                return_statuses = set()

                for port in raw_ports:
                    port_name = port.get('key') or port.get('name')
                    if not port_name:
                        continue
                    current_ports.append(port_name)

                    direction = str(port.get('direction', '')).strip().lower()
                    description = str(port.get('description', '')).strip().lower()
                    if direction == 'input':
                        input_ports.add(port_name)
                    elif direction == 'output':
                        output_ports.add(port_name)

                    if direction == 'input' and 'required' in description:
                        required_inputs.append(port_name)

                raw_returns = node.get('return', {})
                if isinstance(raw_returns, dict):
                    for status_name in raw_returns.keys():
                        if isinstance(status_name, str):
                            return_statuses.add(status_name.strip().upper())

                specs[node['name']] = {
                    'ports': current_ports,
                    'required_inputs': required_inputs,
                    'input_ports': input_ports,
                    'output_ports': output_ports,
                    'type': str(node.get('type', '')).strip().lower(),
                    'return_statuses': return_statuses,
                }
        except Exception:
            return {}
        return specs

    def _extract_known_blackboard_vars(self, objective_text):
        known = set()
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return known

        def collect(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ('available_blackboard_vars', 'inputs') and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                known.add(item.strip())
                    collect(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return known

    def _extract_required_output_vars(self, objective_text):
        required = set()
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return required

        def collect(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'outputs' and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                required.add(item.strip())
                    collect(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return required

    def _coerce_xml_root(self, xml_input):
        if isinstance(xml_input, ET.Element):
            return xml_input
        return ET.fromstring(xml_input)

    def validate_xml_bt(self, xml_input):
        try:
            root = self._coerce_xml_root(xml_input)

            for elem in root.iter():
                children = list(elem)

                if not isinstance(elem.tag, str):
                    continue

                if elem.tag in ['root', 'BehaviorTree']:
                    if len(children) != 1:
                        return False, f"<{elem.tag}> must have exactly 1 child, found {len(children)}"
                elif elem.tag in self.decorators:
                    if len(children) != 1:
                        return False, f"Decorator <{elem.tag}> must have exactly 1 child, found {len(children)}"
                elif elem.tag in self.control_nodes:
                    min_children_by_tag = {
                        'IfThenElse': 3,
                    }
                    min_children = min_children_by_tag.get(elem.tag, 2)
                    if len(children) < min_children:
                        return False, (
                            f"Control node <{elem.tag}> must have at least {min_children} children, "
                            f"found {len(children)}. Control nodes with a single child are structurally valid "
                            f"in XML but semantically pointless in this project."
                        )
                elif elem.tag in ['AlwaysSuccess', 'AlwaysFailure']:
                    if len(children) > 0:
                        return False, f"<{elem.tag}> should not have children, found {len(children)}"

                for req_port in getattr(self, 'structural_required_ports', {}).get(elem.tag, []):
                    if req_port not in elem.attrib:
                        return False, (
                            f"<{elem.tag}> is missing required attribute '{req_port}'. "
                            f"Add it, e.g. {req_port}=\"1\"."
                        )

            for parent_tag in ('Sequence', 'ReactiveSequence'):
                for seq in root.iter(parent_tag):
                    for child in list(seq):
                        if isinstance(child.tag, str) and child.tag == 'Repeat':
                            num_cycles = str(child.attrib.get('num_cycles', '')).strip()
                            if num_cycles == '-1':
                                return False, (
                                    f'<Repeat num_cycles="-1"> inside <{parent_tag}> never returns SUCCESS '
                                    f'and permanently blocks the parent from completing. '
                                    f'Use <ReactiveSequence> with a condition guard (e.g. <IsDetected>) '
                                    f'for continuous tracking/following loops instead.'
                                )

            return True, 'OK'
        except Exception as e:
            return False, str(e)

    def validate_bt_semantics(
        self,
        xml_input,
        node_specs,
        known_bb_vars=None,
        required_outputs=None,
        recovery_policy=None,
    ):
        try:
            root = self._coerce_xml_root(xml_input)
            known_bb_vars = set(known_bb_vars or [])
            required_outputs = set(required_outputs or [])
            produced_in_tree = set()
            parent_map = {child: parent for parent in root.iter() for child in list(parent)}
            recovery_policy = recovery_policy or {'required': False, 'loop_required': False, 'retry_attempts': None}

            for elem in root.iter():
                if elem.tag in self.structural_nodes:
                    continue

                if elem.tag not in node_specs:
                    return False, f"Node <{elem.tag}> does NOT exist in the capabilities YAML."

                spec = node_specs[elem.tag]
                allowed_ports = spec.get('ports', [])
                required_inputs = spec.get('required_inputs', [])
                input_ports = spec.get('input_ports', set())
                output_ports = spec.get('output_ports', set())

                missing_required = []
                for req in required_inputs:
                    if req not in elem.attrib or str(elem.attrib.get(req, '')).strip() == '':
                        missing_required.append(req)
                if missing_required:
                    return False, (
                        f"Node <{elem.tag}> is missing required input port(s): {missing_required}. "
                        f"Provide explicit values for those attributes."
                    )

                for attr in elem.attrib:
                    if attr in ['name', 'ID']:
                        continue
                    if attr not in allowed_ports:
                        return False, f"Node <{elem.tag}> has an illegal port: '{attr}'. Allowed: {allowed_ports}"

                    value = str(elem.attrib[attr]).strip()
                    if '{' in value or '}' in value:
                        refs = re.findall(r'\{[^{}]+\}', value)
                        if len(refs) != 1 or refs[0] != value:
                            return False, (
                                f"Node <{elem.tag}> has malformed blackboard reference in port '{attr}': '{value}'. "
                                f"Use exactly one blackboard variable like '{{my_var}}', or a plain literal."
                            )

                        bb_key = value[1:-1]
                        if ',' in bb_key or ';' in bb_key:
                            return False, (
                                f"Node <{elem.tag}> port '{attr}' uses an invalid blackboard key '{bb_key}'. "
                                f"Do not concatenate multiple variables inside one {{}}. "
                                f"Write to a single combined variable first, then pass that variable."
                            )

                        is_read_ref = (attr in input_ports) or (attr not in output_ports)
                        if is_read_ref and bb_key not in known_bb_vars and bb_key not in produced_in_tree:
                            return False, (
                                f"Node <{elem.tag}> reads unknown blackboard key '{bb_key}' in input port '{attr}'. "
                                f"Declare it in objective inputs/available_blackboard_vars or write it earlier in this step."
                            )

                        if attr in output_ports:
                            produced_in_tree.add(bb_key)
                    elif attr in output_ports and value:
                        literal_key = value.strip()
                        if literal_key.startswith('{') and literal_key.endswith('}'):
                            produced_in_tree.add(literal_key[1:-1])

            missing_outputs = required_outputs - produced_in_tree
            if missing_outputs:
                missing = sorted(missing_outputs)
                return False, (
                    f"BT did not write required output blackboard key(s): {missing}. "
                    f"Map node output ports to these exact keys."
                )

            loop_controls = {'RetryUntilSuccessful', 'Repeat'}
            has_loop = any(
                isinstance(elem.tag, str) and elem.tag in loop_controls
                for elem in root.iter()
            )

            retry_nodes = [
                elem for elem in root.iter()
                if isinstance(elem.tag, str) and elem.tag == 'RetryUntilSuccessful'
            ]
            retry_control_allowed = (
                recovery_policy.get('required', False) or
                recovery_policy.get('loop_required', False)
            )

            if not retry_control_allowed and retry_nodes:
                return False, (
                    'Objective recovery_policy has required=false and loop_until_success=false, '
                    'so BT must not use RetryUntilSuccessful for this step.'
                )

            expected_retry_attempts = recovery_policy.get('retry_attempts', None)
            if retry_control_allowed and expected_retry_attempts is not None:
                if not retry_nodes:
                    return False, (
                        'Objective recovery_policy.retry_attempts is set for a retry-enabled step, but BT has no RetryUntilSuccessful node. '
                        'Use RetryUntilSuccessful with num_attempts matching recovery_policy.retry_attempts.'
                    )

                has_expected_retry = False
                for retry_node in retry_nodes:
                    raw_num = str(retry_node.attrib.get('num_attempts', '')).strip()
                    try:
                        current = int(raw_num)
                    except ValueError:
                        continue

                    if expected_retry_attempts == 'forever' and current == -1:
                        has_expected_retry = True
                        break
                    if isinstance(expected_retry_attempts, int) and current == expected_retry_attempts:
                        has_expected_retry = True
                        break

                if not has_expected_retry:
                    expected_value = '-1' if expected_retry_attempts == 'forever' else str(expected_retry_attempts)
                    return False, (
                        'Objective recovery_policy.retry_attempts requires '
                        f'RetryUntilSuccessful num_attempts="{expected_value}", '
                        'but BT uses a different value.'
                    )

            condition_tags = {
                name for name, spec in node_specs.items()
                if spec.get('type') == 'condition'
            }

            if recovery_policy.get('required', False):
                condition_nodes = [
                    elem for elem in root.iter()
                    if isinstance(elem.tag, str) and elem.tag in condition_tags
                ]

                if condition_nodes:
                    if recovery_policy.get('loop_required', False) and not has_loop:
                        return False, (
                            'Objective recovery_policy requires loop_until_success, but BT has no retry loop control. '
                            'Wrap check+recovery logic with RetryUntilSuccessful (or Repeat).'
                        )

                    branching_controls = {'Fallback', 'ReactiveFallback'}

                    def has_branching_ancestor(node):
                        parent = parent_map.get(node)
                        while parent is not None:
                            if isinstance(parent.tag, str) and parent.tag in branching_controls:
                                return True
                            parent = parent_map.get(parent)
                        return False

                    for cond in condition_nodes:
                        if not has_branching_ancestor(cond):
                            return False, (
                                f'Recovery branch missing for condition <{cond.tag}>. '
                                'For recoverable checks, place conditions under Fallback/ReactiveFallback '
                                'with an explicit failure-recovery branch.'
                            )

            blocking_controls = ('Sequence', 'Fallback', 'ReactiveSequence')

            def unwrap_leaf(node):
                current = node
                chain = []
                while isinstance(current.tag, str) and current.tag in self.decorators:
                    children = [c for c in list(current) if isinstance(c.tag, str)]
                    if len(children) != 1:
                        break
                    chain.append(current.tag)
                    current = children[0]
                return current, chain

            def running_only_effective(node):
                leaf, _ = unwrap_leaf(node)
                if not isinstance(leaf.tag, str):
                    return False
                if leaf.tag in self.structural_nodes:
                    return False
                spec = node_specs.get(leaf.tag)
                if not spec:
                    return False
                statuses = set(spec.get('return_statuses', set()))
                return statuses == {'RUNNING'}

            for seq in root.iter('Sequence'):
                children = [c for c in list(seq) if isinstance(c.tag, str)]
                for child in children:
                    if running_only_effective(child):
                        leaf, chain = unwrap_leaf(child)
                        label = f"{'/'.join(chain)} -> {leaf.tag}" if chain else leaf.tag
                        return False, (
                            f'Plain <Sequence> contains RUNNING-only node <{label}>. '
                            'A non-reactive Sequence with any RUNNING-only child can block indefinitely. '
                            'Use <ReactiveSequence> or redesign the flow so that child can finish '
                            'with SUCCESS/FAILURE.'
                        )

            for fallback in root.iter('Fallback'):
                children = [c for c in list(fallback) if isinstance(c.tag, str)]
                for child in children:
                    if running_only_effective(child):
                        leaf, chain = unwrap_leaf(child)
                        label = f"{'/'.join(chain)} -> {leaf.tag}" if chain else leaf.tag
                        return False, (
                            f'Plain <Fallback> contains RUNNING-only node <{label}>. '
                            'A non-reactive Fallback with any RUNNING-only child can block indefinitely. '
                            'Use <ReactiveFallback> or redesign the flow so that child can finish '
                            'with SUCCESS/FAILURE.'
                        )

            for control_tag in blocking_controls:
                for control in root.iter(control_tag):
                    children = [c for c in list(control) if isinstance(c.tag, str)]
                    for child in children[:-1]:
                        if running_only_effective(child):
                            leaf, chain = unwrap_leaf(child)
                            chain_prefix = ('/'.join(chain) + ' -> ') if chain else ''
                            return False, (
                                f'Node <{chain_prefix}{leaf.tag}> can only return RUNNING and appears before other '
                                f'children inside <{control_tag}>, so later nodes are unreachable. '
                                'Move it to the end of that control node or redesign with non-blocking flow.'
                            )

            reevaluation_controls = ('Sequence', 'Fallback')
            for control_tag in reevaluation_controls:
                for control in root.iter(control_tag):
                    children = [c for c in list(control) if isinstance(c.tag, str)]
                    for index, child in enumerate(children[1:], start=1):
                        if not running_only_effective(child):
                            continue

                        earlier_conditions = []
                        for sibling in children[:index]:
                            leaf, chain = unwrap_leaf(sibling)
                            if isinstance(leaf.tag, str) and leaf.tag in condition_tags:
                                label = f"{'/'.join(chain)} -> {leaf.tag}" if chain else leaf.tag
                                earlier_conditions.append(label)
                        if not earlier_conditions:
                            continue

                        running_leaf, running_chain = unwrap_leaf(child)
                        running_label = (
                            f"{'/'.join(running_chain)} -> {running_leaf.tag}"
                            if running_chain else running_leaf.tag
                        )

                        if control_tag == 'Sequence':
                            return False, (
                                f'Plain <Sequence> contains condition node(s) {earlier_conditions} before '
                                f'RUNNING-only node <{running_label}>. Those conditions will not be rechecked '
                                'while the running action is active. Use <ReactiveSequence> if those '
                                'conditions must stay live during execution.'
                            )

                        return False, (
                            f'Plain <Fallback> contains condition node(s) {earlier_conditions} before '
                            f'RUNNING-only node <{running_label}>. Once <{running_label}> is running, the earlier '
                            'conditions may stop being rechecked. Use <ReactiveFallback> when those '
                            'conditions must be reevaluated while the fallback action runs.'
                        )

            return True, 'OK'
        except Exception as e:
            return False, str(e)