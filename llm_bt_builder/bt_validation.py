import re
import xml.etree.ElementTree as ET

import yaml


class BTValidation:
    def _signals_forced_plan_termination(self, description_text, return_text):
        text = f"{description_text} {return_text}".lower()
        # Generic semantic detector: capability that explicitly emits a forced
        # failure code / restart signal for whole-plan termination.
        markers = [
            'force fail',
            'force plan fail',
            'forced plan',
            'restart plan',
            'restart the plan',
        ]
        return any(marker in text for marker in markers)

    def _normalize_port_type(self, raw_type):
        if not raw_type:
            return None

        normalized = str(raw_type).strip().lower().replace(' ', '')
        if not normalized:
            return None

        if normalized.startswith('std::'):
            normalized = normalized[5:]

        aliases = {
            'string': 'string',
            'std::string': 'string',
            'bool': 'bool',
            'boolean': 'bool',
            'int': 'int',
            'integer': 'int',
            'float': 'float',
            'double': 'float',
            'number': 'float',
        }

        if normalized in ('any', 'auto', 'variant', 'blackboard::any'):
            return None

        return aliases.get(normalized, normalized)

    def _register_blackboard_key_type(self, key_types, bb_key, port_type, elem_tag, attr):
        if not bb_key or not port_type:
            return True, None

        existing = key_types.get(bb_key)
        if existing is None:
            key_types[bb_key] = {
                'type': port_type,
                'node': elem_tag,
                'port': attr,
            }
            return True, None

        if existing['type'] != port_type:
            return False, (
                f"Blackboard key '{{{bb_key}}}' has incompatible types: "
                f"'{existing['type']}' from <{existing['node']}>.{existing['port']} "
                f"and '{port_type}' from <{elem_tag}>.{attr}."
            )

        return True, None

    def _possible_statuses(self, node, node_specs):
        if not isinstance(node.tag, str):
            return set()

        children = [c for c in list(node) if isinstance(c.tag, str)]

        if node.tag == 'root' or node.tag == 'BehaviorTree':
            if len(children) != 1:
                return set()
            return self._possible_statuses(children[0], node_specs)

        if node.tag in self.decorators:
            if len(children) != 1:
                return set()
            child_statuses = self._possible_statuses(children[0], node_specs)
            if node.tag == 'Inverter':
                mapped = set()
                if 'SUCCESS' in child_statuses:
                    mapped.add('FAILURE')
                if 'FAILURE' in child_statuses:
                    mapped.add('SUCCESS')
                if 'RUNNING' in child_statuses:
                    mapped.add('RUNNING')
                return mapped
            if node.tag == 'ForceSuccess':
                mapped = {'SUCCESS'} if child_statuses & {'SUCCESS', 'FAILURE'} else set()
                if 'RUNNING' in child_statuses:
                    mapped.add('RUNNING')
                return mapped
            if node.tag == 'ForceFailure':
                mapped = {'FAILURE'} if child_statuses & {'SUCCESS', 'FAILURE'} else set()
                if 'RUNNING' in child_statuses:
                    mapped.add('RUNNING')
                return mapped
            return child_statuses

        if node.tag in self.control_nodes:
            child_sets = [self._possible_statuses(child, node_specs) for child in children]

            if node.tag in ('Sequence', 'ReactiveSequence'):
                if not child_sets:
                    return set()
                result = set()
                can_reach_current = True
                all_success_possible = True
                for child_statuses in child_sets:
                    if not child_statuses:
                        all_success_possible = False
                        break
                    if can_reach_current and 'FAILURE' in child_statuses:
                        result.add('FAILURE')
                    if can_reach_current and 'RUNNING' in child_statuses:
                        result.add('RUNNING')
                    can_reach_current = can_reach_current and ('SUCCESS' in child_statuses)
                    all_success_possible = all_success_possible and ('SUCCESS' in child_statuses)
                if all_success_possible:
                    result.add('SUCCESS')
                return result

            if node.tag in ('Fallback', 'ReactiveFallback'):
                if not child_sets:
                    return set()
                result = set()
                can_reach_current = True
                all_failure_possible = True
                for child_statuses in child_sets:
                    if not child_statuses:
                        all_failure_possible = False
                        break
                    if can_reach_current and 'SUCCESS' in child_statuses:
                        result.add('SUCCESS')
                    if can_reach_current and 'RUNNING' in child_statuses:
                        result.add('RUNNING')
                    can_reach_current = can_reach_current and ('FAILURE' in child_statuses)
                    all_failure_possible = all_failure_possible and ('FAILURE' in child_statuses)
                if all_failure_possible:
                    result.add('FAILURE')
                return result

            if node.tag == 'Parallel':
                result = set()
                if any('RUNNING' in s for s in child_sets):
                    result.add('RUNNING')
                if all('SUCCESS' in s for s in child_sets):
                    result.add('SUCCESS')
                if any('FAILURE' in s for s in child_sets):
                    result.add('FAILURE')
                return result

            if node.tag == 'IfThenElse':
                if len(child_sets) != 3:
                    return set()
                cond, then_set, else_set = child_sets
                result = set()
                if 'SUCCESS' in cond:
                    result |= then_set
                if 'FAILURE' in cond:
                    result |= else_set
                if 'RUNNING' in cond:
                    result.add('RUNNING')
                return result

        spec = node_specs.get(node.tag, {})
        return set(spec.get('return_statuses', set()))

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
                port_types = {}
                return_statuses = set()
                node_description = str(node.get('description', '')).strip()
                return_text = ""

                for port in raw_ports:
                    port_name = port.get('key') or port.get('name')
                    if not port_name:
                        continue
                    current_ports.append(port_name)

                    direction = str(port.get('direction', '')).strip().lower()
                    description = str(port.get('description', '')).strip().lower()
                    parsed_type = self._normalize_port_type(port.get('type', ''))
                    if parsed_type:
                        port_types[port_name] = parsed_type
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
                    return_text = " ".join(str(v) for v in raw_returns.values())

                specs[node['name']] = {
                    'ports': current_ports,
                    'required_inputs': required_inputs,
                    'input_ports': input_ports,
                    'output_ports': output_ports,
                    'port_types': port_types,
                    'type': str(node.get('type', '')).strip().lower(),
                    'description': node_description,
                    'signals_forced_plan_termination': self._signals_forced_plan_termination(
                        node_description, return_text),
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
                    if key == 'available_blackboard_vars_typed' and isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                typed_key = str(item.get('key', '')).strip()
                                if typed_key:
                                    known.add(typed_key)
                    collect(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return known

    def _extract_known_blackboard_var_types(self, objective_text):
        known_types = {}
        try:
            data = yaml.safe_load(objective_text)
        except Exception:
            return known_types

        def collect(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'available_blackboard_vars_typed' and isinstance(value, list):
                        for item in value:
                            if not isinstance(item, dict):
                                continue
                            typed_key = str(item.get('key', '')).strip()
                            typed_value = self._normalize_port_type(item.get('type', ''))
                            if typed_key and typed_value:
                                known_types[typed_key] = typed_value
                    collect(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        collect(data)
        return known_types

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
        structural_base_hint = (
            "Return a COMPLETE XML from scratch. "
            "Do not emit empty structural nodes. "
            "Every control node must have the required minimum children, "
            "every decorator must wrap exactly one child, "
            "and root/BehaviorTree arity constraints must be satisfied. "
            "Never use self-closing structural tags for control/decorator nodes."
        )

        try:
            root = self._coerce_xml_root(xml_input)

            for elem in root.iter():
                children = list(elem)

                if not isinstance(elem.tag, str):
                    continue

                if elem.tag in ['root', 'BehaviorTree']:
                    if len(children) != 1:
                        return False, f"<{elem.tag}> must have exactly 1 child, found {len(children)}", (
                            f"{structural_base_hint} "
                            "Rebuild the tree skeleton as "
                            "<root><BehaviorTree ID=\"MainTree\"> ... </BehaviorTree></root> "
                            "with exactly one child at each structural root level."
                        )
                elif elem.tag in self.decorators:
                    if len(children) != 1:
                        return False, f"Decorator <{elem.tag}> must have exactly 1 child, found {len(children)}", (
                            f"{structural_base_hint} "
                            f"Decorator <{elem.tag}> must wrap exactly one child. "
                            f"Wrap one valid subtree inside <{elem.tag}> ... </{elem.tag}>. "
                            f"Never output <{elem.tag}/> or <{elem.tag}></{elem.tag}>."
                        )
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
                        ), (
                            f"{structural_base_hint} "
                            f"Add valid child nodes inside <{elem.tag}> ... </{elem.tag}> "
                            f"or remove that wrapper if unnecessary. "
                            f"Never output <{elem.tag}/> or <{elem.tag}></{elem.tag}>."
                        )
                elif elem.tag in ['AlwaysSuccess', 'AlwaysFailure']:
                    if len(children) > 0:
                        return False, f"<{elem.tag}> should not have children, found {len(children)}", (
                            f"{structural_base_hint} "
                            f"Remove all children from <{elem.tag}>; "
                            f"it must be a childless structural wrapper."
                        )

                for req_port in getattr(self, 'structural_required_ports', {}).get(elem.tag, []):
                    if req_port not in elem.attrib:
                        return False, (
                            f"<{elem.tag}> is missing required attribute '{req_port}'. "
                            f"Add it, e.g. {req_port}=\"1\"."
                        ), (
                            f"{structural_base_hint} "
                            f"Add '{req_port}' with a valid value or {{blackboard_key}} reference "
                            f"to <{elem.tag}> before returning XML."
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
                                    f'Use <ReactiveSequence> with a condition guard) '
                                    f'for continuous tracking/following loops instead.'
                                ), (
                                    f"{structural_base_hint} "
                                    f'Replace <Repeat num_cycles="-1"> with a <ReactiveSequence> that '
                                    f'includes a condition guard to break the loop.'
                                )

            return True, 'OK', ''
        except Exception as e:
            return False, str(e), structural_base_hint



    def validate_bt_semantics(
        self,
        xml_input,
        node_specs,
        known_bb_vars=None,
        known_bb_var_types=None,
        required_outputs=None,
        recovery_policy=None,
        allow_forced_plan_fail=None,
    ):
        try:
            root = self._coerce_xml_root(xml_input)
            known_bb_vars = set(known_bb_vars or [])
            known_bb_var_types = dict(known_bb_var_types or {})
            required_outputs = set(required_outputs or [])
            allow_forced_plan_fail = bool(allow_forced_plan_fail) if allow_forced_plan_fail is not None else None
            produced_in_tree = set()
            key_types = {}
            parent_map = {child: parent for parent in root.iter() for child in list(parent)}
            recovery_policy = recovery_policy or {'required': False, 'loop_required': False, 'retry_attempts': None}

            for elem in root.iter():
                if elem.tag in self.structural_nodes:
                    continue

                if elem.tag not in node_specs:
                    return False, f"Node <{elem.tag}> does NOT exist in the capabilities YAML.", (
                        "Replace every unknown node with valid capabilities-only nodes. "
                        "Do not invent helper/intermediate nodes."
                    )

                spec = node_specs[elem.tag]
                if (
                    allow_forced_plan_fail is False and
                    spec.get('signals_forced_plan_termination', False)
                ):
                    return False, (
                        f"Node <{elem.tag}> signals forced whole-plan termination, but "
                        "objective.allow_forced_plan_fail is false for this step."
                    ), (
                        "Remove this forced-termination node from the step, or set "
                        "objective.allow_forced_plan_fail=true only if mission policy explicitly allows it."
                    )
                allowed_ports = spec.get('ports', [])
                required_inputs = spec.get('required_inputs', [])
                input_ports = spec.get('input_ports', set())
                output_ports = spec.get('output_ports', set())
                port_types = spec.get('port_types', {})

                missing_required = []
                for req in required_inputs:
                    if req not in elem.attrib or str(elem.attrib.get(req, '')).strip() == '':
                        missing_required.append(req)
                if missing_required:
                    return False, (
                        f"Node <{elem.tag}> is missing required input port(s): {missing_required}. "
                        f"Provide explicit values for those attributes."
                    ), (
                        f"Add the missing port(s) {missing_required} to <{elem.tag}> "
                        f"with explicit values or {{{{blackboard_key}}}} references."
                    )

                for attr in elem.attrib:
                    if attr in ['name', 'ID']:
                        continue
                    if attr not in allowed_ports:
                        return False, f"Node <{elem.tag}> has an illegal port: '{attr}'. Allowed: {allowed_ports}", (
                            f"Remove the illegal port '{attr}' from <{elem.tag}>. "
                            f"Only use ports from: {allowed_ports}."
                        )

                    value = str(elem.attrib[attr]).strip()
                    if '{' in value or '}' in value:
                        refs = re.findall(r'\{[^{}]+\}', value)
                        if len(refs) != 1 or refs[0] != value:
                            return False, (
                                f"Node <{elem.tag}> has malformed blackboard reference in port '{attr}': '{value}'. "
                                f"Use exactly one blackboard variable like '{{my_var}}', or a plain literal."
                            ), (
                                "Use exactly ONE blackboard variable per attribute "
                                "(e.g., text=\"{full_order}\"). "
                                "Do NOT mix literals with placeholders or concatenate variables."
                            )

                        bb_key = value[1:-1]
                        if ',' in bb_key or ';' in bb_key:
                            return False, (
                                f"Node <{elem.tag}> port '{attr}' uses an invalid blackboard key '{bb_key}'. "
                                f"Do not concatenate multiple variables inside one {{}}. "
                                f"Write to a single combined variable first, then pass that variable."
                            ), (
                                "Write to a single combined variable first using a dedicated node, "
                                "then pass that single variable as a blackboard reference."
                            )

                        is_read_ref = (attr in input_ports) or (attr not in output_ports)
                        if is_read_ref and bb_key not in known_bb_vars and bb_key not in produced_in_tree:
                            return False, (
                                f"Node <{elem.tag}> reads unknown blackboard key '{bb_key}' in input port '{attr}'. "
                                f"Declare it in objective inputs/available_blackboard_vars or write it earlier in this step."
                            ), (
                                "Only read variables from objective inputs/available_blackboard_vars "
                                "or variables written earlier in this step. "
                                "Do not invent helper variables — declare them in inputs or write them first."
                            )

                        port_type = port_types.get(attr)
                        known_type = known_bb_var_types.get(bb_key)
                        if known_type and port_type and known_type != port_type:
                            return False, (
                                f"Blackboard key '{{{bb_key}}}' is declared as type '{known_type}' "
                                f"in available_blackboard_vars_typed, but <{elem.tag}>.{attr} expects '{port_type}'. "
                                "This causes BT port type conflicts at runtime."
                            ), (
                                f"Use a different key for <{elem.tag}>.{attr} or align the port type with "
                                f"the declared blackboard type '{known_type}'."
                            )

                        ok_type, type_err = self._register_blackboard_key_type(
                            key_types,
                            bb_key,
                            port_type,
                            elem.tag,
                            attr,
                        )
                        if not ok_type:
                            return False, (
                                f"{type_err} This causes BT port type conflicts at runtime."
                            ), (
                                f"Use different blackboard keys for incompatible port types, "
                                f"or align node port types when sharing '{{{bb_key}}}'."
                            )

                        if attr in output_ports:
                            produced_in_tree.add(bb_key)
                    elif attr in output_ports and value:
                        literal_key = value.strip()
                        if literal_key.startswith('{') and literal_key.endswith('}'):
                            bb_key = literal_key[1:-1]
                            produced_in_tree.add(bb_key)

                            port_type = port_types.get(attr)
                            known_type = known_bb_var_types.get(bb_key)
                            if known_type and port_type and known_type != port_type:
                                return False, (
                                    f"Blackboard key '{{{bb_key}}}' is declared as type '{known_type}' "
                                    f"in available_blackboard_vars_typed, but <{elem.tag}>.{attr} writes '{port_type}'. "
                                    "This causes BT port type conflicts at runtime."
                                ), (
                                    f"Write <{elem.tag}>.{attr} to a key with type '{port_type}', or update "
                                    f"the declared type for '{{{bb_key}}}' if that key is intended to store '{port_type}'."
                                )

                            ok_type, type_err = self._register_blackboard_key_type(
                                key_types,
                                bb_key,
                                port_type,
                                elem.tag,
                                attr,
                            )
                            if not ok_type:
                                return False, (
                                    f"{type_err} This causes BT port type conflicts at runtime."
                                ), (
                                    f"Use different blackboard keys for incompatible port types, "
                                    f"or align node port types when sharing '{{{bb_key}}}'."
                                )

            missing_outputs = required_outputs - produced_in_tree
            if missing_outputs:
                missing = sorted(missing_outputs)
                return False, (
                    f"BT did not write required output blackboard key(s): {missing}. "
                    f"Map node output ports to these exact keys."
                ), (
                    f"Map the output port(s) of the appropriate leaf node(s) to these exact keys: {missing}. "
                    f"The port must be an output port bound with {{{{key}}}} notation."
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
                ), (
                    "Remove RetryUntilSuccessful and keep a plain Sequence "
                    "unless objective.recovery_policy.required=true or loop_until_success=true."
                )

            expected_retry_attempts = recovery_policy.get('retry_attempts', None)
            if retry_control_allowed and expected_retry_attempts is not None:
                if not retry_nodes:
                    return False, (
                        'Objective recovery_policy.retry_attempts is set for a retry-enabled step, but BT has no RetryUntilSuccessful node. '
                        'Use RetryUntilSuccessful with num_attempts matching recovery_policy.retry_attempts.'
                    ), (
                        f"Add RetryUntilSuccessful with num_attempts matching "
                        f"recovery_policy.retry_attempts ({expected_retry_attempts})."
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
                    ), (
                        f"Set RetryUntilSuccessful num_attempts=\"{expected_value}\" "
                        f"to exactly match recovery_policy.retry_attempts."
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
                        ), (
                            'Add RetryUntilSuccessful or Repeat around the recoverable branch to satisfy '
                            'recovery_policy.loop_until_success=true.'
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
                            ), (
                                "Wrap conditions inside Fallback/ReactiveFallback with an explicit "
                                "recovery action branch. recovery_policy.required=true demands "
                                "explicit branching with a fallback path."
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
                        ), (
                            "Wrap the RUNNING-only node inside a <ReactiveSequence> together with a "
                            "condition that can interrupt it (e.g. <IsDetected/>). "
                            "The condition is re-ticked each cycle and can halt the action when appropriate."
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
                        ), (
                            "Use <ReactiveFallback> instead of plain <Fallback>, "
                            "or redesign the flow so the node can eventually return SUCCESS or FAILURE."
                        )

            for control_tag in blocking_controls:
                for control in root.iter(control_tag):
                    children = [c for c in list(control) if isinstance(c.tag, str)]
                    for child in children[:-1]:
                        if running_only_effective(child):
                            leaf, chain = unwrap_leaf(child)
                            chain_prefix = ('/'.join(chain) + ' -> ') if chain else ''
                            if control_tag == 'ReactiveSequence':
                                return False, (
                                    f'Node <{chain_prefix}{leaf.tag}> can only return RUNNING and appears before other '
                                    f'children inside <{control_tag}>, so later nodes are unreachable. '
                                    'Move it to the end of that control node or redesign with non-blocking flow.'
                                ), (
                                    "A node that only returns RUNNING inside a ReactiveSequence blocks all following siblings. "
                                    "ReactiveSequence stops at the first RUNNING child and does not advance further. "
                                    "If you need to keep ticking a condition while an action runs, use "
                                    "ReactiveFallback with the condition FIRST: "
                                    "<ReactiveFallback> <IsCondition/> <LongRunningAction/> </ReactiveFallback>."
                                )
                            return False, (
                                f'Node <{chain_prefix}{leaf.tag}> can only return RUNNING and appears before other '
                                f'children inside <{control_tag}>, so later nodes are unreachable. '
                                'Move it to the end of that control node or redesign with non-blocking flow.'
                            ), (
                                f"Move <{leaf.tag}> to the LAST position inside <{control_tag}>, "
                                f"or use <ReactiveSequence>/<ReactiveFallback> "
                                f"so later siblings remain reachable."
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
                            ), (
                                "Change <Sequence> to <ReactiveSequence> so conditions are "
                                "re-ticked every BT tick while the action is RUNNING."
                            )

                        return False, (
                            f'Plain <Fallback> contains condition node(s) {earlier_conditions} before '
                            f'RUNNING-only node <{running_label}>. Once <{running_label}> is running, the earlier '
                            'conditions may stop being rechecked. Use <ReactiveFallback> when those '
                            'conditions must be reevaluated while the fallback action runs.'
                        ), (
                            "Change <Fallback> to <ReactiveFallback> so conditions are "
                            "re-evaluated while the fallback action runs."
                        )

            return True, 'OK', ''
        except Exception as e:
            return False, str(e), ''
