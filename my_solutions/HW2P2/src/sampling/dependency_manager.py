"""
Dependency management for search space execution ordering.

This module builds a path-based DAG from the `search_spaces` configuration and
returns a new, sorted representation of the tree:
- Top-level sort unit: STRATEGY nodes (e.g., architectures, training, ...)
- Within each strategy: dependency-aware ordering of TECHNIQUE → INSTANCE → PARAM

Only the global `search_spaces` container is supported as input. Use same-level
`depends_on` to express ordering between strategies and between techniques.
"""

import dataclasses
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict, deque
import pprint

from omegaconf import DictConfig, OmegaConf

from .enums import ConfigClass, parse_config_class


@dataclasses.dataclass
class _ConfigNode:
    """Internal representation of a node within the configuration tree."""

    path: str
    key: str
    config: DictConfig
    node_class: ConfigClass
    parent_path: Optional[str] = None
    children_paths: List[str] = dataclasses.field(default_factory=list)


class DependencyManager:
    """
    Resolve dependencies and produce a sorted representation of the tree where
    nodes at each level are ordered topologically.
    """

    def __init__(self, sampler=None, silent: bool = False):
        """Initializes the DependencyManager."""
        self.sampler = sampler
        self.silent = silent
        self._nodes: Dict[str, _ConfigNode] = {}
        self._name_to_paths: Dict[str, List[str]] = defaultdict(list)
        self._graph = defaultdict(list)

    def _log(self, message: str):
        """Prints a log message if the silent flag is not set."""
        if not self.silent:
            print(message)

    def _reset_state(self):
        """Resets the internal state of the manager for a new analysis."""
        self._nodes = {}
        self._name_to_paths = defaultdict(list)
        self._graph = defaultdict(list)

    def get_sorted_config_tree(self, cfg: DictConfig) -> List[DictConfig]:
        """
        Analyze dependencies and return a new configuration tree with sorted nodes.
        Expects the top-level `search_spaces` container (no 'class' at root).
        """
        self._reset_state()  # Reset state for a clean run
        self._log("▶️  [DM] Starting dependency analysis...")
        self._collect_nodes_and_dependencies(cfg)
        self._log(f"   [DM] Collected {len(self._nodes)} total nodes.")

        self._resolve_name_based_dependencies()
        self._log("   [DM] Resolved name-based dependencies to path-based dependencies.")

        # Enforce global container: root has no 'class'; top-level nodes must be STRATEGY.
        # Identify top-level STRATEGY nodes to sort.
        top_level_strategy_paths = [
            path
            for path, node in self._nodes.items()
            if node.node_class == ConfigClass.STRATEGY.value and not node.parent_path
        ]
        if not top_level_strategy_paths:
            raise ValueError(
                "DependencyManager expects the global search_spaces container as input (with STRATEGY children)."
            )

        self._log(
            f"   [DM] Found {len(top_level_strategy_paths)} top-level strategies to sort: {top_level_strategy_paths}"
        )

        sorted_strategy_paths = self._topological_sort(top_level_strategy_paths)
        self._log(f"   [DM] Top-level strategies sorted: {sorted_strategy_paths}")

        # Build sorted tree (strategies at top-level). Children of each strategy
        # will be sorted recursively by _build_sorted_tree using the same logic.
        sorted_tree = self._build_sorted_tree(sorted_strategy_paths)
        self._log("✅  [DM] Dependency analysis complete. Sorted tree created.")
        return sorted_tree

    def _collect_nodes_and_dependencies(self, node: DictConfig, path: str = "", parent_path: Optional[str] = None):
        if not parent_path and not hasattr(node, "class"):
            for key, child in node.items():
                if isinstance(child, DictConfig) and "class" in child:
                    self._collect_nodes_and_dependencies_recursive(child, key, key, path)
        else:
            self._collect_nodes_and_dependencies_recursive(node, path, path, parent_path)

    def _collect_nodes_and_dependencies_recursive(
        self, node: DictConfig, key: str, current_path: str, parent_path: Optional[str]
    ):
        if not isinstance(node, DictConfig) or "class" not in node:
            return
        node_class = parse_config_class(node["class"])
        self._nodes[current_path] = _ConfigNode(
            path=current_path, key=key, config=node, node_class=node_class, parent_path=parent_path
        )
        if parent_path and parent_path in self._nodes:
            self._nodes[parent_path].children_paths.append(current_path)

        # Record the node's own key and its param_name (if applicable) in the
        # one-to-many name-to-paths mapping. This allows us to find all occurrences
        # of a name, which is crucial for context-aware dependency resolution.
        self._name_to_paths[key].append(current_path)
        if node_class == ConfigClass.PARAM.value:
            param_name = node.get("param_name", key)
            if key != param_name:  # Avoid duplicate entries if key is the param_name
                self._name_to_paths[param_name].append(current_path)

        # Collect dependencies from 'depends_on' field, regardless of node class
        if hasattr(node, "depends_on"):
            dependencies = node.depends_on
            if isinstance(dependencies, str):
                dependencies = [dependencies]

            # The dependent is the current node itself (or its main param)
            dependent_name = node.get("param_name", key)
            for dep_name in dependencies:
                self._graph[dep_name].append(dependent_name)

        for child_key, child_node in node.items():
            if child_key in {
                "class",
                "description",
                "condition",
                "_ordered_children",
                "_original_key",
                "_path",
                "depends_on",
            }:
                continue
            if isinstance(child_node, DictConfig):
                self._collect_nodes_and_dependencies_recursive(
                    child_node, child_key, f"{current_path}.{child_key}", current_path
                )

    def _resolve_name_based_dependencies(self):
        path_graph = defaultdict(list)
        self._log("   [Resolver] Resolving all name-based dependencies to paths...")

        for dep_name, dependent_names in self._graph.items():
            for dependent_name in dependent_names:
                # Find all possible paths for the dependent node
                dependent_node_paths = self._name_to_paths.get(dependent_name)
                if not dependent_node_paths:
                    self._log(
                        f"      [Resolver] ❗ WARNING: Could not find any path for dependent node '{dependent_name}'. Skipping."
                    )
                    continue

                # A dependency declaration is tied to a specific location. We assume the *last*
                # collected path for a given name is the one currently being processed.
                # This holds true due to the depth-first traversal of the collector.
                dependent_node_path = dependent_node_paths[-1]

                # Find the dependency's path using context-aware search
                dep_path = self._find_dependency_path(dep_name, dependent_node_path)

                if dep_path:
                    if dep_path != dependent_node_path:
                        path_graph[dep_path].append(dependent_node_path)
                else:
                    self._log(
                        f"      [Resolver] ❗ WARNING: Could not find a valid path for dependency '{dep_name}' relative to '{dependent_node_path}'. Skipping."
                    )

        self._graph = path_graph

    def _find_dependency_path(self, dep_name: str, dependent_path: str) -> Optional[str]:
        """
        Context-aware search for a dependency's path.
        Searches from the dependent's local context outwards and intelligently
        chooses the best candidate if multiple paths exist for a dependency name.
        """
        current_node = self._nodes.get(dependent_path)
        if not current_node:
            # Fallback to the first globally registered path if the dependent node itself isn't found
            return self._name_to_paths.get(dep_name, [None])[0]

        # Traverse up from the current node's parent
        search_node = self._nodes.get(current_node.parent_path) if current_node.parent_path else None

        while search_node:
            # Check children of the current search_node (siblings of the previous node)
            for child_path in search_node.children_paths:
                child_node = self._nodes.get(child_path)
                if child_node:
                    # Check if the child's key or param_name matches the dependency name
                    if child_node.key == dep_name or (
                        child_node.node_class == ConfigClass.PARAM.value
                        and child_node.config.get("param_name", child_node.key) == dep_name
                    ):
                        return child_path

            # Move up to the next parent in the hierarchy
            search_node = self._nodes.get(search_node.parent_path) if search_node.parent_path else None

        # If not found in the immediate family tree, check all globally known paths
        # and find the one with the longest common path prefix, assuming it's the "closest relative".
        possible_paths = self._name_to_paths.get(dep_name, [])
        if not possible_paths:
            return None
        if len(possible_paths) == 1:
            return possible_paths[0]

        # Find the best match by comparing path prefixes
        best_match = None
        max_common_len = -1
        for path in possible_paths:
            common_prefix_len = len(self._get_common_prefix(path, dependent_path))
            if common_prefix_len > max_common_len:
                max_common_len = common_prefix_len
                best_match = path

        return best_match

    def _get_common_prefix(self, path1: str, path2: str) -> str:
        """Helper to find the longest common prefix between two dot-separated paths."""
        p1_parts = path1.split(".")
        p2_parts = path2.split(".")
        common = []
        for i in range(min(len(p1_parts), len(p2_parts))):
            if p1_parts[i] == p2_parts[i]:
                common.append(p1_parts[i])
            else:
                break
        return ".".join(common)

    def get_all_descendants(self, path: str) -> Set[str]:
        descendants = set()
        nodes_to_visit = [path]
        while nodes_to_visit:
            current_path = nodes_to_visit.pop()
            if current_path in descendants:
                continue
            descendants.add(current_path)
            node = self._nodes.get(current_path)
            if node:
                nodes_to_visit.extend(node.children_paths)
        return descendants

    def _topological_sort(self, node_paths: List[str]) -> List[str]:
        # Build a local subgraph that includes all descendants to resolve indirect dependencies
        subgraph = defaultdict(list)
        all_nodes_in_scope = set(node_paths)

        # The key insight is that the global graph already contains all param-level dependencies.
        # When sorting a list of nodes (e.g., children of a technique), we need to know
        # if any descendant of child A depends on any descendant of child B.

        in_degree = defaultdict(int)
        # Create a temporary graph for the current sorting context
        local_graph = defaultdict(list)

        for dep_path, dependent_paths in self._graph.items():
            dep_ancestor = self.find_ancestor_in_list(dep_path, node_paths)
            for dependent_path in dependent_paths:
                dependent_ancestor = self.find_ancestor_in_list(dependent_path, node_paths)

                if dep_ancestor and dependent_ancestor and dep_ancestor != dependent_ancestor:
                    if dependent_ancestor not in local_graph[dep_ancestor]:
                        local_graph[dep_ancestor].append(dependent_ancestor)
                        in_degree[dependent_ancestor] += 1

        queue = deque([path for path in node_paths if in_degree[path] == 0])
        sorted_order = []
        while queue:
            path = queue.popleft()
            sorted_order.append(path)
            for dependent in local_graph.get(path, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_order) != len(node_paths):
            remaining = set(node_paths) - set(sorted_order)
            raise ValueError(f"Circular dependency detected among nodes: {remaining}")
        return sorted_order

    def find_ancestor_in_list(self, path: str, ancestor_list: List[str]) -> Optional[str]:
        current = self._nodes.get(path)
        while current:
            if current.path in ancestor_list:
                return current.path
            current = self._nodes.get(current.parent_path) if current.parent_path else None
        return None

    def _build_sorted_tree(self, sorted_node_paths: List[str]) -> List[DictConfig]:
        sorted_tree = []
        for path in sorted_node_paths:
            node = self._nodes[path]
            new_config_node = node.config.copy()
            OmegaConf.set_struct(new_config_node, False)
            if node.children_paths:
                sorted_children_paths = self._topological_sort(node.children_paths)
                new_config_node["_ordered_children"] = self._build_sorted_tree(sorted_children_paths)
            new_config_node["_original_key"] = node.key
            new_config_node["_path"] = node.path
            sorted_tree.append(new_config_node)
        return sorted_tree
