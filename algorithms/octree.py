
import numpy as np


class OctreeNode:

    def __init__(self, center, half_size, depth=0):

        self.center = np.asarray(center, dtype=np.float64)
        self.half_size = float(half_size)
        self.depth = depth

        self.mass = 0.0
        self.center_of_mass = np.zeros(3)

        self.children = [None] * 8
        self.indices = []

    @property
    def is_leaf(self):
        return all(child is None for child in self.children)

    def get_octant(self, position):

        octant = 0

        if position[0] >= self.center[0]:
            octant |= 1

        if position[1] >= self.center[1]:
            octant |= 2

        if position[2] >= self.center[2]:
            octant |= 4

        return octant

    def create_child(self, octant):

        offset = self.half_size / 2.0

        direction = np.array([
            1 if octant & 1 else -1,
            1 if octant & 2 else -1,
            1 if octant & 4 else -1
        ])

        child_center = (
            self.center + direction * offset
        )

        self.children[octant] = OctreeNode(
            child_center,
            offset,
            self.depth + 1
        )

        return self.children[octant]


class Octree:

    def __init__(
        self,
        positions,
        masses,
        capacity=1,
        max_depth=40
    ):

        self.positions = np.asarray(
            positions, dtype=np.float64
        )

        self.masses = np.asarray(
            masses, dtype=np.float64
        )

        self.capacity = capacity
        self.max_depth = max_depth

        if len(self.positions) != len(self.masses):
            raise ValueError("Position and mass counts differ.")

        if len(self.positions) == 0:
            raise ValueError("Cannot build an empty Octree.")

        if not np.all(np.isfinite(self.positions)):
            raise ValueError("Positions must be finite.")

        if not np.all(np.isfinite(self.masses)):
            raise ValueError("Masses must be finite.")

        if np.any(self.masses <= 0):
            raise ValueError("Masses must be positive.")

        self.root = self._build_root()

        self._build_node(
            self.root,
            np.arange(len(self.positions))
        )

        self._calculate_mass(self.root)

    def _build_root(self):

        minimum = np.min(self.positions, axis=0)
        maximum = np.max(self.positions, axis=0)

        center = (minimum + maximum) / 2.0

        half_size = max(
            np.max(maximum - minimum) / 2.0,
            1e-6
        )

        # Padding keeps boundary particles inside the root.
        half_size = half_size * (1.0 + 1e-10) + 1e-9

        return OctreeNode(center, half_size)

    def _build_node(self, node, indices):

        if (
            len(indices) <= self.capacity
            or node.depth >= self.max_depth
            or node.half_size <= 1e-12
        ):

            node.indices = indices.tolist()
            return

        groups = [[] for _ in range(8)]

        for index in indices:

            octant = node.get_octant(
                self.positions[index]
            )

            groups[octant].append(index)

        # Avoid subdividing when every particle follows
        # the same path through the tree.
        if sum(bool(group) for group in groups) == 1:

            occupied = next(
                group for group in groups if group
            )

            if len(occupied) == len(indices):
                pass

        for octant, group in enumerate(groups):

            if not group:
                continue

            child = node.create_child(octant)

            self._build_node(
                child,
                np.asarray(group, dtype=np.int64)
            )

    def _calculate_mass(self, node):

        if node.is_leaf:

            if not node.indices:
                return

            indices = np.asarray(
                node.indices, dtype=np.int64
            )

            local_masses = self.masses[indices]

            node.mass = np.sum(local_masses)

            node.center_of_mass = np.sum(
                self.positions[indices]
                * local_masses[:, None],
                axis=0
            ) / node.mass

            return

        total_mass = 0.0
        weighted_position = np.zeros(3)

        for child in node.children:

            if child is None:
                continue

            self._calculate_mass(child)

            total_mass += child.mass

            weighted_position += (
                child.mass * child.center_of_mass
            )

        node.mass = total_mass

        if total_mass > 0:

            node.center_of_mass = (
                weighted_position / total_mass
            )
            
            if sum(bool(group) for group in groups) == 1:
                
                occupied = next(
                    group for group in groups if group
            )

            group_indices = np.asarray(
                occupied, dtype=np.int64
            )

            if np.all(
                self.positions[group_indices]
                == self.positions[group_indices[0]]
            ):

                node.indices = group_indices.tolist()
                return
            
            
    def count_nodes(self):

        def count(node):

            if node is None:
                return 0

            return 1 + sum(
                count(child)
                for child in node.children
            )

        return count(self.root)
