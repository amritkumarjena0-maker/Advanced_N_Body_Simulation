import numpy as np


class CollisionEngine:

    def detect_collisions(self, bodies):

        collisions = []

        for i in range(len(bodies)):

            for j in range(i + 1, len(bodies)):

                displacement = (
                    bodies[i].position - bodies[j].position
                )

                distance = np.linalg.norm(displacement)

                minimum_distance = (
                    bodies[i].radius + bodies[j].radius
                )

                if distance <= minimum_distance:

                    collisions.append((i, j))

        return collisions

    def merge_bodies(self, bodies):

        collisions = self.detect_collisions(bodies)

        removed = set()

        for i, j in collisions:

            if i in removed or j in removed:
                continue

            first = bodies[i]
            second = bodies[j]

            total_mass = first.mass + second.mass

            new_position = (
                first.mass * first.position
                + second.mass * second.position
            ) / total_mass

            new_velocity = (
                first.mass * first.velocity
                + second.mass * second.velocity
            ) / total_mass

            new_radius = (
                first.radius ** 3 + second.radius ** 3
            ) ** (1 / 3)

            from core.body import Body

            bodies[i] = Body(
                mass=total_mass,
                position=new_position,
                velocity=new_velocity,
                radius=new_radius,
                name=f"{first.name}+{second.name}"
            )

            removed.add(j)

        return [
            body for index, body in enumerate(bodies)
            if index not in removed
        ]