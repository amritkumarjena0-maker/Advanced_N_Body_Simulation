class SemiImplicitEuler:

    def __init__(self, gravity_engine):
        self.gravity = gravity_engine

    def step(self, bodies, dt):

        accelerations = (
            self.gravity.calculate_accelerations(bodies)
        )

        for body, acceleration in zip(
            bodies, accelerations
        ):

            body.velocity += acceleration * dt

            body.position += body.velocity * dt

        return bodies