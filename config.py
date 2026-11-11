import numpy as np

# Physical constants
GRAVITATIONAL_CONSTANT = 6.67430e-11
SOFTENING = 1e7

# Simulation configuration
NUMBER_OF_BODIES = 300
TIME_STEP = 1000.0
TOTAL_STEPS = 1000

# Adaptive integration
MIN_TIME_STEP = 10.0
MAX_TIME_STEP = 5000.0

# Collision configuration
COLLISION_ENABLED = True

# Performance
CHUNK_SIZE = 128
RANDOM_SEED = 42

# Barnes-Hut configuration

BARNES_HUT_THETA = 0.5

OCTREE_CAPACITY = 1

OCTREE_MAX_DEPTH = 40

# Gravitational solver

GRAVITY_METHOD = "barnes_hut"

np.random.seed(RANDOM_SEED)