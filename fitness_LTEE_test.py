import numpy as np
import curveball.competitions

# Generate synthetic data
np.random.seed(42)  # For reproducibility
time_points = 10
replicates = 5

# Assume population densities for assay strain and reference strain
# Shape: (time, strains, replicates)
y = np.zeros((time_points, 2, replicates))
y[:, 0, :] = np.exp(np.linspace(0, 2, time_points))[:, None] * (1 + np.random.normal(0, 0.1, (time_points, replicates)))
y[:, 1, :] = np.exp(np.linspace(0, 1.5, time_points))[:, None] * (1 + np.random.normal(0, 0.1, (time_points, replicates)))

# Calculate relative fitness without confidence interval
fitness = curveball.competitions.fitness_LTEE(y, ref_strain=0, assay_strain=1, t0=0, t1=-1, ci=0)
print("Relative fitness (without CI):", fitness)

# Calculate relative fitness with confidence interval
fitness, low_ci, high_ci = curveball.competitions.fitness_LTEE(y, ref_strain=0, assay_strain=1, t0=0, t1=-1, ci=0.95)
print("Relative fitness (with 95% CI):", fitness)
print("95% CI:", (low_ci, high_ci))
