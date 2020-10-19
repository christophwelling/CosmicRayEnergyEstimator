import cr_energy_estimator
from NuRadioReco.utilities import units

energy_estimator = cr_energy_estimator.CosmicRayEnergyEstimator('.')
energy_estimator.draw_plots(5. * units.deg)
