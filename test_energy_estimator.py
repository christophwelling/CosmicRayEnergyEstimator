import cr_energy_estimator
from NuRadioReco.utilities import units

energy_estimator = cr_energy_estimator.CosmicRayEnergyEstimator('.', 'iron')
energy_estimator.draw_plots(75. * units.deg)
