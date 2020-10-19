import MCEq.core
import crflux.models
import matplotlib.pyplot as plt
import argparse
import numpy as np
from NuRadioReco.utilities import units
import pickle

parser = argparse.ArgumentParser('Calculate a priori probability distributions for muon and cosmic ray energies')
parser.add_argument(
    '--delta_zenith',
    type=float,
    default=10.,
    help='Width of each zenith angle bin (in degrees)'
)
parser.add_argument(
    '--plot_folder',
    type=str,
    default='plots',
    help='Folder to save the plots in'
)
parser.add_argument(
    '--altitude',
    type=float,
    default=3200,
    help='Observer height above sea level'
)
args = parser.parse_args()
linestyles = ['-', '--', ':']

mceq = MCEq.core.MCEqRun(
    interaction_model='SIBYLL23C',
    primary_model=(crflux.models.HillasGaisser2012, 'H3a'),
    theta_deg=85.
)
zenith_angles = (np.arange(0, 90, args.delta_zenith) + .5 * args.delta_zenith) * units.deg

h_grid = np.linspace(50 * 1e3 * 1e2, 0, 500)    # altitudes from 0 to 50 km (in cm)
X_grid = mceq.density_model.h2X(h_grid)
alt_idx = np.abs(h_grid-args.altitude).argmin()

output_data = {
    'zenith_angles': zenith_angles,
    'energies': np.log10(mceq.e_grid * units.GeV)
}
fig1 = plt.figure(figsize=(12, 8))
ax1_1 = fig1.add_subplot(121)
ax1_1.grid()
ax1_1.set_title('Muon Spectrum')
ax1_1.set_xscale('log')
ax1_1.set_yscale('log')
# ax1_1.set_ylim([1.e-35, .5])
ax1_1.set_xlabel(r'$E_\mu [eV]$')
ax1_1.set_ylabel(r'$\Phi (E_\mu)$')
ax1_2 = fig1.add_subplot(122, sharey=ax1_1, sharex=ax1_1)
ax1_2.set_title('Cosmic Ray Spectrum')
ax1_2.set_xscale('log')
ax1_2.set_yscale('log')
ax1_2.set_xlabel(r'$E_{CR} [eV]$')
ax1_2.set_ylabel(r'$\Phi (E_{CR})$')
ax1_2.grid()
cr_spectrum = crflux.models.HillasGaisser2012().total_flux(mceq.e_grid)
output_data['cr_spectrum'] = cr_spectrum
muon_fluxes = np.zeros((len(zenith_angles), len(mceq.e_grid)))
ax1_2.plot(mceq.e_grid * units.GeV, cr_spectrum)
for i_zenith, zenith in enumerate(zenith_angles):
    mceq.set_theta_deg(zenith / units.deg)
    mceq.solve(int_grid=X_grid)
    muon_flux = (mceq.get_solution('mu+', grid_idx=alt_idx) +
                 mceq.get_solution('mu-', grid_idx=alt_idx))
    label = r'$\theta=%.1f ^\circ$' % (zenith / units.deg)
    ax1_1.plot(mceq.e_grid * units.GeV, muon_flux, label=label, color='C{}'.format(i_zenith))
    muon_fluxes[i_zenith] = muon_flux
output_data['muon_flux'] = muon_fluxes
ax1_1.legend()
fig1.tight_layout()
fig1.savefig('{}/a_priori_prabilities.png'.format(args.plot_folder))
pickle.dump(output_data, open('a_priori_probabilities.p', 'wb'))
