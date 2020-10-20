import MCEq.core
import crflux.models
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
from NuRadioReco.utilities import units

parser = argparse.ArgumentParser('Calculate probability distributions for muon energies')
parser.add_argument(
    '--cr_primary',
    type=str,
    default='proton',
    help='Assumed particle type of the primary cosmic ray. Options are proton, helium, carbon, oxygen and iron'
)
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
parser.add_argument(
    '--min_energy',
    type=float,
    default=15,
    help='Log10 of the smallest energy bin for which muon flux is calculated, in eV'
)
parser.add_argument(
    '--max_energy',
    type=float,
    default=20,
    help='Log 10 of the largest energy bin for which muon flux is calculated, in eV'
)
parser.add_argument(
    '--energy_bin_size',
    type=float,
    default=.5,
    help='Spacing of the energy bins in log10(E/eV) '
)
parser.add_argument(
    '--interaction_model',
    type=str,
    default='SIBYLL23C',
    help='Name of the interaction model to use for calculating the muon fluxes'
)
args = parser.parse_args()
linestyles = ['-', '--', ':']

mceq = MCEq.core.MCEqRun(
    interaction_model=args.interaction_model,
    primary_model=(crflux.models.HillasGaisser2012, 'H3a'),
    theta_deg=85.
)
energies = np.power(10., np.arange(args.min_energy, args.max_energy, args.energy_bin_size))
zenith_angles = (np.arange(0, 90, args.delta_zenith) + .5 * args.delta_zenith) * units.deg
corsika_ids = {
    'proton': 14,
    'helium': 402,
    'carbon': 1206,
    'oxygen': 1608,
    'iron': 5626
}

h_grid = np.linspace(50 * 1e3 * 1e2, 0, 500)    # altitudes from 0 to 50 km (in cm)
X_grid = mceq.density_model.h2X(h_grid)
alt_idx = np.abs(h_grid-args.altitude).argmin()

output_data = {
    'mu_energies': np.log10(mceq.e_grid * units.GeV),
    'zeniths': zenith_angles,
    'cr_energies': np.log10(energies)
}
fig1 = plt.figure(figsize=(20, 15))
muon_fluxes = np.zeros((len(energies), len(zenith_angles), len(mceq.e_grid)))
axes_array = []
for i_energy, energy in enumerate(energies):
    if i_energy == 0:
        if len(energies) <= 4:
            ax1_1 = fig1.add_subplot(1, len(energies), i_energy + 1)
        else:
            ax1_1 = fig1.add_subplot(2, len(energies) // 2 + len(energies) % 2, i_energy + 1)
    else:
        if len(energies) <= 4:
            ax1_1 = fig1.add_subplot(1, len(energies), i_energy + 1, sharey=axes_array[0], sharex=axes_array[0])
        else:
            ax1_1 = fig1.add_subplot(2, len(energies) // 2 + len(energies) % 2, i_energy + 1, sharey=axes_array[0], sharex=axes_array[0])
    print(i_energy, energy)
    axes_array.append(ax1_1)
    ax1_1.grid()
    ax1_1.set_xscale('log')
    ax1_1.set_yscale('log')
    ax1_1.set_xlabel(r'$E_\mu [eV]$')
    ax1_1.set_ylabel(r'$\Phi (E_\mu | E_{CR})$')
    ax1_1.set_title(r'$log_{10}(E_{CR}/eV)=%.1f$' % (np.log10(energy)))
    mceq.set_single_primary_particle(energy / 1.e9, corsika_id=corsika_ids[args.cr_primary])
    for i_zenith, zenith in enumerate(zenith_angles):
        mceq.set_theta_deg(zenith / units.deg)
        mceq.solve(int_grid=X_grid)
        muon_flux = (mceq.get_solution('mu+', grid_idx=alt_idx) +
                     mceq.get_solution('mu-', grid_idx=alt_idx))
        label = r'$\theta=%.1f ^\circ$' % (zenith / units.deg)
        ax1_1.plot(mceq.e_grid * units.GeV, muon_flux, label=label, color='C{}'.format(i_zenith))
        muon_fluxes[i_energy, i_zenith] = muon_flux
        if i_energy == 0:
            ax1_1.legend()
output_data['muon_flux'] = muon_fluxes
fig1.tight_layout()
plt.savefig('{}/muon_energy_distribution_{}.png'.format(args.plot_folder, args.cr_primary))
pickle.dump(output_data, open('muon_energy_probabilities_{}.p'.format(args.cr_primary), 'wb'))
