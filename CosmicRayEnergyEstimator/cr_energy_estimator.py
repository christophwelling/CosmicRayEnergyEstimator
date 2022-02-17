import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.interpolate
import pickle
import crflux.models
from NuRadioReco.utilities import units


class CosmicRayEnergyEstimator:

    def __init__(
        self,
        file_location
    ):
        self.__a_priori_fluxes = pickle.load(open('{}/a_priori_probabilities.p'.format(file_location), 'rb'))
        self.__mu_spectrum = {
            'proton': pickle.load(open('{}/muon_energy_probabilities_proton.p'.format(file_location), 'rb')),
            'helium': pickle.load(open('{}/muon_energy_probabilities_helium.p'.format(file_location), 'rb')),
            'carbon': pickle.load(open('{}/muon_energy_probabilities_carbon.p'.format(file_location), 'rb')),
            'oxygen': pickle.load(open('{}/muon_energy_probabilities_oxygen.p'.format(file_location), 'rb')),
            'iron': pickle.load(open('{}/muon_energy_probabilities_iron.p'.format(file_location), 'rb'))
        }
        self.__gsf_table = np.loadtxt('{}/gsf_data_table.txt'.format(file_location), comments='#', delimiter=' ')
        self.__cosmic_ray_energy_table = np.log10(self.__gsf_table[:, 0] * units.GeV)
        self.__cosmic_rayflux = np.zeros((4, self.__cosmic_ray_energy_table.shape[0]))
        self.__cosmic_rayflux[0] = self.__gsf_table[:, 1]
        self.__cosmic_rayflux[1] = self.__gsf_table[:, 2]
        self.__cosmic_rayflux[2] = np.sum(self.__gsf_table[:, 3: 10], axis=1)
        self.__cosmic_rayflux[3] = np.sum(self.__gsf_table[:, 10:], axis=1)
        self.__flux_ids = {
            'proton': 0,
            'helium': 1,
            'carbon': 2,
            'oxygen': 2,
            'iron': 3
        }
        self.__corsika_ids = {
            'proton': 14,
            'helium': 402,
            'carbon': 1206,
            'iron': 5626
        }
        self.__linestyles = ['-', '--', ':', '-.']

    def get_muon_spectrum(self, log_energy, zenith):
        i_zenith = np.argmin(np.abs(zenith - self.__a_priori_fluxes['zenith_angles']))
        mu_spectra = self.__a_priori_fluxes['muon_flux'][i_zenith]
        mu_energies = self.__a_priori_fluxes['energies']
        return mu_spectra[np.argmin(np.abs(log_energy - mu_energies))]

    def get_cosmic_ray_spectrum(self, log_energy, element=None):
        index = np.zeros(log_energy.shape, dtype=int)
        for i_energy, energy in enumerate(log_energy):
            index[i_energy] = np.argmin(np.abs(energy - self.__cosmic_ray_energy_table))
        if element is None:
            return np.sum(self.__cosmic_rayflux, axis=0)[index]
        else:
            return self.__cosmic_rayflux[self.__flux_ids[element], index]

    def get_shower_muon_flux(self, log_cr_energy, log_mu_energy, zenith, primary='proton'):
        cr_energies = self.__mu_spectrum[primary]['cr_energies']
        mu_energies = self.__mu_spectrum[primary]['mu_energies']
        muon_flux = self.__mu_spectrum[primary]['muon_flux']
        i_zenith = np.argmin(np.abs(zenith - self.__mu_spectrum[primary]['zeniths']))
        flux_interpolation = scipy.interpolate.interp2d(
            mu_energies,
            cr_energies,
            np.log10(np.abs(muon_flux[:, i_zenith])),
            kind='quintic'
        )
        return 10.**flux_interpolation(log_mu_energy, log_cr_energy)

    def get_conditional_cr_probability(self, log_mu_energy, log_cr_energies, zenith):
        """
        Returns the probabilities that a muon with a specific energy and zenith angle was created by an air shower with
        the given cosmic ray energies

        Parameters:
        ------------------
        log_mu_energy: float
            log10 of the energy of the muon (in eV)
        log_cr_energies: array of floats
            log10 of the cosmic ray energy bins (in eV) for which the probabilities shall be calculated. The bins need to be
            evenly spaced in log space.
        zenith: float
            Zenith angle (in radions) of the muon

        Returns:
        ----------------
        array of floats: Probabilities that the muon stems from an air shower with the cosmic ray energies specified
            in the log_cr_energies parameter. The probabilites are normalized so that their sum is 1, which means that
            cosmic ray energies not in log_cr_energies are not considered.
        """
        log_bin_size = log_cr_energies[1] - log_cr_energies[0]
        probabilities = np.zeros_like(log_cr_energies)
        for i_energy, log_cr_energy in enumerate(log_cr_energies):
            mini_bins = np.arange(log_cr_energy - .45 * log_bin_size, log_cr_energy + .5 * log_bin_size, .1 * log_bin_size)
            mini_bin_sizes = np.power(10., mini_bins + .05 * log_bin_size) - np.power(10., mini_bins - .05 * log_bin_size)
            for primary in self.__corsika_ids.keys():
                n_particles = np.sum(mini_bin_sizes * self.get_cosmic_ray_spectrum(mini_bins, primary))
                probabilities[i_energy] += n_particles * self.get_shower_muon_flux(log_cr_energy, log_mu_energy, zenith, primary)
        return probabilities / np.sum(probabilities)

    def draw_plots(self, zenith):
        cr_energy_bins = np.arange(15, 20., .5)
        mu_energies = np.arange(14.5, 19.05, .1)
        cr_energy_probabilities = np.zeros((len(cr_energy_bins), len(mu_energies)))
        shower_mu_flux = np.zeros((len(cr_energy_bins), len(self.__corsika_ids.keys()), len(mu_energies)))
        for i_bin, mu_energy in enumerate(mu_energies):
            cr_energy_probabilities[:, i_bin] = self.get_conditional_cr_probability(mu_energy, cr_energy_bins, zenith)
            for i_cr_energy, cr_energy in enumerate(cr_energy_bins):
                for i_primary, primary in enumerate(self.__corsika_ids.keys()):
                    shower_mu_flux[i_cr_energy, i_primary, i_bin] = self.get_shower_muon_flux(cr_energy, mu_energy, zenith, primary)
        # for i_cr_energy, cr_energy in enumerate(cr_energy_bins):
        fig1 = plt.figure(figsize=(12, 12))
        ax1_1 = fig1.add_subplot(2, 2, (1, 2))
        ax1_1.grid()
        # ax1_1.set_yscale('log')
        ax1_1.set_ylim([.01, 1.1])
        ax1_1.set_xlabel(r'$log_{10}(E_\mu/eV)$')
        ax1_1.set_ylabel(r'$p(E_{CR}|E_\mu)$')
        ax1_2 = fig1.add_subplot(223)
        ax1_2.grid()
        ax1_2.set_yscale('log')
        ax1_2.set_ylim([1.e-15, 1.e-5])
        ax1_2.set_xlabel(r'$log_{10}(E_\mu/eV)$')
        ax1_2.set_ylabel(r'$N(E_\mu)$')
        ax1_3 = fig1.add_subplot(224)
        ax1_3.grid()
        ax1_3.set_yscale('log')
        ax1_3.set_xlabel(r'$log_{10}(E_{CR}/eV)$')
        ax1_3.set_ylabel(r'$\Phi(E_{CR})$')
        for i_cr_energy, cr_energy in enumerate(cr_energy_bins):
            ax1_1.plot(mu_energies, cr_energy_probabilities[i_cr_energy], label=r'$log_{10}(E_{CR}/eV)=%.1f$' % (cr_energy))
            for i_primary in range(len(self.__corsika_ids.keys())):
                ax1_2.plot(mu_energies, shower_mu_flux[i_cr_energy, i_primary], color='C{}'.format(i_cr_energy), linestyle=self.__linestyles[i_primary])
        cosmic_ray_flux = self.get_cosmic_ray_spectrum(cr_energy_bins)
        ax1_3.plot(cr_energy_bins, cosmic_ray_flux, '-x', color='C0', label='total')
        for i_primary, primary in enumerate(self.__corsika_ids.keys()):
            ax1_3.plot(cr_energy_bins, self.get_cosmic_ray_spectrum(cr_energy_bins, primary), 'x', color='C{}'.format(i_primary + 1), label=primary, linestyle=self.__linestyles[i_primary])
        ax1_3.set_ylim([.5 * np.min(cosmic_ray_flux), 1.5 * np.max(cosmic_ray_flux)])
        ax1_1.legend(ncol=3)
        ax1_3.legend()
        legend_handles = []
        for i_primary, primary in enumerate(self.__corsika_ids.keys()):
            legend_handles.append(mlines.Line2D([], [], color='k', linestyle=self.__linestyles[i_primary], label=primary))
        ax1_2.legend(handles=legend_handles)
        fig1.tight_layout()
        fig1.savefig('cr_energy_estimate.png')
