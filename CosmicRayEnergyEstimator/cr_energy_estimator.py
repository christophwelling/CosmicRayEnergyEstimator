import numpy as np
import matplotlib.pyplot as plt
import pickle
from NuRadioReco.utilities import units


class CosmicRayEnergyEstimator():

    def __init__(
        self,
        file_location,
        cr_primary
    ):
        self.__a_priori_fluxes = pickle.load(open('{}/a_priori_probabilities.p'.format(file_location), 'rb'))
        self.__mu_energy_fluxes = pickle.load(open('{}/muon_energy_probabilities_{}.p'.format(file_location, cr_primary), 'rb'))
        self.__cr_primary = cr_primary
    def get_muon_spectrum(self, log_energy, zenith):
        i_zenith = np.argmin(np.abs(zenith - self.__a_priori_fluxes['zenith_angles']))
        mu_spectra = self.__a_priori_fluxes['muon_flux'][i_zenith]
        mu_energies = self.__a_priori_fluxes['energies']
        return mu_spectra[np.argmin(np.abs(log_energy - mu_energies))]

    def get_cosmic_ray_spectrum(self, log_energy):
        cr_energies = self.__a_priori_fluxes['energies']
        fluxes = self.__a_priori_fluxes['cr_spectrum']
        return fluxes[np.argmin(np.abs(log_energy - cr_energies))]

    def get_shower_muon_flux(self, log_cr_energy, log_mu_energy, zenith):
        cr_energies = self.__mu_energy_fluxes['cr_energies']
        mu_energies = self.__mu_energy_fluxes['mu_energies']
        i_zenith = np.argmin(np.abs(zenith - self.__mu_energy_fluxes['zeniths']))
        i_energy = np.argmin(np.abs(log_cr_energy - cr_energies))
        muon_fluxes = self.__mu_energy_fluxes['muon_flux'][i_energy, i_zenith]
        return muon_fluxes[np.argmin(np.abs(log_mu_energy - mu_energies))]

    def get_conditional_cr_probability(self, log_mu_energy, log_cr_energies, zenith):
        probabilities = np.zeros_like(log_cr_energies)
        for i_energy, log_cr_energy in enumerate(log_cr_energies):
            probabilities[i_energy] = self.get_cosmic_ray_spectrum(log_cr_energy) * self.get_shower_muon_flux(log_cr_energy, log_mu_energy, zenith)
        return probabilities / np.sum(probabilities)

    def draw_plots(self, zenith):
        cr_energy_bins = np.arange(16, 20., .5)
        mu_energies = np.arange(15, 19, .1)
        cr_energy_probabilities = np.zeros((len(cr_energy_bins), len(mu_energies)))
        cr_spectrum = np.zeros_like(cr_energy_bins)
        shower_mu_flux = np.zeros((len(cr_energy_bins), len(mu_energies)))
        for i_bin, mu_energy in enumerate(mu_energies):
            cr_energy_probabilities[:, i_bin] = self.get_conditional_cr_probability(mu_energy, cr_energy_bins, zenith)
            for i_cr_energy, cr_energy in enumerate(cr_energy_bins):
                shower_mu_flux[i_cr_energy, i_bin] = self.get_shower_muon_flux(cr_energy, mu_energy, zenith)
        for i_cr_energy, cr_energy in enumerate(cr_energy_bins):
            cr_spectrum[i_cr_energy] = self.get_cosmic_ray_spectrum(cr_energy)
        fig1 = plt.figure(figsize=(8, 8))
        ax1_1 = fig1.add_subplot(2, 2, (1, 2))
        ax1_1.grid()
        ax1_1.set_yscale('log')
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
            ax1_2.plot(mu_energies, shower_mu_flux[i_cr_energy])
        ax1_3.plot(cr_energy_bins, cr_spectrum, '-x')
        ax1_1.legend()
        fig1.tight_layout()
        fig1.savefig('cr_energy_estimate_{}.png'.format(self.__cr_primary))