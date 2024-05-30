"""
Fonctions utilitaires
"""
import shutil

import numpy as np
from scipy.signal import medfilt
import skrf as rf
from skrf.calibration.deembedding import ImpedanceCancel

def charger_fichier_s2p(fichier, nom_fichier=None):
    """
    Loads an S2P file and returns a Network object.

    Args:
        fichier (str): The path to the S2P file to load.
        nom_fichier (str): The name to assign to the Network object.

    Returns:
        Network: A Network object representing the loaded S2P file, or None if an error occurred.
    """
    try:
        if fichier.endswith('.s2p'):
            data = rf.Network(fichier, name=nom_fichier)
        else:
            shutil.copyfile(fichier, fichier+'.s2p')
            data = rf.Network(fichier+'.s2p', name=nom_fichier)
    except ImportError:
        print('Erreur lors de la lecture du fichier')
        return None
    #data.resample(101)
    return data

def degre_lin(phase, period=360):
    """
    Unwraps the given phase values to produce continuous phase values.
    
    Args:
        phase (numpy.ndarray): The phase values to unwrap.
        period (float, optional): The period of the phase values, defaults to 360.
    
    Returns:
        numpy.ndarray: The unwrapped phase values.
    """
    return np.unwrap(phase, period)

def linear_fit(data, size=51):
    """
    Fits a linear function to a signal. equivalent of medsmooth.
    """
    return medfilt(data, kernel_size = size)

def deembed(air_l1:rf.Network, air_l2:rf.Network, ref_l2=None, cible_l2=None):
    """
    Performs de-embedding of a network using the ImpedanceCancel class.
    
    Args:
        air_l1 (rf.Network): The network representing the air line 1.
        air_l2 (rf.Network): The network representing the air line 2.
        ref_l2 (rf.Network, optional): The network representing the reference material for line 2.
        cible_l2 (rf.Network): The network representing the target material for line 2.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            The ABCD matrices for the air line 2,
            reference material
            and target material, respectively deembedded from the air line 1.
    
    Raises:
        ValueError: If `cible_l2` is not provided.
    """

    if cible_l2 is None:
        raise ValueError('Cible_l2 est obligatoire')
    ## De-Embedding par rapport à air_1
    dm = ImpedanceCancel(dummy_thru = air_l1, name = 'air_l1')
    # Matrice ABCD Deembedding Ligne 2 à air
    m_air = dm.deembed(air_l2).a
    # Matrice ABCD Deembedding Ligne 2 à matériau de référence
    if ref_l2 is not None:
        m_ref = dm.deembed(ref_l2).a
    # Matrice ABCD Deembedding Ligne 2 à matériau cible
    m_cible = dm.deembed(cible_l2).a
    return m_air, m_ref, m_cible

def eigenvalues(m):
    """
    Compute the eigenvalues of a 2x2 matrix.
    
    Args:
        m (numpy.ndarray): A 2x2 matrix.
    
    Returns:
        tuple: The two eigenvalues of the input matrix.
    """
    s = np.linalg.eigvals(m)
    lambda_1 = s[:,0]
    lambda_2 = s[:,1]
    return lambda_1, lambda_2

def propagation(a, b):
    """
    Calculates the logarithm of `a`, the negative logarithm of `b`,
    and the inverse hyperbolic cosine of the average of `a` and `b`.

    Args:
        a (float): The first input value.
        b (float): The second input value.

    Returns:
        list: A list containing the three calculated values.
    """

    return [np.log(a), -np.log(b), np.arccosh((a+b)/2)]

def gamma_to_angle(gamma):
    """
    Converts the propagation constant gamma to the quasi-static parameters,
    qs_1 and qs_2."""
    return gamma.imag*(180/np.pi)

def get_tan_delta(gamma_air, gamma_cible, er=3.4):
    """
    Calculates the tangent delta (tan δ) of a target material based on the complex propagation constants of the air and the target material.
    
    Args:
        gamma_air (complex): The complex propagation constant of air.
        gamma_cible (complex): The complex propagation constant of the target material.
        er (float, optional): The relative permittivity of the target material. Defaults to 3.4.
    
    Returns:
        float: The tangent delta of the target material.
    """
    beta_a = gamma_air.imag
    tan_delta = 2*np.abs(
        (gamma_cible.real/(beta_a*np.sqrt(er))) - \
        (gamma_air.real/beta_a)
        )
    return tan_delta

def mydeembed(air_1, air_2, milieu_1, milieu_2, er=3.4):
    """
    De-embeds the material properties (relative permittivity and loss tangent)
    from a set of S-parameter measurements.

    Args:
        air_1 (object): S-parameter measurements of the air line.
        air_2 (object): S-parameter measurements of the air line.
        milieu_1 (object): S-parameter measurements of the reference material.
        milieu_2 (object): S-parameter measurements of the target material.
        er (float, optional): Relative permittivity of the target material. Defaults to 3.4.

    Returns:
        tuple: The relative permittivity and loss tangent of the target material.
    """

    # me_a_k = air_1.a
    # mg_a2 = air_2.a
    # mg_m1 = milieu_1.a
    # mg_m2 = milieu_2.a

    # me_a_k = lissage(air_1.a, 51)
    # mg_a2 = lissage(air_2.a, 51)
    # mg_m1 = lissage(milieu_1.a, 51)
    # mg_m2 = lissage(milieu_2.a, 51)

    ## De-Embedding par rapport à air_1 
    dm = ImpedanceCancel(dummy_thru = air_1, name = 'air_1')
    # Matrice ABCD Deembedding Ligne 2 à air
    m_air = dm.deembed(air_2).a
    # Matrice ABCD Deembedding Ligne 2 à matériau de référence
    m_ref = dm.deembed(milieu_1).a
    # Matrice ABCD Deembedding Ligne 2 à matériau cible
    m_mat = dm.deembed(milieu_2).a

    ## Cacul des paramètres de propagation
    # Calcul valeurs propres
    # pour l'air
    s_air = np.linalg.eigvals(m_air)
    lambda_air_1 = s_air[:,0]
    lambda_air_2 = s_air[:,1]

    # pour le matériau de référence
    s_ref = np.linalg.eigvals(m_ref)
    lambda_ref_1 = s_ref[:,0]
    lambda_ref_2 = s_ref[:,1]

    # pour le matériau cible
    s_mat = np.linalg.eigvals(m_mat)
    lambda_mat_1 = s_mat[:,0]
    lambda_mat_2 = s_mat[:,1]

    # Détermination de la constante de propagation de la ligne
    prop_air = propagation(lambda_air_1, lambda_air_2)
    # gamma_1_air = prop_air[0]
    gamma_2_air = prop_air[1]

    prop_ref = propagation(lambda_ref_1, lambda_ref_2)
    # gamma_1_ref = prop_ref[0]
    gamma_2_ref = prop_ref[1]

    prop_mat = propagation(lambda_mat_1, lambda_mat_2)
    # gamma_1_mat = prop_mat[0]
    gamma_2_mat = prop_mat[1]

    # Facteur Q
    qs_air = gamma_2_air.imag*(180/np.pi)
    qs_ref = gamma_2_ref.imag*(180/np.pi)
    qs_mat = gamma_2_mat.imag*(180/np.pi)

    # Linéarisation de la phase et correction de la constante de phase
    theta_air = np.unwrap(qs_air, 360)
    theta_ref = np.unwrap(qs_ref, 360)
    theta_mat = np.unwrap(qs_mat, 360)

    epsilon_ref = (theta_ref/theta_air)**2
    epsilon_rem = (theta_mat/theta_air)**2

    eps_ref = epsilon_ref # medfilt(epsilon_ref, kernel_size = 51)
    eps_rem = epsilon_rem # medfilt(epsilon_rem, kernel_size = 51)

    eps_mat = ((eps_rem-1)/(eps_ref-1))*er

    # tan_delta
    beta_a = gamma_2_air.imag
    tan_delta_0 = 2*np.abs(
        (gamma_2_mat.real/(beta_a*np.sqrt(er))) - \
        (gamma_2_air.real/beta_a)
        )
    tan_delta = medfilt(tan_delta_0, kernel_size = 51)

    return eps_mat, tan_delta

def plot_s_db_deg(list_s, axes):
    """
    Plots the S-parameter in dB for a given measurement.
    """
    for s in list_s:
        # S11_dB
        s.plot_s_db(m=0,n=0,ax=axes[0][0])

        # S11_phase
        s.plot_s_deg(m=0,n=0,ax=axes[0][1])

        # S21_dB
        s.plot_s_db(m=1,n=0,ax=axes[1][0])

        # S21_phase
        s.plot_s_deg(m=1,n=0,ax=axes[1][1])

def plot_er_tan_delta(list_data:list, freq, axes):
    """
    Plots the loss tangent and relative permittivity of a given measurement.
    """
    axes[0].set_ylim(1, 6)
    axes[0].set_ylabel('Permittivity (ε_r)')
    # axes[0].set_xscale("log")
    axes[1].set_ylabel('Permittivity (ε_r)')
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    for i, data in enumerate(list_data):
        axes[i].plot(freq/1e9, data)
        axes[i].set_xlabel('Frequency [GHz]')

if __name__ == "__main__":
    pass
