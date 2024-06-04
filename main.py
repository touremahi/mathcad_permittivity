"""
Calculates the sum of two numbers.

Args:
    a (int): The first number to add.
    b (int): The second number to add.

Returns:
    int: The sum of the two input numbers.
"""

from matplotlib import pyplot as plt

from fonctions import (
    charger_fichier_s2p, deembed, propagation,
    degre_lin, linear_fit, eigenvalues,
    gamma_to_angle, get_tan_delta,
    plot_s_db_deg, plot_er_tan_delta)

if __name__ ==  "__main__":
    CHEMIN = 'my_test_data/datas' # Contenant les fichiers S2P
    FICHIERS = {
        "air_l1"   : 'Ligne_25mm_vide.s2p',
        "air_l2"   : 'Ligne_35mm_vide.s2p',
        "ref_l2"   : 'Ligne_35mm_4Rogers.s2p',
        "cible_l2" : 'Ligne_35mm_2FR4.s2p'
    }
    ER_REF = 3.4

    # NE PAS MODIFIFER SI VOUS N'ETES PAS SUR
    # Chargement des fichiers
    air_l1 = charger_fichier_s2p(
        fichier = f'{CHEMIN}/{FICHIERS["air_l1"]}',
        nom_fichier="Ligne L1 air")
    air_l2 = charger_fichier_s2p(
        fichier = f'{CHEMIN}/{FICHIERS["air_l2"]}',
        nom_fichier="Ligne L2 air")

    ref_l2 = charger_fichier_s2p(
        fichier = f'{CHEMIN}/{FICHIERS["ref_l2"]}',
        nom_fichier="Ligne L2 Réf")
    cible_l2 = charger_fichier_s2p(
        fichier = f'{CHEMIN}/{FICHIERS["cible_l2"]}',
        nom_fichier="ligne 35mm milieu")

    # Deembed
    m_air, m_ref, m_cible = deembed(
        air_l1 = air_l1,
        air_l2 = air_l2,
        ref_l2 = ref_l2,
        cible_l2 = cible_l2
        )

    # Valeurs propres lambda
    lambda_air_1, lambda_air_2 = eigenvalues(m_air)
    lambda_ref_1, lambda_ref_2 = eigenvalues(m_ref)
    lambda_cible_1, lambda_cible_2 = eigenvalues(m_cible)

    # Constante de propagation gamma
    gamma_2_air = propagation(lambda_air_1, lambda_air_2)[1]
    gamma_2_ref = propagation(lambda_ref_1, lambda_ref_2)[1]
    gamma_2_cible = propagation(lambda_cible_1, lambda_cible_2)[1]

    # Calcul de la phase
    theta_air =  gamma_to_angle(gamma_2_air)
    theta_ref =  gamma_to_angle(gamma_2_ref)
    theta_cible =  gamma_to_angle(gamma_2_cible)

    # Linéarisation de phase
    theta_air = degre_lin(theta_air)
    theta_ref = degre_lin(theta_ref)
    theta_cible = degre_lin(theta_cible)

    epsilon_ref = (theta_ref/theta_air)**2
    epsilon_rem = (theta_cible/theta_air)**2

    # Lissage des valeurs
    epsilon_ref = linear_fit(epsilon_ref)
    epsilon_rem = linear_fit(epsilon_rem)

    # cacul de espsilon
    epsilon_r = ((epsilon_rem-1)/(epsilon_ref-1))*ER_REF

    # Calcul de tan_delta
    tan_delta = get_tan_delta(
        gamma_air=gamma_2_air,
        gamma_cible=gamma_2_cible,
        er=ER_REF
        )

    # Lissage de tan_delta
    tan_delta = linear_fit(tan_delta)

    ## FIGURES
    # Figure 1 - S-Param mesure Air
    # axs = ((ax01, ax02), (ax03, ax04))
    fig_air, axes_air = plt.subplots(2, 2)
    fig_air.suptitle('Mesure avec Air pour deux lignes')
    plot_s_db_deg([air_l1, air_l2], axes_air)

    # Figure 2 - S-Param milieu cible
    fig, axes_cible = plt.subplots(2, 2)
    fig.suptitle('Avec milieu')
    plot_s_db_deg([cible_l2], axes_cible)

    # Figure 3 - plots epsilon_r et Tan_delta
    freq = air_l2.frequency.f
    # axes = (ax01, ax02)
    fig, axes = plt.subplots(2,1)
    # Tracer epsilon_r et tangente delta 
    # en fonction de la fréquence
    plot_er_tan_delta(
        list_data=[epsilon_r, tan_delta],
        freq = freq,
        axes=axes)

    plt.show()
