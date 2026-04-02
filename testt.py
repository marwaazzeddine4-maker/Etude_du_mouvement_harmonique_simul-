import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Définition du modèle
def modele(t, A, omega, phi):
    return A * np.cos(omega * t + phi)

# 2. Paramètres réels
A_reel = 2.0        # amplitude
omega_reel = 2*np.pi  # pulsation
phi_reel = 0.5      # phase

# 3. Génération du temps
t = np.linspace(0, 10, 200)

# 4. Signal propre
signal_propre = modele(t, A_reel, omega_reel, phi_reel)


# 5. Ajout du bruit
bruit = np.random.normal(0, 0.5, size=t.shape)
signal_bruite = signal_propre + bruit

# 6. Ajustement (moindres carrés)
# estimation initiale
params_init = [1.5, 6.0, 0.0]

params_opt, covariance = curve_fit(modele, t, signal_bruite, p0=params_init)

A_est, omega_est, phi_est = params_opt

# 7. Signal ajusté
signal_fit = modele(t, A_est, omega_est, phi_est)


# 8. Affichage des résultats
print("Paramètres réels :")
print(f"A = {A_reel}, omega = {omega_reel}, phi = {phi_reel}")

print("\nParamètres estimés :")
print(f"A = {A_est:.3f}, omega = {omega_est:.3f}, phi = {phi_est:.3f}")

# 9. Visualisation
plt.figure(figsize=(10, 6))

plt.scatter(t, signal_bruite, label="Signal bruité", color='red', s=15)
plt.plot(t, signal_propre, label="Signal propre", linestyle='--')
plt.plot(t, signal_fit, label="Ajustement", linewidth=2)

plt.xlabel("Temps")
plt.ylabel("Position")
plt.title("Mouvement harmonique avec bruit")
plt.legend()
plt.grid()

plt.show()