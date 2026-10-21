import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SAFFMAN-TURNER COLLISION/COALESCENCE MODEL
# Mass-variable formulation
#
# Current physics:
#     J = 0
#     condensation term = 0
#     only collision gain + collision loss
#
# Time integration:
#     midpoint RK2
#
# Size variable:
#     q = m / m0  (dimensionless relative droplet mass)
#
# Initial condition:
#     Gaussian radius distribution centered at r0 = 10 um
#     with width sigma_r = 1 um, as specified for the present test.
#
# IMPORTANT:
#     "width = 1 um" is interpreted here as 1-sigma.
#  
# ============================================================


# ------------------------------------------------------------
# 1. Physical parameters
# ------------------------------------------------------------

rho_l = 1000.0                  # kg/m^3, liquid-water density

r0 = 10.0e-6                    # m, initial/reference droplet radius
sigma_r = 1.0e-6               # m, width of initial radius distribution

# Saffman-Turner example:
N0_cm3 = 240.0                  # droplets/cm^3
N0 = N0_cm3 * 1.0e6             # droplets/m^3

# Saffman-Turner values
# E = 1000 cm^2/s^3 is the turbulent-cumulus value discussed
# in the original paper.
epsilon = 1000.0 * 1.0e-4       # m^2/s^3

# nu = 0.17 cm^2/s in the Saffman-Turner calculation
nu = 0.17 * 1.0e-4              # m^2/s


# ------------------------------------------------------------
# 2. Reference droplet mass
# ------------------------------------------------------------

m0 = (4.0 / 3.0) * np.pi * rho_l * r0**3

print("\nReference droplet:")
print(f"r0 = {r0:.4e} m = {r0*1e6:.2f} um")
print(f"m0 = {m0:.6e} kg")
print(f"N0 = {N0_cm3:.1f} droplets/cm^3")
print(f"epsilon = {epsilon:.4e} m^2/s^3")
print(f"nu = {nu:.4e} m^2/s")


# ------------------------------------------------------------
# 3. Numerical mass grid
#
# We construct q=m/m0 from a uniform radius grid.
# This gives good resolution around the initial 10-um peak
# while still allowing the mass range to grow.
# ------------------------------------------------------------

r_min = 5.0e-6                 # m
r_max = 60.0e-6                # m
N_grid = 300

r_grid = np.linspace(r_min, r_max, N_grid)

# q = m/m0 = (r/r0)^3
q = (r_grid / r0)**3

# Actual mass corresponding to q
m_grid = q * m0


# ------------------------------------------------------------
# 4. Initial distribution
#
# First construct p_r(r), normalized so integral p_r dr = 1.
# Then f_r = N0 * p_r is the physical number distribution
# per unit radius.
#
# Transform to g(q), where
#
#     g(q)dq = f_r(r)dr
#
# so g(q) is number concentration per unit relative mass q.
# ------------------------------------------------------------

p_r = (
    np.exp(-0.5 * ((r_grid - r0) / sigma_r)**2)
    / (sigma_r * np.sqrt(2.0 * np.pi))
)

# Normalize on the finite radius domain
p_r /= np.trapezoid(p_r, r_grid)

# Physical number distribution per unit radius
f_r0 = N0 * p_r

# dq/dr = 3 r^2 / r0^3
dq_dr = 3.0 * r_grid**2 / r0**3

# Number distribution per unit q
g0 = f_r0 / dq_dr

# Check initial number concentration
N_check = np.trapezoid(g0, q)

print(f"\nInitial number check = {N_check:.6e} m^-3")
print(f"Expected             = {N0:.6e} m^-3")


# ------------------------------------------------------------
# 5. Saffman-Turner kernel in q-space
#
# q = m/m0
# r(q) = r0*q^(1/3)
#
# K(q1,q2) = 1.30*(r1+r2)^3*sqrt(epsilon/nu)
# ------------------------------------------------------------

def radius_from_q(q_value):
    return r0 * np.cbrt(q_value)


def saffman_turner_kernel_q(q1, q2):
    r1 = radius_from_q(q1)
    r2 = radius_from_q(q2)

    return (
        1.30
        * (r1 + r2)**3
        * np.sqrt(epsilon / nu)
    )


# ------------------------------------------------------------
# 6. Precompute loss kernel matrix
# ------------------------------------------------------------

Q1 = q[:, None]
Q2 = q[None, :]

K_loss = saffman_turner_kernel_q(Q1, Q2)


# ------------------------------------------------------------
# 7. Precompute gain kernel matrix
#
# Gain term:
#
# G(q) = 1/2 int K(q-y,y) g(q-y) g(y) dy
#
# Because q-y may not be on the grid, interpolation is used.
# ------------------------------------------------------------

Q_target = q[:, None]
Q_y = q[None, :]
Q_minus = Q_target - Q_y

valid_gain = Q_minus >= q[0]

# Avoid taking cube roots of negative values.
Q_minus_safe = np.where(valid_gain, Q_minus, q[0])

K_gain = saffman_turner_kernel_q(
    Q_minus_safe,
    Q_y
)

# No contribution for y > q_target or q-y below the grid
K_gain *= valid_gain
K_gain *= (Q_y <= Q_target)


# ------------------------------------------------------------
# 8. Collision RHS
#
# dg/dt = gain - loss
# ------------------------------------------------------------

def collision_rhs(g):
    # -------------------------
    # LOSS
    # -------------------------
    loss_integrand = K_loss * g[None, :]

    collision_frequency = np.trapezoid(
        loss_integrand,
        q,
        axis=1
    )

    loss = g * collision_frequency

    # -------------------------
    # GAIN
    # -------------------------
    # Interpolate g(q-y) for every target q and y.
    g_minus = np.interp(
        Q_minus,
        q,
        g,
        left=0.0,
        right=0.0
    )

    gain_integrand = (
        K_gain
        * g_minus
        * g[None, :]
    )

    gain = 0.5 * np.trapezoid(
        gain_integrand,
        q,
        axis=1
    )

    return gain - loss


# ------------------------------------------------------------
# 9. Characteristic collision time
#
# This is useful for deciding whether dt is small enough.
# ------------------------------------------------------------

rhs0 = collision_rhs(g0)

loss_rate_integrand = K_loss * g0[None, :]
lambda0 = np.trapezoid(
    loss_rate_integrand,
    q,
    axis=1
)

lambda_max = np.max(lambda0)
tau_min = 1.0 / lambda_max

print("\nInitial collision-time estimate:")
print(f"maximum collision frequency = {lambda_max:.6e} 1/s")
print(f"minimum characteristic time = {tau_min:.6e} s")
print(f"                         = {tau_min/60:.3f} min")


# ------------------------------------------------------------
# 10. RK2 function
# ------------------------------------------------------------

def rk2_step(g, dt):
    k1 = collision_rhs(g)

    g_mid = g + 0.5 * dt * k1

    # Do not silently clip negative values.
    # If this becomes negative, reduce dt.
    if np.min(g_mid) < -1e-12 * max(1.0, np.max(np.abs(g))):
        print("WARNING: negative value encountered at RK2 midpoint.")

    k2 = collision_rhs(g_mid)

    g_new = g + dt * k2

    return g_new


# ------------------------------------------------------------
# 11. Choose the production timestep
#
# Start with 1 s ONLY after checking that timestep convergence
# is acceptable. 
# ------------------------------------------------------------

dt = 0.1                   # s
t_end = 600.0               # s = 1 hour

print(f"\nRK2 timestep:")
print(f"dt = {dt:.4e} s")
print(f"dt/tau_min = {dt/tau_min:.6e}")

# Times to plot
output_times = np.array([
    0.0,
    60.0,
    120.0,
    180.0,
    240.0,
    300.0,
    360.0,
    420.0,
    480.0,
    540.0,
    600.0
])

output_times = output_times[output_times <= t_end]


# ------------------------------------------------------------
# 12. Time integration
# ------------------------------------------------------------

g = g0.copy()

results = {0.0: g.copy()}

t = 0.0
next_output_index = 1

n_steps = int(np.ceil(t_end / dt))

print("\nStarting RK2 integration...")
print(f"dt = {dt} s")
print(f"t_end = {t_end} s")
print(f"number of RK2 steps = {n_steps}")

for step in range(1, n_steps + 1):

    g = rk2_step(g, dt)

    t = step * dt

    # Store requested output times
    while (
        next_output_index < len(output_times)
        and t >= output_times[next_output_index] - 0.5 * dt
    ):
        t_out = output_times[next_output_index]
        results[float(t_out)] = g.copy()
        next_output_index += 1

    # Basic physical check
    if np.min(g) < -1e-10:
        print(
            f"\nWARNING: negative distribution at t={t:.3f} s. "
            "Reduce dt."
        )
        break

    if step % max(1, n_steps // 10) == 0:
        print(f"completed {100*step/n_steps:.0f}%")

print("Integration finished.")


# ------------------------------------------------------------
# 13. Conservation diagnostics
#
# For pure collision/coalescence:
#
# total number decreases,
# total mass should remain approximately constant.
#
# q = m/m0, so integral(q*g dq) is proportional to total mass.
# ------------------------------------------------------------

print("\nConservation check:")

initial_total_number = np.trapezoid(g0, q)
initial_total_mass = np.trapezoid(q * g0, q)

for t_out, g_out in results.items():

    total_number = np.trapezoid(g_out, q)
    total_mass = np.trapezoid(q * g_out, q)

    number_ratio = total_number / initial_total_number
    mass_ratio = total_mass / initial_total_mass

    print(
        f"t={t_out:7.1f} s : "
        f"N/N0={number_ratio:.8f}, "
        f"M/M0={mass_ratio:.8f}, "
        f"min(g)={np.min(g_out):.3e}"
    )


# ------------------------------------------------------------
# 14. GRAPH 1: normalized mass distribution
#
# This is the easiest graph for observing the pattern.
#
# x-axis: q = m/m0
# y-axis: g(q,t)/N0
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for t_out, g_out in results.items():

    p_q = g_out / N0

    plt.plot(
        q,
        p_q,
        linewidth=1.8,
        label=f"t = {t_out:.0f} s"
    )

plt.xlabel(r"Relative droplet mass $q=m/m_0$")
plt.ylabel(r"Normalized mass distribution $g(q,t)/N_0$")
plt.title("Evolution of Droplet Mass Distribution")
plt.xlim(0, 10)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# GRAPH: Change in mass distribution relative to t = 0
# ------------------------------------------------------------

g_initial = results[0.0]

plt.figure(figsize=(9, 6))

for t_out, g_out in results.items():

    delta_g = (g_out - g_initial) / N0

    plt.plot(
        q,
        delta_g,
        linewidth=1.8,
        label=f"t = {t_out:.0f} s"
    )

plt.xlabel(r"Relative droplet mass $q=m/m_0$")
plt.ylabel(r"Change in normalized distribution $(g-g_0)/N_0$")
plt.title("Change in Droplet Mass Distribution")

plt.xlim(0, 10)

plt.axhline(0.0, linewidth=0.8)

plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# GRAPH: Mean droplet mass
# ------------------------------------------------------------

times = []
mean_q = []

for t_out, g_out in results.items():

    total_number = np.trapezoid(g_out, q)
    total_mass = np.trapezoid(q * g_out, q)

    mean_q.append(total_mass / total_number)
    times.append(t_out)

plt.figure(figsize=(9, 6))

plt.plot(
    times,
    mean_q,
    marker="o",
    linewidth=1.8
)

plt.xlabel("Time (s)")
plt.ylabel(r"Mean relative droplet mass $\langle m/m_0\rangle$")
plt.title("Evolution of Mean Droplet Mass")

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. GRAPH 2: same distribution with logarithmic mass axis
#
# Useful because the mass range grows strongly during coalescence.
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for t_out, g_out in results.items():

    p_q = g_out / N0

    plt.semilogx(
        q,
        p_q,
        linewidth=1.8,
        label=f"t = {t_out:.0f} s"
    )

plt.xlabel(r"Relative droplet mass $q=m/m_0$")
plt.ylabel(r"Normalized mass distribution $g(q,t)/N_0$")
plt.title("Evolution of Droplet Mass Distribution (log mass axis)")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. Optional: actual mass axis
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for t_out, g_out in results.items():

    p_q = g_out / N0

    plt.plot(
        m_grid,
        p_q,
        linewidth=1.8,
        label=f"t = {t_out:.0f} s"
    )

plt.xlabel(r"Droplet mass $m$ (kg)")
plt.ylabel(r"Normalized mass distribution $g/N_0$")
plt.title("Droplet Mass Distribution")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.show()
