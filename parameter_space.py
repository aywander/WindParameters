import numpy as np
import matplotlib.pyplot as plt
import physconst as pc
import matplotlib.cm as cm
import matplotlib.colors as cl

from cycler import cycler

plt.style.use('seaborn')

use_kpfonts = True


dark_mode = False
if dark_mode:
    fc = 'white'
    fc_i = 'black'
    fc_n = (1, 1, 1, 0.3)
    # TODO: Make the colorpalette below a bit lighter
    tableau10_colors = ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79',
                        'CFCFCF']
else:
    fc = 'black'
    fc_i = 'white'
    fc_n = (0, 0, 0, 0.1)
    tableau10_colors = ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79',
                        'CFCFCF']

plt.rcParams['axes.prop_cycle'] = cycler(color=['#' + s for s in tableau10_colors])

plt.rcParams.update({
    'text.color': fc,
    'axes.labelcolor': fc,
    'xtick.color': fc,
    'ytick.color': fc,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'grid.color': fc_i,
    # 'axes.axisbelow': False,
    'grid.alpha': 0.5,
    'axes.facecolor': fc_n,
    # 'axes.grid.which': 'both',
    # 'axes.grid.axis': 'both',
    'figure.facecolor': (0, 0, 0, 0),
    # 'figure.edgecolor': 'black',
    'savefig.facecolor': (0, 0, 0, 0),
})

if use_kpfonts:
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'text.latex.preamble': [
            r'\usepackage{amsmath}',
            r'\usepackage{amssymb}',
            r'\usepackage{siunitx}',
            r'\usepackage[notextcomp]{kpfonts}',
        ],
    })

else:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.sans-serif': 'Times New Roman',
        'mathtext.fontset': 'cm',
    })

# _______________________________________________________________________________
# Parameters

# Wind power (erg s-1)
power = 1.e44

# Mass outflow rates (Msun / yr)
mdot = np.logspace(-2, 2, 1000)
mdot = np.logspace(-2, 2, 1000)

# Wind velocities (speed of light)
vel = np.logspace(-2, -0.1, 1000)
vel = np.logspace(-2, -0.1, 1000)

# Simulation box width (kpc)
box_width = 6.

# N cells per dimension
ncells = np.array([64, 128, 256, 512, 1024])

# Number of cells that wind should be resolved with
wind_ncells = 8

# Wind inlet radius (kpc)
# radius = 0.5 * box_width / ncells * wind_ncells
radius = 0.1

# Area of inlet
area = np.pi * radius ** 2

# Ambient gas (density [cm^-3], temperature [K])
dens_a = 1.0
temp_a = 1.e7
mu = 0.60364
pres_a = dens_a * pc.kboltz * temp_a


# Plot Mach number or kinetic energy fraction as baground imshow
plot_mach_bg = False


# _______________________________________________________________________________
# Functions


# cgs parameter arrays for broadcasting
mdot_b = mdot[:, None]
vel_b = vel[None, :]

# adiabatic index
gamma = 5. / 3.


def mach(power, mdot, vel):

    fkin_inv = power / (0.5 * mdot * vel * vel)

    return np.sqrt( 2. / (fkin_inv - 1.) / (gamma - 1.) )


def fkin(M):

    return 1. / (1. + 2. / ((gamma - 1.) * M * M))


def pressure(power, mdot, vel, area):

    return (2. * power / vel - mdot * vel) * (gamma - 1.) / (2. * gamma * area)


def density(mdot, vel, area):

    return mdot / (vel * area)


# _______________________________________________________________________________
# Main bit


mdot_cgs_b = mdot_b * pc.msun / pc.yr
vel_cgs_b = vel_b * pc.c
area_cgs = area * pc.kpc * pc.kpc
mach_b = mach(power, mdot_cgs_b, vel_cgs_b)
pres_b = pressure(power, mdot_cgs_b, vel_cgs_b, area_cgs)
dens_b = density(mdot_cgs_b, vel_cgs_b, area_cgs) / (mu * pc.amu)

mach_ticks = [0.3, 1, 3, 10, 30]
fkin_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
pres_ticks = [10, 30, 100, 300, 1000]
dens_ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

f1 = plt.figure(figsize=(6, 4))

if plot_mach_bg:
    plt.imshow(mach_b, cmap=cm.viridis_r, norm=cl.LogNorm(0.3, 30), origin='lower', aspect='auto',
               extent=(np.log10(vel[0]), np.log10(vel[-1]), np.log10(mdot[0]), np.log10(mdot[-1])))
    cb = plt.colorbar(ticks=mach_ticks)
    cb.ax.set_yticklabels(map(str, mach_ticks))
    cb.ax.set_ylabel(r'Mach Number')
else:
    plt.imshow(fkin(mach_b), cmap=cm.pink_r, origin='lower', aspect='auto',
               extent=(np.log10(vel[0]), np.log10(vel[-1]), np.log10(mdot[0]), np.log10(mdot[-1])),
               norm=cl.Normalize(0.0, 1.0))
    cb = plt.colorbar(ticks=fkin_ticks)
    cb.ax.set_yticklabels(map(str, fkin_ticks))
    cb.ax.set_ylabel(r'$\frac{1}{2} \dot{m} v^2$ / L ')


cs = plt.contour(np.log10(vel), np.log10(mdot), mach_b, cmap=cm.pink, linewidths=1.0,
                 levels=mach_ticks[:-2], norm=cl.Normalize(0., 5), linestyles=':')
plt.clabel(cs, fmt='%2.1f', fontsize='smaller')
plt.xlabel(r'$\log_{10} \beta$ [ $c$ ]')
plt.ylabel(r'$\log_{10} \dot{m}$ [ M$_\odot$ yr $^{-1}$ ]')


c1 = plt.contour(np.log10(vel), np.log10(mdot), pres_b / pres_a,
                 cmap=cm.copper_r, linewidths=0.5,
                 levels=pres_ticks, linestyles='-',
                 norm=cl.LogNorm(10, 1000))

plt.clabel(c1, fmt='%3.0f', fontsize='smaller')

plt.grid(False)


c2 = plt.contour(np.log10(vel), np.log10(mdot), dens_b / dens_a, colors='C2',
                 linewidths=0.5,
                 levels=dens_ticks, linestyles='--')
plt.clabel(c2, fmt='%3.2f', fontsize='smaller')


f1.savefig('mach.pdf', bbox_inches='tight')
plt.close(f1)

mach_numbers = np.logspace(-0.5, 1.5, 100)
f2 = plt.figure(figsize=(6, 4))
plt.plot(mach_numbers, fkin(mach_numbers))
plt.xscale('log')
plt.xticks(mach_ticks, map(str, mach_ticks))
plt.xlabel(r'Mach number')
plt.ylabel(r'$\frac{1}{2} \dot{m} v^2$ / L ')
f2.savefig('fkin.pdf', bbox_inches='tight')
plt.close(f1)

