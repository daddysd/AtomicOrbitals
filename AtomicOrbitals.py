import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

# create figure and 3d axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


available_orbitals = [
    (1, 0, 0, "1s", "red"),
    (2, 0, 0, "2s", "green"),
    (2, 1, -1, "2p", "blue"),
    (2, 1, 0, "2p", "cyan"),
    (2, 1, 1, "2p", "magenta"),
    (3, 0, 0, "3s", "yellow"),
    (3, 1, -1, "3p", "purple"),
    (3, 1, 0, "3p", "brown"),
    (3, 1, 1, "3p", "pink"),
    (4, 0, 0, "4s", "orange"),
    (3, 2, -2, "3d", "lime"),
    (3, 2, -1, "3d", "gray"),
    (3, 2, 0, "3d", "olive"),
    (3, 2, 1, "3d", "navy"),
    (3, 2, 2, "3d", "teal"),
    (4, 1, -1, "4p", "coral"),
    (4, 1, 0, "4p", "gold"),
    (4, 1, 1, "4p", "violet"),
]

current_orbitals = [
    # 1s²
    (1, 0, 0, "1s", "red"),
    # 2s² 2p⁶
    (2, 0, 0, "2s", "green"),
    (2, 1, -1, "2p", "blue"),
    (2, 1, 0, "2p", "cyan"),
    (2, 1, 1, "2p", "magenta"),
    # 3s² 3p⁶
    (3, 0, 0, "3s", "yellow"),
    (3, 1, -1, "3p", "purple"),
    (3, 1, 0, "3p", "brown"),
    (3, 1, 1, "3p", "pink"),
    # 4s² 3d¹⁰ 4p⁶
    (4, 0, 0, "4s", "orange"),
    (3, 2, -2, "3d", "lime"),
    (3, 2, -1, "3d", "gray"),
    (3, 2, 0, "3d", "olive"),
    (3, 2, 1, "3d", "navy"),
    (3, 2, 2, "3d", "teal"),
    (4, 1, -1, "4p", "coral"),
    (4, 1, 0, "4p", "gold"),
    (4, 1, 1, "4p", "violet"),
]

orbital_cache = {}

info_ax = fig.add_axes([0.05, 0.05, 0.6, 0.075])
info_ax.axis('off')
info_text = info_ax.text(0, 0.5, "", fontsize=10)

def get_orbital_info():
    config = {}
    for n, l, m, name, _ in current_orbitals:
        if name not in config:
            config[name] = 0
        config[name] += 1
    
    # group by principal quantum numbers
    n_groups = {}
    for orbital, count in config.items():
        n = orbital[0]
        if n not in n_groups:
            n_groups[n] = []
        n_groups[n].append(f"{orbital}{count if count > 1 else ''}")
    
    result = []
    for n in sorted(n_groups.keys()):
        result.append("  ".join(n_groups[n]))
    
    return "\nElectron Configuration:\n" + "\n".join(result)

# calculate orbital data with caching for performance
def calculate_orbital(n, l, m, scale_factor=0.5):
    # create a cache key
    cache_key = (n, l, m, scale_factor)
    
    # check if we've already calculated this orbital
    if cache_key in orbital_cache:
        return orbital_cache[cache_key]
    

    theta_count = 12  
    phi_count = 12    
    
    theta = np.linspace(0, np.pi, theta_count)
    phi = np.linspace(0, 2*np.pi, phi_count)
    theta, phi = np.meshgrid(theta, phi)
    
    # calculate the spherical harmonics
    Y = sph_harm(abs(m), l, phi, theta)
    
    # for m != 0, we need to handle real spherical harmonics
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    else:
        Y = Y.real
    
    r = n  
    R = np.exp(-r/n) * (2*r/n)**(l)
    
    Psi = R * np.abs(Y)
    
    # scale the orbital for better visualization
    r_scale = scale_factor * n
    
    # convert to Cartesian coordinates
    x = r_scale * Psi * np.sin(theta) * np.cos(phi)
    y = r_scale * Psi * np.sin(theta) * np.sin(phi)
    z = r_scale * Psi * np.cos(theta)
    
    # Cache the results
    orbital_cache[cache_key] = (x, y, z)
    
    return x, y, z

# Precompute all orbitals for faster switching
def precompute_orbitals():
    max_n = max([orbital[0] for orbital in available_orbitals])
    for n, l, m, _, _ in available_orbitals:
        scale = 0.8 + 0.4 * (n / max_n)
        calculate_orbital(n, l, m, scale_factor=scale)
    print("Orbital data precomputed!")

# Function to plot orbitals - optimized for performance
def plot_orbitals():
    ax.clear()
    
    if not current_orbitals:
        ax.text(0, 0, 0, "No orbitals to display", fontsize=14)
        plt.draw()
        return
    
    max_n = max([orbital[0] for orbital in current_orbitals])
    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    seen_orbitals = set()
    proxies = []
    labels = []
    
    sorted_orbitals = sorted(current_orbitals, key=lambda o: o[0])
    
    for idx, (n, l, m, name, color) in enumerate(sorted_orbitals):
        scale = 0.8 + 0.4 * (n / max_n)
        x, y, z = calculate_orbital(n, l, m, scale_factor=scale)
        
        x_min = min(x_min, np.min(x))
        x_max = max(x_max, np.max(x))
        y_min = min(y_min, np.min(y))
        y_max = max(y_max, np.max(y))
        z_min = min(z_min, np.min(z))
        z_max = max(z_max, np.max(z))
        
        alpha = 0.3
        ax.plot_surface(x, y, z, color=color, alpha=alpha, 
                        rstride=1, cstride=1,  
                        linewidth=0, shade=False)  
        
      
        if name not in seen_orbitals:
            proxy = plt.Line2D([0], [0], linestyle="none", marker='o', 
                              markersize=8, markerfacecolor=color)
            proxies.append(proxy)
            labels.append(name)
            seen_orbitals.add(name)
    
    if proxies:
        ax.legend(proxies, labels, loc='upper right', ncol=2, fontsize=8)
    
    if x_min != float('inf'):
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Atomic Orbitals", fontsize=14)
    
    info_text.set_text(get_orbital_info())
    
    plt.gcf().canvas.draw_idle()

def add_orbital(event):
    for orbital in available_orbitals:
        if orbital not in current_orbitals:
            current_orbitals.append(orbital)
            plot_orbitals()
            break

def remove_orbital(event):
    if current_orbitals:
        current_orbitals.pop()
        plot_orbitals()

add_button_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
add_button = Button(add_button_ax, 'Add Orbital')
add_button.on_clicked(add_orbital)

remove_button_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])
remove_button = Button(remove_button_ax, 'Remove Orbital')
remove_button.on_clicked(remove_orbital)

precompute_orbitals()

plot_orbitals()

ax.set_proj_type('ortho') 

plt.tight_layout()
plt.show()