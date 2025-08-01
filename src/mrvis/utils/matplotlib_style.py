import matplotlib.pyplot as plt
import cmasher

plt.style.use("light")


CMAP = plt.get_cmap("cmr.lavender")
CMAP_DIV = plt.get_cmap("cmr.redshift")


def gcolor(color):
    """
    Convert a color to a hex string.
    """
    if color == "blue":
        c = "#4285f4"
    elif color == "red":
        c = "#ea4335"
    elif color == "yellow":
        c = "#fbbc05"
    elif color == "green":
        c = "#34a853"
    elif color == "violet":
        c = "#886eb7"
    else:
        Exception(f"Unknown color: {color}")
    return c


plt.rcParams.update(
    {
        # 1. Use LaTeX to render all text:
        "text.usetex": True,
        # 2. Choose a serif family and ask for Computer Modern Roman:
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # 3. Match AIP’s default 10 pt base size:
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)
