"""
Configure shared Matplotlib defaults for ABM figures.
"""


def abm_figure(plt):                                                            # shared Matplotlib configuration
    """
    Apply the project-wide Matplotlib style.

    Input
        plt : matplotlib.pyplot module

    Output
        plt : configured matplotlib.pyplot module
    """
    cfg = {                                                                     # shared publication style
    'font.family'      : 'serif',                                               # primary font family
    'font.serif'       : 'CMU Serif',                                           # serif typeface
    'font.weight'      : 500,                                                   # CMU Roman weight
    'mathtext.fontset' : 'cm',                                                  # Computer Modern mathematics
    'text.usetex'      : True,                                                  # LaTeX text rendering
    'axes.labelweight' : 500,                                                   # CMU axis-label weight
    'axes.titleweight' : 500,                                                   # CMU axis-title weight
    'axes.spines.top'  : False,                                                 # omit upper axis spine
    'axes.spines.right': False}                                                 # omit right axis spine
    plt.rcParams.update(cfg)                                                    # apply shared settings
    return plt                                                                  # configured pyplot module
