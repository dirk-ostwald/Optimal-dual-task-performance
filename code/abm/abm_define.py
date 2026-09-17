"""
Define estimation parameter spaces of FCDT agent models.
"""

from types import SimpleNamespace                                               # parameter space structure


def abm_define(mod):                                                            # estimation parameter space definition
    """
    Specify the parameter-free A0 or the three-parameter A1 model.

    Input
        mod : SimpleNamespace() object with required attributes:
            .M     : analysis model identifier, 'A0' or 'A1'

    Output
        mod : input SimpleNamespace with the new attributes:
            .Θ : parameter space (SimpleNamespace) with fields:
                .k  : number of estimated parameters, zero or three
                .l  : physical parameter labels in order (beta, sigma, tau)
                .b  : physical coordinate bounds, with zero excluded for tau

    A0 has empty labels and bounds. A1 fits finite beta, sigma, and tau,
    with beta unrestricted in sign, sigma >= 0, and tau > 0.
    Bounds refer directly to (beta, sigma, tau). None denotes an unbounded
    endpoint. The zero lower bound for tau records its theoretical boundary,
    which is excluded. The estimator substitutes a strictly positive lower
    bound and a finite upper bound as numerical safeguards. Starting points
    and numerical temperature guards belong to the estimator.
    Existing parameter space fields are replaced on repeated calls.
    """

    # estimation parameter space
    # -------------------------------------------------------------------------
    mod.Θ            = SimpleNamespace()                                        # model-specific parameter space
    if mod.M == 'A0':                                                           # parameter-free uniform baseline
        mod.Θ.k      = 0                                                        # no fitted parameters
        mod.Θ.l      = ()                                                       # no parameter labels
        mod.Θ.b      = ()                                                       # no optimization bounds
    elif mod.M == 'A1':                                                         # optimal action values with softmax responses
        mod.Θ.k      = 3                                                        # jointly fitted parameters
        mod.Θ.l      = (r'$\beta$', r'$\sigma$', r'$\tau$')                     # physical parameter labels
        mod.Θ.b      = ((None, None), (0.0, None), (0.0, None))                 # physical beta, sigma, and tau boundaries

    return mod                                                                  # updated model specification
