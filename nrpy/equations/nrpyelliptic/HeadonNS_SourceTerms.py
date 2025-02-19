"""
Construct symbolic expressions for source term.

Author: Thiago Assumpção
        assumpcaothiago **at** gmail **dot* com

License: BSD 2-Clause
"""

# Import needed modules:
import sympy as sp  # For symbolic computations


class SourceTerms:
    """Class sets up and stores sympy expressions for all source terms."""

    def __init__(self, SourceType: str) -> None:
        """
        Compute the expression for a single source.

        :param SourceType: Type of source term. Only valid option: "polytropic_fit".
        """

        # Define symbol for radial variable
        self.r = sp.Symbol("r", real=True)

        # Define symbols for central density, pressure and star radius
        self.rho_central, self.P_central, self.star_radius = sp.symbols(
            "rho_central P_central star_radius", real=True
        )

        # Define symbols for fit parameters
        self.n_rho, self.n_P, self.sigma_rho, self.sigma_P = sp.symbols(
            "n_rho n_P sigma_rho sigma_P", real=True
        )

        if SourceType == "polytropic_fit":
            self._compute_sources()
        else:
            raise ValueError(
                f"Error: '{SourceType}' is an unknown type of SourceType. Only valid option: 'polytropic_fit'."
            )

    def _compute_sources(self) -> None:
        """Compute source terms for a fitted polytrope solution."""

        # Compute density
        self.rho = (
            self.rho_central
            * (1 - self.r**2 / self.star_radius**2)
            * sp.exp(-(self.r**self.n_rho) / self.sigma_rho)
        )
        # Compute pressure
        self.P = (
            self.P_central
            * (1 - self.r**2 / self.star_radius**2)
            * sp.exp(-(self.r**self.n_P) / self.sigma_P)
        )


if __name__ == "__main__":
    import doctest
    import os
    import sys
    import nrpy.validate_expressions.validate_expressions as ve

    results = doctest.testmod()
    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    for sourceType in ["polytropic_fit"]:
        source = SourceTerms(SourceType=sourceType)
        results_dict = ve.process_dictionary_of_expressions(
            source.__dict__, fixed_mpfs_for_free_symbols=True
        )
        ve.compare_or_generate_trusted_results(
            os.path.abspath(__file__),
            os.getcwd(),
            # File basename. If this is set to "trusted_module_test1", then
            #   trusted results_dict will be stored in tests/trusted_module_test1.py
            f"{os.path.splitext(os.path.basename(__file__))[0]}_{sourceType}",
            results_dict,
        )
