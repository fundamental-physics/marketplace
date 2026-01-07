#!/usr/bin/env python
"""Anesthetic example plots.

Generate example visualizations from posterior chains.

Usage:
    python example_plots.py <path_to_chains>

Outputs:
    corner_plot.png       - Corner/triangle plot of all parameters
    1d_marginals.png      - 1D KDE marginal distributions
    prior_posterior.png   - Prior vs posterior comparison (nested sampling only)
    statistics.txt        - Bayesian statistics (nested sampling only)

Example:
    python example_plots.py chains/polychord_run
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = sys.argv[1]

    # Import anesthetic
    from anesthetic import read_chains, make_1d_axes, make_2d_axes

    print(f"Loading chains from: {root}")
    samples = read_chains(root)
    print(f"Loaded {len(samples)} samples")

    # Get parameter columns (exclude logL, logL_birth, etc.)
    params = [col for col in samples.columns.get_level_values(0)
              if col not in ['logL', 'logL_birth', 'nlive', 'insertion']]
    print(f"Parameters: {params}")

    # Limit to first 5 parameters if too many
    if len(params) > 5:
        print(f"Using first 5 parameters for plots")
        params = params[:5]

    # 1. Corner plot
    print("Creating corner plot...")
    fig, axes = make_2d_axes(params)
    samples.plot_2d(axes, kind='kde')
    fig.savefig('corner_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: corner_plot.png")

    # 2. 1D marginals
    print("Creating 1D marginal plots...")
    fig, axes = make_1d_axes(params)
    samples.plot_1d(axes, kind='kde_1d')
    fig.savefig('1d_marginals.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: 1d_marginals.png")

    # 3. Prior vs posterior (nested sampling only)
    if hasattr(samples, 'prior'):
        print("Creating prior vs posterior comparison...")
        prior = samples.prior()

        fig, axes = make_2d_axes(params[:3] if len(params) >= 3 else params)
        prior.plot_2d(axes, kind='kde', label='Prior', alpha=0.5)
        samples.plot_2d(axes, kind='kde', label='Posterior')

        # Add legend
        axes.iloc[-1, 0].legend(
            bbox_to_anchor=(len(axes)/2, len(axes)),
            loc='lower center'
        )

        fig.savefig('prior_posterior.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: prior_posterior.png")

        # 4. Compute statistics
        print("Computing Bayesian statistics...")
        stats = samples.stats(nsamples=100)

        with open('statistics.txt', 'w') as f:
            f.write("Bayesian Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"log(Z) = {stats['logZ'].mean():.2f} +/- {stats['logZ'].std():.2f}\n")
            f.write(f"D_KL   = {stats['D_KL'].mean():.2f} +/- {stats['D_KL'].std():.2f}\n")
            f.write(f"d_G    = {stats['d_G'].mean():.2f} +/- {stats['d_G'].std():.2f}\n")
            f.write(f"logL_P = {stats['logL_P'].mean():.2f} +/- {stats['logL_P'].std():.2f}\n")

        print("  Saved: statistics.txt")
        print(f"\n  log(Z) = {stats['logZ'].mean():.2f} +/- {stats['logZ'].std():.2f}")
        print(f"  D_KL   = {stats['D_KL'].mean():.2f} +/- {stats['D_KL'].std():.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
