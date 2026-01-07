#!/usr/bin/env python3
"""
Science Skill Initializer - Creates a new physics research skill from template

Adapted from Anthropic's skill-creator template:
https://github.com/anthropics/skills/tree/main/skills/skill-creator

Modified for fundamental physics research use cases with physics workflow
components and scientific examples.

Usage:
    init_skill.py <skill-name> --path <path>

Examples:
    init_skill.py jax-nested-sampler --path plugins/core/skills
    init_skill.py anesthetic-plotter --path plugins/core/skills
    init_skill.py jax-bandflux --path plugins/astro-ph.CO/skills
"""

import sys
from pathlib import Path


SKILL_TEMPLATE = """---
name: {skill_name}
description: [TODO: Complete and informative explanation of what the skill does and when to use it. Include WHEN to use this skill - specific scenarios, data types, or physics tasks that trigger it.]
---

# {skill_title}

## Overview

[TODO: 1-2 sentences explaining what this skill enables for physics research]

## Workflow Component

[TODO: Identify which workflow component(s) this skill provides:

**Physics Workflow Components:**
```
(1) Research/Brainstorm → (2) Science → (3) Inference → (4) Visualization
```

1. **Research/Brainstorm** - Literature discovery, data sources, ideation (e.g., arxiv, inspire)
2. **Science** - Domain-specific calculations and data processing (e.g., bandflux, simulations)
3. **Inference** - Statistical analysis and parameter estimation (e.g., nested sampling, MCMC)
4. **Visualization** - Physics-appropriate plotting (e.g., corner plots, spectra)

This skill provides component(s): [TODO: e.g., "(3) Inference" or "(2+3) Science and Inference"]

Delete this section when done - it's just guidance.]

## Structuring This Skill

[TODO: Choose the structure that best fits this skill's purpose. Common patterns:

**1. Workflow-Based** (best for sequential analysis pipelines)
- Works well when there are clear step-by-step procedures
- Example: CMB analysis with "Data loading" → "Power spectrum" → "Parameter fitting" → "Plotting"
- Structure: ## Overview → ## Workflow → ## Step 1 → ## Step 2...

**2. Task-Based** (best for tool collections)
- Works well when the skill offers different operations/capabilities
- Example: Plotting skill with "Quick Start" → "Corner plots" → "Spectra" → "Sky maps"
- Structure: ## Overview → ## Quick Start → ## Task 1 → ## Task 2...

**3. Reference/Guidelines** (best for domain knowledge)
- Works well for physics conventions, data formats, or standards
- Example: Coordinate systems with "Overview" → "Galactic" → "Equatorial" → "Conversions"
- Structure: ## Overview → ## Guidelines → ## Specifications → ## Usage...

**4. Capabilities-Based** (best for integrated systems)
- Works well when the skill provides multiple interrelated features
- Example: Inference skill with "Core Capabilities" → "MCMC" → "Nested Sampling" → "Diagnostics"
- Structure: ## Overview → ## Core Capabilities → ### 1. Feature → ### 2. Feature...

Delete this entire section when done - it's just guidance.]

## [TODO: Replace with the first main section based on chosen structure]

[TODO: Add content here. See examples in existing skills:
- Code samples with numpy, scipy, astropy, JAX
- Decision trees for method selection
- Concrete examples with realistic physics use cases
- References to scripts/references/assets as needed]

## Resources

This skill includes example resource directories:

### scripts/
Executable code (Python) for physics calculations requiring deterministic reliability.

**Examples from marketplace skills:**
- arxiv: `scripts/arxiv.py` - Paper search and LaTeX extraction
- inspire: `scripts/inspire.py` - HEP literature queries
- hepdata: `scripts/hepdata.py` - Experimental data retrieval

**Appropriate for:** MCMC samplers, data processors, format converters, plotting utilities.

### references/
Documentation loaded into context as needed.

**Examples:**
- `references/pdg_constants.md` - Particle physics parameters
- `references/filter_definitions.md` - Photometry filter curves
- `references/convergence_diagnostics.md` - Sampling convergence criteria

**Appropriate for:** Physical constants, data format specs, method documentation, domain conventions.

### assets/
Files used in output, not loaded into context.

**Examples:**
- `assets/physics_plots.mplstyle` - Publication-quality matplotlib styling
- `assets/sed_templates/` - SED model files
- `assets/paper_template.tex` - LaTeX templates

**Appropriate for:** Plot styles, templates, configuration files, data files.

---

**Delete unneeded directories.** Not every skill requires all three types of resources.
"""

EXAMPLE_SCRIPT = '''#!/usr/bin/env python3
"""
Example helper script for {skill_name}

This is a placeholder script. Replace with actual implementation or delete if not needed.

Example physics scripts:
- run_nested.py - Run nested sampling on a likelihood
- compute_bandflux.py - Calculate bandflux predictions
- plot_posteriors.py - Create corner plots from chains
"""

import numpy as np

def main():
    print("This is an example script for {skill_name}")
    # TODO: Add actual script logic here
    # Examples:
    # - Data processing with numpy/scipy
    # - JAX-based calculations
    # - Plotting with matplotlib/anesthetic

if __name__ == "__main__":
    main()
'''

EXAMPLE_REFERENCE = """# Reference Documentation for {skill_title}

This is a placeholder for detailed reference documentation.
Replace with actual reference content or delete if not needed.

## When Reference Docs Are Useful

Reference docs are ideal for:
- Physical constants and standard values
- Data format specifications
- Method documentation (e.g., convergence criteria)
- Domain conventions (e.g., coordinate systems)

## Structure Suggestions

### Physical Constants Example
- Cosmological parameters (H0, Omega_m, etc.)
- Particle masses and couplings
- Unit conversions

### Method Reference Example
- Algorithm overview
- Input/output specifications
- Convergence diagnostics
- Common pitfalls
"""

EXAMPLE_ASSET = """# Example Asset File

This placeholder represents where asset files would be stored.
Replace with actual asset files or delete if not needed.

Asset files are NOT loaded into context, but used in output.

## Common Physics Asset Types

- Plot styles: physics_plots.mplstyle, journal_style.mplstyle
- Templates: paper_template.tex, config_template.yaml
- Data files: filter_curves.ecsv, sed_templates/
- Configuration: sampler_defaults.yaml
"""


def title_case_skill_name(skill_name):
    """Convert hyphenated skill name to Title Case for display."""
    return ' '.join(word.capitalize() for word in skill_name.split('-'))


def init_skill(skill_name, path):
    """
    Initialize a new skill directory with template SKILL.md.

    Args:
        skill_name: Name of the skill
        path: Path where the skill directory should be created

    Returns:
        Path to created skill directory, or None if error
    """
    skill_dir = Path(path).resolve() / skill_name

    if skill_dir.exists():
        print(f"Error: Skill directory already exists: {skill_dir}")
        return None

    try:
        skill_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created skill directory: {skill_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return None

    skill_title = title_case_skill_name(skill_name)
    skill_content = SKILL_TEMPLATE.format(
        skill_name=skill_name,
        skill_title=skill_title
    )

    skill_md_path = skill_dir / 'SKILL.md'
    try:
        skill_md_path.write_text(skill_content)
        print("Created SKILL.md")
    except Exception as e:
        print(f"Error creating SKILL.md: {e}")
        return None

    try:
        scripts_dir = skill_dir / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        example_script = scripts_dir / 'example.py'
        example_script.write_text(EXAMPLE_SCRIPT.format(skill_name=skill_name))
        example_script.chmod(0o755)
        print("Created scripts/example.py")

        references_dir = skill_dir / 'references'
        references_dir.mkdir(exist_ok=True)
        example_reference = references_dir / 'reference.md'
        example_reference.write_text(EXAMPLE_REFERENCE.format(skill_title=skill_title))
        print("Created references/reference.md")

        assets_dir = skill_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)
        example_asset = assets_dir / 'example_asset.txt'
        example_asset.write_text(EXAMPLE_ASSET)
        print("Created assets/example_asset.txt")
    except Exception as e:
        print(f"Error creating resource directories: {e}")
        return None

    print(f"\nSkill '{skill_name}' initialized successfully at {skill_dir}")
    print("\nNext steps:")
    print("1. Edit SKILL.md to complete the TODO items and update the description")
    print("2. Customize or delete the example files in scripts/, references/, and assets/")
    print("3. Run quick_validate.py to check the skill structure")

    return skill_dir


def main():
    if len(sys.argv) < 4 or sys.argv[2] != '--path':
        print("Usage: init_skill.py <skill-name> --path <path>")
        print("\nSkill name requirements:")
        print("  - Hyphen-case identifier (e.g., 'jax-nested-sampler')")
        print("  - Lowercase letters, digits, and hyphens only")
        print("  - Max 64 characters")
        print("\nExamples:")
        print("  init_skill.py jax-nested-sampler --path plugins/core/skills")
        print("  init_skill.py anesthetic-plotter --path plugins/core/skills")
        print("  init_skill.py jax-bandflux --path plugins/astro-ph.CO/skills")
        sys.exit(1)

    skill_name = sys.argv[1]
    path = sys.argv[3]

    print(f"Initializing skill: {skill_name}")
    print(f"   Location: {path}")
    print()

    result = init_skill(skill_name, path)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
