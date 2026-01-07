---
name: science-skill-creator
description: Guide for creating physics research skills. Use when users want to create a new skill (or update an existing skill) for fundamental physics workflows including data analysis, theoretical calculations, statistical inference, scientific visualization, or literature/data discovery.
---

# Science Skill Creator

This skill provides guidance for creating effective skills for fundamental physics research.

## About Skills

Skills are modular, self-contained packages that extend Claude's capabilities by providing specialized knowledge, workflows, and tools. Think of them as "onboarding guides" for specific physics domains—they transform Claude from a general-purpose agent into a specialized research assistant equipped with procedural knowledge that no model can fully possess.

### What Skills Provide

1. **Specialized workflows** - Multi-step procedures for physics research (data pipelines, analysis chains, simulation workflows)
2. **Tool integrations** - Instructions for working with physics data formats and libraries (FITS, HDF5, ROOT, astropy, emcee)
3. **Domain expertise** - Physics-specific knowledge, conventions, and standard parameters
4. **Bundled resources** - Scripts, references, and assets for complex physics tasks

## Physics Workflow Components

Physics research workflows typically follow a pattern combining four types of components:

```
(1) Research/Brainstorm → (2) Science → (3) Inference → (4) Visualization
```

**(1) Research & Brainstorm Tools**: Literature discovery, data sources, ideation
- Examples: arXiv scanning, citation networks, dataset discovery, LLM-assisted brainstorming
- Existing marketplace skills:
  - `arxiv` - Paper search and LaTeX source extraction
  - `inspire` - Particle physics literature and citations
  - `ads` - NASA ADS astronomy publications
  - `zenodo` - Research artifacts and datasets with DOIs
  - `hepdata` - Experimental HEP data tables
- LLM interactions: Hypothesis generation, equation checking, literature synthesis

**(2) Science Tools**: Domain-specific calculations and data processing
- Examples: QFT computations, N-body simulations, detector data processing, cosmological calculations
- Existing: `astrophysics` skill (observational astronomy, FITS/HDF5 data handling, photometry, spectroscopy)

**(3) Inference Tools**: Statistical analysis and parameter estimation
- Examples: MCMC sampling, likelihood maximization, Bayesian model comparison, chi-squared fitting
- Common libraries: emcee, PyMC, Cobaya, dynesty, scipy.optimize

**(4) Visualization Tools**: Physics-appropriate plotting and data presentation
- Examples: Corner plots, power spectra, phase space diagrams, sky maps, Feynman diagrams
- Common libraries: matplotlib, corner.py, healpy, ROOT

### Typical Research Workflow

A complete physics analysis often chains these components:

1. **Research**: Search arxiv/inspire for related work, find existing datasets on hepdata/zenodo
2. **Science**: Process raw data, run simulations, compute observables
3. **Inference**: Fit model parameters, estimate uncertainties, compare models
4. **Visualization**: Create publication-quality figures, diagnostic plots

Skills may implement any subset of this workflow. The existing `arxiv`, `inspire`, `ads`, `zenodo`, and `hepdata` skills handle step (1). The `astrophysics` skill handles steps (2), (3), and (4) together.

### Skill Granularity Patterns

**Monolithic skills (2+3+4)**: Complete domain-specific workflow

Example: The `astrophysics` skill handles data processing (2), parameter fitting (3), and visualization (4) for observational astronomy in a single skill.

Best when: The workflow is highly domain-specific and components are tightly coupled.

**Modular skills (single component)**: Reusable across multiple workflows

<!-- TODO: Update with real skills as marketplace is populated -->
Examples:
- `jax-nested-sampler` (3): JAX-based nested sampling for parameter inference
- `anesthetic-plotter` (4): Posterior visualization with anesthetic
- `jax-bandflux` (2): JAX-based bandflux calculations for photometry

Best when: The component is reused across different physics domains.

**Composable workflows**: Multiple skills interact via clear interfaces

<!-- TODO: Update with real skills as marketplace is populated -->
Example workflow for "Investigate new physics in collider data":
1. `inspire` (1) → search literature for theoretical predictions and existing constraints
2. `hepdata` (1) → fetch published experimental data for comparison
3. `root-reader` (2) → process detector data into structured format
4. `likelihood-fitter` (3) → fit theoretical model, compare with SM predictions
5. `hep-plotter` (4) → create exclusion plots, pull distributions

<!-- TODO: Update with real skills as marketplace is populated -->
Example workflow for "Constrain cosmological parameters from supernova data":
1. `arxiv` (1) → find latest analysis papers, extract priors
2. `jax-bandflux` (2) → compute theoretical bandflux predictions
3. `jax-nested-sampler` (3) → run nested sampling over parameter space
4. `anesthetic-plotter` (4) → visualize posterior distributions

Design for composition by specifying clear input/output formats (HDF5, JSON, numpy arrays).

## Core Principles

### Concise is Key

The context window is a public good. Skills share it with everything else Claude needs.

**Default assumption: Claude is already very smart.** Only add context Claude doesn't already have. Challenge each piece of information: "Does Claude really need this explanation?" and "Does this paragraph justify its token cost?"

Prefer concise examples over verbose explanations.

### Set Appropriate Degrees of Freedom

Match specificity to the task's fragility:

**High freedom (text-based instructions)**: When multiple approaches are valid, or heuristics guide the approach.

**Medium freedom (pseudocode or scripts with parameters)**: When a preferred pattern exists but some variation is acceptable.

**Low freedom (specific scripts, few parameters)**: When operations are fragile, consistency is critical, or a specific sequence must be followed (e.g., numerical algorithms, data format conversions).

### Anatomy of a Skill

Every skill consists of a required SKILL.md file and optional bundled resources:

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter metadata (required)
│   │   ├── name: (required)
│   │   └── description: (required)
│   └── Markdown instructions (required)
└── Bundled Resources (optional)
    ├── scripts/          - Executable code (Python/Bash)
    ├── references/       - Documentation loaded into context as needed
    └── assets/           - Files used in output (templates, styles)
```

#### SKILL.md (required)

- **Frontmatter** (YAML): Contains `name` and `description` fields. These determine when the skill triggers—be clear and comprehensive.
- **Body** (Markdown): Instructions and guidance. Only loaded AFTER the skill triggers.

#### Bundled Resources (optional)

##### Scripts (`scripts/`)

Executable code for physics calculations requiring deterministic reliability.

- **When to include**: Repeated calculations, numerical algorithms, data format conversions
- **Existing examples**:
  - `arxiv` skill: `scripts/arxiv.py` - Downloads papers and extracts LaTeX sources
  - `hepdata` skill: `scripts/hepdata.py` - Fetches experimental data tables
  - `inspire` skill: `scripts/inspire.py` - Queries particle physics literature
- **Physics examples**:
  - `scripts/run_mcmc.py` - MCMC sampler with convergence diagnostics
  - `scripts/compute_power_spectrum.py` - Fourier analysis for cosmology
  - `scripts/parse_root_file.py` - Extract data from ROOT format

##### References (`references/`)

Documentation loaded into context as needed.

- **When to include**: Domain knowledge Claude should reference while working
- **Physics examples**:
  - `references/pdg_constants.md` - Particle Data Group standard values
  - `references/coordinate_systems.md` - Astronomy coordinate conventions
  - `references/likelihood_definitions.md` - Statistical model specifications
  - `references/data_formats.md` - FITS, HDF5, ROOT format documentation
- **Best practice**: If files are large (>10k words), include grep search patterns in SKILL.md

##### Assets (`assets/`)

Files used in output, not loaded into context.

- **When to include**: Templates and resources for final deliverables
- **Physics examples**:
  - `assets/physics_plots.mplstyle` - Publication-quality matplotlib style
  - `assets/paper_template.tex` - LaTeX template with physics packages
  - `assets/analysis_config.yaml` - Pipeline configuration template

#### What to Not Include

Do NOT create extraneous documentation:

- README.md
- INSTALLATION_GUIDE.md
- CHANGELOG.md

The skill should only contain information needed for Claude to do the job.

### Progressive Disclosure Design Principle

Skills use a three-level loading system:

1. **Metadata (name + description)** - Always in context (~100 words)
2. **SKILL.md body** - When skill triggers (<5k words)
3. **Bundled resources** - As needed (unlimited, scripts can execute without loading)

#### Progressive Disclosure Patterns

Keep SKILL.md under 500 lines. Split content when approaching this limit.

**Pattern 1: High-level guide with references**

```markdown
# Cosmological Analysis

## Quick start
Compute distances with astropy:
[code example]

## Advanced features
- **Parameter estimation**: See references/mcmc_guide.md
- **Power spectra**: See references/power_spectrum.md
```

**Pattern 2: Physics domain organization**

```
particle-physics/
├── SKILL.md (overview and subdomain selection)
└── references/
    ├── colliders.md (LHC, detector data)
    ├── neutrinos.md (oscillation experiments)
    ├── dark_matter.md (direct/indirect detection)
    └── beyond_sm.md (SUSY, extra dimensions)
```

When user asks about LHC analysis, Claude only reads colliders.md.

**Pattern 3: Method/framework variants**

```
bayesian-inference/
├── SKILL.md (workflow + sampler selection)
└── references/
    ├── mcmc_samplers.md (Metropolis-Hastings, HMC, emcee)
    ├── nested_sampling.md (dynesty, MultiNest)
    └── variational.md (ADVI, normalizing flows)
```

When user chooses MCMC, Claude only reads mcmc_samplers.md.

**Guidelines:**
- Keep references one level deep from SKILL.md
- For files >100 lines, include a table of contents at the top

## Skill Creation Process

1. Understand the skill with concrete examples
2. Plan reusable skill contents (scripts, references, assets)
3. Initialize the skill directory
4. Edit the skill (implement resources and write SKILL.md)
5. Iterate based on real usage

### Step 1: Understanding the Skill with Concrete Examples

To create an effective physics skill, understand concrete examples of how it will be used.

**Component identification:**
- "Does this skill perform research/discovery (1), domain calculations (2), statistical inference (3), visualization (4), or multiple?"
- "Where does it fit in the 1→2→3→4 workflow?"
- "Is this a complete workflow or a reusable component?"
- "Should this skill interact with other skills? Which ones?"

**Domain context:**
- "What physics domain does this cover?" (particle physics, cosmology, QFT, astrophysics)
- "What standard tools/libraries are used?" (astropy, ROOT, emcee, numpy/scipy)
- "What data formats are standard?" (FITS, HDF5, ROOT, VOTable)

**Workflow understanding:**
- "Walk me through a typical analysis"
- "What calculations are performed repeatedly?"
- "What parameters need to be fitted or inferred?"
- "What plots/outputs are needed for papers?"

Conclude this step with a clear sense of the functionality the skill should support.

### Step 2: Planning the Reusable Skill Contents

Analyze concrete examples to identify useful scripts, references, and assets:

<!-- TODO: Update with real skills as marketplace is populated -->
**Example**: Building a `theory-scanner` skill (component 1) for "Find papers on dark matter direct detection":

1. Searching literature requires querying multiple sources (arxiv, inspire) and synthesizing results
2. Useful resources:
   - `scripts/multi_source_search.py` - Query arxiv + inspire, deduplicate results
   - `references/arxiv_categories.md` - Relevant category codes (hep-ph, hep-ex, astro-ph.CO)
   - Could chain with LLM for summarization and hypothesis generation

<!-- TODO: Update with real skills as marketplace is populated -->
**Example**: Building a `jax-nested-sampler` skill (component 3) for "Run nested sampling on this likelihood":

1. Nested sampling requires JAX-compatible likelihood functions and convergence monitoring
2. Useful resources:
   - `scripts/run_nested.py` - JAX-based nested sampler wrapper
   - `references/nested_sampling_diagnostics.md` - How to assess convergence
   - `assets/sampler_config.yaml` - Configuration template

<!-- TODO: Update with real skills as marketplace is populated -->
**Example**: Building a `jax-bandflux` skill (component 2) for "Compute bandflux predictions":

1. Bandflux calculations require JAX-compatible SED models and filter throughputs
2. Useful resources:
   - `scripts/compute_bandflux.py` - JAX bandflux computation
   - `references/filter_definitions.md` - Standard filter throughput curves
   - `assets/sed_templates/` - Common SED model templates

<!-- TODO: Update with real skills as marketplace is populated -->
**Example**: Building an `anesthetic-plotter` skill (component 4) for "Plot these posterior samples":

1. Posterior visualization requires consistent styling and support for nested sampling outputs
2. Useful resources:
   - `scripts/plot_posteriors.py` - Anesthetic-based plotting wrapper
   - `references/plot_styles.md` - Publication-ready style guidelines
   - `assets/anesthetic_style.mplstyle` - Consistent matplotlib styling

**Example**: Existing `astrophysics` skill (monolithic 2+3+4):

Handles queries like "Analyze this FITS image" or "Calculate cosmological distances":
- Domain calculations (2): Coordinate transforms, photometry, cosmology
- Inference (3): Spectral line fitting, parameter estimation
- Visualization (4): Sky plots, publication-quality figures
- Uses standard libraries (astropy, numpy, matplotlib) without bundled scripts

### Step 3: Initializing the Skill

Create the skill directory structure:

```bash
mkdir -p plugins/core/skills/<skill-name>/scripts
mkdir -p plugins/core/skills/<skill-name>/references
mkdir -p plugins/core/skills/<skill-name>/assets
```

Create SKILL.md with proper frontmatter:

```yaml
---
name: skill-name
description: [Clear description of what the skill does and when to use it]
---

# Skill Name

[Instructions here]
```

### Step 4: Edit the Skill

Remember: the skill is for another Claude instance to use. Include information that would be beneficial and non-obvious.

#### Start with Reusable Contents

Implement the scripts, references, and assets identified in Step 2. This may require user input (e.g., specific data formats, institutional conventions).

**Test scripts** by running them to ensure correctness.

#### Update SKILL.md

**Writing Guidelines:** Use imperative/infinitive form.

**Frontmatter:**
- `name`: The skill name (lowercase, hyphens)
- `description`: Primary triggering mechanism. Include:
  - What the skill does
  - Specific triggers/contexts for when to use it
  - All "when to use" information (the body only loads after triggering)

Example description for a `cosmology` skill:
> "Cosmological calculations and analysis including distance measures, power spectra, parameter estimation, and CMB analysis. Use when working with: (1) Cosmological distances and redshifts, (2) Matter/CMB power spectra, (3) Cosmological parameter fitting, (4) CAMB/CLASS computations."

**Body:** Write instructions for using the skill and its bundled resources.

### Step 5: Iterate

After testing the skill on real tasks:

1. Notice struggles or inefficiencies
2. Identify how SKILL.md or resources should be updated
3. Implement changes and test again

**Common iterations:**
- Add scripts for repeatedly-written code
- Add references for domain knowledge Claude keeps needing
- Refine description to improve triggering accuracy
- Split large SKILL.md into references for progressive disclosure
