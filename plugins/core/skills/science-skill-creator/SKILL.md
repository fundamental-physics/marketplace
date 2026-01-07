---
name: science-skill-creator
description: Guide for creating effective skills for fundamental physics research. This skill should be used when users want to create a new skill (or update an existing skill) that extends Claude's capabilities with specialized scientific knowledge, research workflows, or physics tool integrations.
---

<!--
  Adapted from Anthropic's skill-creator template:
  https://github.com/anthropics/skills/tree/main/skills/skill-creator

  Modified for fundamental physics research use cases with:
  - Physics workflow components (Research → Science → Inference → Visualization)
  - Physics-specific examples (nested sampling, bandflux, anesthetic plotting)
  - Scientific data formats and conventions
-->

# Science Skill Creator

This skill provides guidance for creating effective skills for fundamental physics research.

## About Skills

Skills are modular, self-contained packages that extend Claude's capabilities by providing
specialized knowledge, workflows, and tools. Think of them as "onboarding guides" for specific
domains or tasks—they transform Claude from a general-purpose agent into a specialized research
assistant equipped with procedural knowledge that no model can fully possess.

### What Skills Provide

1. Specialized workflows - Multi-step procedures for physics research
2. Tool integrations - Instructions for working with scientific data formats or APIs
3. Domain expertise - Physics-specific knowledge, conventions, standard parameters
4. Bundled resources - Scripts, references, and assets for complex and repetitive tasks

### Physics Workflow Components

Physics research skills typically provide one or more of these workflow components:

```
(1) Research/Brainstorm → (2) Science → (3) Inference → (4) Visualization
```

1. **Research/Brainstorm** - Literature discovery, data sources, ideation (e.g., `arxiv`, `inspire`, `ads`)
2. **Science** - Domain-specific calculations and data processing (e.g., `astrophysics`, `jax-bandflux`)
3. **Inference** - Statistical analysis and parameter estimation (e.g., `jax-nested-sampler`)
4. **Visualization** - Physics-appropriate plotting (e.g., `anesthetic-plotter`)

Skills may be monolithic (covering multiple components) or modular (single component, composable with others).

## Core Principles

### Concise is Key

The context window is a public good. Skills share the context window with everything else Claude needs: system prompt, conversation history, other Skills' metadata, and the actual user request.

**Default assumption: Claude is already very smart.** Only add context Claude doesn't already have. Challenge each piece of information: "Does Claude really need this explanation?" and "Does this paragraph justify its token cost?"

Prefer concise examples over verbose explanations.

### Set Appropriate Degrees of Freedom

Match the level of specificity to the task's fragility and variability:

**High freedom (text-based instructions)**: Use when multiple approaches are valid, decisions depend on context, or heuristics guide the approach.

**Medium freedom (pseudocode or scripts with parameters)**: Use when a preferred pattern exists, some variation is acceptable, or configuration affects behavior.

**Low freedom (specific scripts, few parameters)**: Use when operations are fragile and error-prone, consistency is critical, or a specific sequence must be followed.

Think of Claude as exploring a path: a narrow bridge with cliffs needs specific guardrails (low freedom), while an open field allows many routes (high freedom).

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
    ├── scripts/          - Executable code (Python/Bash/etc.)
    ├── references/       - Documentation intended to be loaded into context as needed
    └── assets/           - Files used in output (templates, icons, fonts, etc.)
```

#### SKILL.md (required)

Every SKILL.md consists of:

- **Frontmatter** (YAML): Contains `name` and `description` fields. These are the only fields that Claude reads to determine when the skill gets used, thus it is very important to be clear and comprehensive in describing what the skill is, and when it should be used.
- **Body** (Markdown): Instructions and guidance for using the skill. Only loaded AFTER the skill triggers (if at all).

#### Bundled Resources (optional)

##### Scripts (`scripts/`)

Executable code (Python/Bash/etc.) for tasks that require deterministic reliability or are repeatedly rewritten.

- **When to include**: When the same code is being rewritten repeatedly or deterministic reliability is needed
- **Example**: `scripts/arxiv.py` for paper search and LaTeX extraction, `scripts/run_nested.py` for nested sampling
- **Benefits**: Token efficient, deterministic, may be executed without loading into context
- **Note**: Scripts may still need to be read by Claude for patching or environment-specific adjustments

##### References (`references/`)

Documentation and reference material intended to be loaded as needed into context to inform Claude's process and thinking.

- **When to include**: For documentation that Claude should reference while working
- **Examples**: `references/pdg_constants.md` for particle physics parameters, `references/coordinate_systems.md` for astronomy conventions, `references/filter_definitions.md` for photometry
- **Use cases**: Physical constants, data format specifications, API documentation, domain conventions
- **Benefits**: Keeps SKILL.md lean, loaded only when Claude determines it's needed
- **Best practice**: If files are large (>10k words), include grep search patterns in SKILL.md
- **Avoid duplication**: Information should live in either SKILL.md or references files, not both. Prefer references files for detailed information unless it's truly core to the skill.

##### Assets (`assets/`)

Files not intended to be loaded into context, but rather used within the output Claude produces.

- **When to include**: When the skill needs files that will be used in the final output
- **Examples**: `assets/physics_plots.mplstyle` for publication-quality matplotlib styling, `assets/paper_template.tex` for LaTeX templates, `assets/sed_templates/` for SED model files
- **Use cases**: Plot styles, document templates, configuration files, data templates
- **Benefits**: Separates output resources from documentation, enables Claude to use files without loading them into context

#### What to Not Include in a Skill

A skill should only contain essential files that directly support its functionality. Do NOT create extraneous documentation or auxiliary files, including:

- README.md
- INSTALLATION_GUIDE.md
- QUICK_REFERENCE.md
- CHANGELOG.md
- etc.

The skill should only contain the information needed for an AI agent to do the job at hand. It should not contain auxiliary context about the process that went into creating it, setup and testing procedures, user-facing documentation, etc. Creating additional documentation files just adds clutter and confusion.

### Progressive Disclosure Design Principle

Skills use a three-level loading system to manage context efficiently:

1. **Metadata (name + description)** - Always in context (~100 words)
2. **SKILL.md body** - When skill triggers (<5k words)
3. **Bundled resources** - As needed by Claude (Unlimited because scripts can be executed without reading into context window)

#### Progressive Disclosure Patterns

Keep SKILL.md body to the essentials and under 500 lines to minimize context bloat. Split content into separate files when approaching this limit. When splitting out content into other files, it is very important to reference them from SKILL.md and describe clearly when to read them, to ensure the reader of the skill knows they exist and when to use them.

**Key principle:** When a skill supports multiple variations, frameworks, or options, keep only the core workflow and selection guidance in SKILL.md. Move variant-specific details (patterns, examples, configuration) into separate reference files.

**Pattern 1: High-level guide with references**

```markdown
# Cosmological Analysis

## Quick start

Compute distances with astropy:
[code example]

## Advanced features

- **Parameter estimation**: See references/mcmc_guide.md for sampling workflows
- **Power spectra**: See references/power_spectrum.md for Fourier methods
- **Data formats**: See references/fits_handling.md for FITS file operations
```

Claude loads the reference files only when needed.

**Pattern 2: Domain-specific organization**

For Skills with multiple physics domains, organize content by domain to avoid loading irrelevant context:

```
particle-physics/
├── SKILL.md (overview and navigation)
└── references/
    ├── colliders.md (LHC, detector data)
    ├── neutrinos.md (oscillation experiments)
    ├── dark_matter.md (direct/indirect detection)
    └── beyond_sm.md (SUSY, extra dimensions)
```

When a user asks about LHC analysis, Claude only reads colliders.md.

Similarly, for skills supporting multiple methods or frameworks:

```
bayesian-inference/
├── SKILL.md (workflow + method selection)
└── references/
    ├── mcmc.md (Metropolis-Hastings, HMC, emcee)
    ├── nested_sampling.md (dynesty, MultiNest)
    └── variational.md (ADVI, normalizing flows)
```

When the user chooses nested sampling, Claude only reads nested_sampling.md.

**Pattern 3: Conditional details**

Show basic content, link to advanced content:

```markdown
# Spectral Analysis

## Basic fitting

Use scipy.optimize for simple fits.

## Advanced methods

**For MCMC sampling**: See references/mcmc.md
**For nested sampling**: See references/nested.md
**For JAX acceleration**: See references/jax_fitting.md
```

Claude reads the specific reference only when the user needs those features.

**Important guidelines:**

- **Avoid deeply nested references** - Keep references one level deep from SKILL.md. All reference files should link directly from SKILL.md.
- **Structure longer reference files** - For files longer than 100 lines, include a table of contents at the top so Claude can see the full scope when previewing.

## Skill Creation Process

Skill creation involves these steps:

1. Understand the skill with concrete examples
2. Plan reusable skill contents (scripts, references, assets)
3. Initialize the skill (run init_skill.py)
4. Edit the skill (implement resources and write SKILL.md)
5. Package the skill (run package_skill.py)
6. Iterate based on real usage

Follow these steps in order, skipping only if there is a clear reason why they are not applicable.

### Step 1: Understanding the Skill with Concrete Examples

Skip this step only when the skill's usage patterns are already clearly understood. It remains valuable even when working with an existing skill.

To create an effective skill, clearly understand concrete examples of how the skill will be used. This understanding can come from either direct user examples or generated examples that are validated with user feedback.

For example, when building a nested-sampling skill, relevant questions include:

- "What functionality should the skill support? Running samplers, diagnostics, post-processing?"
- "Can you give some examples of how this skill would be used?"
- "I can imagine users asking for things like 'Run nested sampling on this likelihood' or 'Plot the posterior from these chains'. Are there other ways you imagine this skill being used?"
- "What would a user say that should trigger this skill?"
- "Which workflow component(s) does this cover?" (Research, Science, Inference, Visualization)

To avoid overwhelming users, avoid asking too many questions in a single message. Start with the most important questions and follow up as needed for better effectiveness.

Conclude this step when there is a clear sense of the functionality the skill should support.

### Step 2: Planning the Reusable Skill Contents

To turn concrete examples into an effective skill, analyze each example by:

1. Considering how to execute on the example from scratch
2. Identifying what scripts, references, and assets would be helpful when executing these workflows repeatedly

Example: When building a `jax-nested-sampler` skill to handle queries like "Run nested sampling on this likelihood," the analysis shows:

1. Running nested sampling requires re-writing sampler setup code each time
2. A `scripts/run_nested.py` script would be helpful to store in the skill

Example: When designing a `jax-bandflux` skill for queries like "Compute the expected flux in these bands," the analysis shows:

1. Bandflux calculations require loading filter curves and SED models each time
2. A `references/filter_definitions.md` file and `assets/sed_templates/` directory would be helpful

Example: When building an `anesthetic-plotter` skill to handle queries like "Create a corner plot from these samples," the analysis shows:

1. Creating publication-quality plots requires consistent styling
2. An `assets/anesthetic_style.mplstyle` file would be helpful to store in the skill

To establish the skill's contents, analyze each concrete example to create a list of the reusable resources to include: scripts, references, and assets.

### Step 3: Initializing the Skill

At this point, it is time to actually create the skill.

Skip this step only if the skill being developed already exists, and iteration is needed. In this case, continue to the next step.

When creating a new skill from scratch, run the `init_skill.py` script. The script generates a new template skill directory that automatically includes everything a skill requires.

Usage:

```bash
scripts/init_skill.py <skill-name> --path <output-directory>
```

Examples:

```bash
scripts/init_skill.py jax-nested-sampler --path plugins/core/skills
scripts/init_skill.py anesthetic-plotter --path plugins/core/skills
scripts/init_skill.py jax-bandflux --path plugins/astro-ph.CO/skills
```

The script:

- Creates the skill directory at the specified path
- Generates a SKILL.md template with proper frontmatter and TODO placeholders
- Creates example resource directories: `scripts/`, `references/`, and `assets/`
- Adds example files in each directory that can be customized or deleted

After initialization, customize or remove the generated SKILL.md and example files as needed.

### Step 4: Edit the Skill

When editing the (newly-generated or existing) skill, remember that the skill is being created for another instance of Claude to use. Include information that would be beneficial and non-obvious to Claude. Consider what procedural knowledge, domain-specific details, or reusable assets would help another Claude instance execute these tasks more effectively.

#### Learn Proven Design Patterns

Consult these helpful guides based on your skill's needs:

- **Multi-step processes**: See references/workflows.md for sequential workflows and conditional logic
- **Specific output formats or quality standards**: See references/output-patterns.md for template and example patterns

These files contain established best practices for effective skill design.

#### Start with Reusable Skill Contents

To begin implementation, start with the reusable resources identified above: `scripts/`, `references/`, and `assets/` files. Note that this step may require user input. For example, when implementing a `cosmology` skill, the user may need to provide specific parameter conventions or institutional data formats.

Added scripts must be tested by actually running them to ensure there are no bugs and that the output matches what is expected. If there are many similar scripts, only a representative sample needs to be tested to ensure confidence that they all work while balancing time to completion.

Any example files and directories not needed for the skill should be deleted.

#### Update SKILL.md

**Writing Guidelines:** Always use imperative/infinitive form.

##### Frontmatter

Write the YAML frontmatter with `name` and `description`:

- `name`: The skill name
- `description`: This is the primary triggering mechanism for your skill, and helps Claude understand when to use the skill.
  - Include both what the Skill does and specific triggers/contexts for when to use it.
  - Include all "when to use" information here - Not in the body. The body is only loaded after triggering, so "When to Use This Skill" sections in the body are not helpful to Claude.
  - Example description for a `cosmology` skill: "Cosmological calculations and analysis including distance measures, power spectra, parameter estimation, and CMB analysis. Use when working with: (1) Cosmological distances and redshifts, (2) Matter/CMB power spectra, (3) Cosmological parameter fitting, (4) CAMB/CLASS computations."

Do not include any other fields in YAML frontmatter.

##### Body

Write instructions for using the skill and its bundled resources.

### Step 5: Packaging a Skill

Once development of the skill is complete, it can be packaged into a distributable .skill file. The packaging process automatically validates the skill first:

```bash
scripts/package_skill.py <path/to/skill-folder>
```

Optional output directory specification:

```bash
scripts/package_skill.py <path/to/skill-folder> ./dist
```

The packaging script will:

1. **Validate** the skill automatically, checking:
   - YAML frontmatter format and required fields
   - Skill naming conventions and directory structure
   - Description completeness and quality
   - File organization and resource references

2. **Package** the skill if validation passes, creating a .skill file named after the skill (e.g., `my-skill.skill`).

If validation fails, fix the errors and run the packaging command again.

### Step 6: Iterate

After testing the skill, users may request improvements. Often this happens right after using the skill, with fresh context of how the skill performed.

**Iteration workflow:**

1. Use the skill on real tasks
2. Notice struggles or inefficiencies
3. Identify how SKILL.md or bundled resources should be updated
4. Implement changes and test again
