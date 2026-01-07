# Fundamental Physics Plugins

Claude Code plugins for fundamental physics analysis, with an initial focus on astrophysics and cosmology.

## Overview

This repository provides skills, tools, and commands for scientific workflows in fundamental physics research. Plugins integrate with Claude Code to assist with:

- Astronomical data processing and analysis
- Cosmological computations
- Literature search and management (arXiv, INSPIRE-HEP, NASA ADS)
- Scientific Python workflows

## Available Plugins

### astro-ph.CO

Astrophysics analysis workflows, scientific computing, and astronomical data processing.

**Skills:**
- `astrophysics` - Observational astronomy, cosmological analysis, FITS/HDF5 data handling, spectroscopy, photometry

### core

Core tools for fundamental physics research: literature access, data repositories, and paper management.

**Skills:**
- `arxiv` - Search arXiv and download paper source (LaTeX/BibTeX)
- `inspire` - INSPIRE-HEP search, citations, BibTeX for high-energy physics
- `ads` - NASA ADS search, citations, BibTeX for astronomy/astrophysics (requires API token)
- `zenodo` - Search and download datasets, software, and research artifacts with DOIs
- `hepdata` - Download experimental data tables from HEP publications (cross-sections, limits, etc.)
- `science-skill-creator` - Guide for creating effective skills for fundamental physics research

**Dependencies:** `requests` (`pip install requests`)

## Installation

Within Claude Code install directly with:

```
/plugin marketplace add git@github.com:fundamental-physics/marketplace.git
```

## Creating Your Own Physics Skills

Want to create a custom skill for your research workflow? Use the `science-skill-creator`:

1. Ask Claude: "Help me create a new skill for [your physics workflow]"
2. The science-skill-creator will guide you through:
   - Understanding your use case with concrete examples
   - Planning reusable components (scripts, references, assets)
   - Implementing the skill following physics research patterns
   - Testing and iterating on the skill

**Example usage:**
- "Help me create a skill for running JAX-based nested sampling on my cosmology models"
- "I need a skill for processing LIGO gravitational wave data"
- "Create a skill for lattice QCD simulations with gauge field configurations"

The skill creator follows a physics workflow model (Research → Science → Inference → Visualization) and helps you build skills that integrate seamlessly with your research pipeline.

## Repository Structure

```
plugins/
└── <plugin-name>/
    ├── .claude-plugin/
    │   └── plugin.json       # Plugin manifest
    ├── skills/
    │   └── <skill-name>/
    │       ├── SKILL.md      # Skill documentation
    │       └── scripts/      # Optional: bundled scripts
    ├── commands/             # Optional: slash commands
    └── agents/               # Optional: subagents
```

## Contributing

### Adding a New Skill to an Existing Plugin

If your skill fits within an existing plugin's domain, add it there:

1. Create a new directory: `plugins/<plugin>/skills/<skill-name>/`
2. Add a `SKILL.md` file with YAML frontmatter and documentation
3. Optionally add supporting scripts in `scripts/`

### Creating a New Plugin

Plugins are organized by arXiv category (e.g., `astro-ph.CO`, `hep-th`, `gr-qc`). To add a new plugin:

1. Create a directory under `plugins/` named after the arXiv category
2. Add skills, commands, or agents as needed
3. **Register the plugin in `.claude-plugin/marketplace.json`** by adding an entry to the `plugins` array:
   ```json
   {
     "name": "your-category",
     "source": "./plugins/your-category"
   }
   ```

### Skill Review Requirements

All new skills must be reviewed against the official Claude Code skills documentation:

**https://code.claude.com/docs/en/skills**

Reviewers should verify:
- Correct YAML frontmatter structure (`name`, `description`, `allowed-tools`)
- Clear trigger conditions in the description
- Appropriate tool permissions
- Well-documented usage examples
- Following progressive disclosure (overview first, details later)

## License

MIT
