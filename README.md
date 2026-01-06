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

Core tools for fundamental physics research: literature access and paper management.

**Skills:**
- `arxiv` - Search arXiv and download paper source (LaTeX/BibTeX)
- `inspire` - INSPIRE-HEP search, citations, BibTeX for high-energy physics
- `ads` - NASA ADS search, citations, BibTeX for astronomy/astrophysics (requires API token)

**Dependencies:** `requests` (`pip install requests`)

## Installation

Within Claude Code install directly with:

```
/plugin marketplace add git@github.com:fundamental-physics/marketplace.git
```

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

### Adding a New Skill

1. Create a new directory under the appropriate plugin: `plugins/<plugin>/skills/<skill-name>/`
2. Add a `SKILL.md` file with YAML frontmatter and documentation
3. Follow the skill format specification at https://code.claude.com/docs/en/skills

### Skill Review Requirements

All new skills must be reviewed against the official Claude Code skills documentation:

**https://code.claude.com/docs/en/skills**

Reviewers should verify:
- Correct YAML frontmatter structure (`name`, `description`, `allowed-tools`)
- Clear trigger conditions in the description
- Appropriate tool permissions
- Well-documented usage examples
- Following progressive disclosure (overview first, details later)

### Creating a New Plugin

1. Create a directory under `plugins/`
2. Add `.claude-plugin/plugin.json` with plugin metadata
3. Add skills, commands, or agents as needed
4. Update the root `marketplace.json` if creating a standalone plugin

## License

MIT
