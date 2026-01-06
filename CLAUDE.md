# CLAUDE.md

This repository contains Claude Code plugins for fundamental physics analysis, with an initial focus on astrophysics and cosmology.

## Project Structure

- `plugins/` - Plugin directories, each containing skills, commands, and agents
- `plugins/<name>/.claude-plugin/plugin.json` - Plugin manifest
- `plugins/<name>/skills/<skill>/SKILL.md` - Skill documentation files

## Development Guidelines

### Writing Skills

Skills are markdown files with YAML frontmatter that provide domain knowledge to Claude Code. When creating or modifying skills:

1. **Frontmatter required fields:**
   - `name` - Short identifier
   - `description` - When to trigger (include phrases like "use when user asks to...")
   - `allowed-tools` (optional) - Tool restrictions

2. **Content structure:**
   - Start with a brief overview
   - Include concrete code examples
   - Document common workflows
   - List recommended libraries/tools

3. **Scientific accuracy:** Ensure all physics/astronomy content is technically correct. Include proper units, error handling guidance, and references to standard practices.

### Skill Review Checklist

All new skills must be reviewed against the official documentation:
**https://code.claude.com/docs/en/skills**

Verify:
- [ ] YAML frontmatter is valid and complete
- [ ] Description clearly states trigger conditions
- [ ] Tool permissions are appropriate and minimal
- [ ] Examples are working and follow best practices
- [ ] Content follows progressive disclosure (overview → details)

### Plugin Conventions

- Plugin names should be descriptive (e.g., `astro-ph.CO` for astrophysics/cosmology)
- Skills should be narrowly scoped to specific domains
- Prefer explicit tool allowlists over broad permissions
- Include practical code examples users can adapt

## Common Commands

```bash
# Validate plugin structure
claude plugins validate plugins/<plugin-name>

# Test a skill locally
claude --plugin ./plugins/<plugin-name>
```

## Scientific Python Stack

This project assumes familiarity with:
- **astropy** - Astronomical computations, FITS handling, coordinates
- **numpy/scipy** - Numerical computing
- **matplotlib** - Visualization
- **JAX** - Differentiable computing (for advanced workflows)
