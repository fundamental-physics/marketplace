<!--
  Adapted from Anthropic's skill-creator template:
  https://github.com/anthropics/skills/tree/main/skills/skill-creator
  Modified with physics-specific examples.
-->

# Output Patterns

Use these patterns when skills need to produce consistent, high-quality output.

## Template Pattern

Provide templates for output format. Match the level of strictness to your needs.

**For strict requirements (like data formats or analysis outputs):**

```markdown
## Results structure

ALWAYS use this exact template structure:

# [Analysis Title]

## Summary
[One-paragraph overview of key findings]

## Parameter constraints
| Parameter | Value | Uncertainty |
|-----------|-------|-------------|
| param1    | X.XX  | +/- Y.YY    |

## Diagnostics
- Convergence: [metric and value]
- Effective samples: [number]

## Figures
- Figure 1: [description]
```

**For flexible guidance (when adaptation is useful):**

```markdown
## Results structure

Here is a sensible default format, but adapt as needed:

# [Analysis Title]

## Summary
[Overview of findings]

## Results
[Adapt sections based on what you discover]

## Figures
[Create appropriate visualizations for the analysis type]
```

## Examples Pattern

For skills where output quality depends on seeing examples, provide input/output pairs:

```markdown
## Commit message format for physics code

Generate commit messages following these examples:

**Example 1:**
Input: Added JAX-based nested sampling implementation
Output:
```
feat(inference): implement JAX nested sampler

Add jax_nested.py with vectorized likelihood evaluation
and GPU support for parameter estimation
```

**Example 2:**
Input: Fixed bug where bandflux was computed with wrong filter normalization
Output:
```
fix(bandflux): correct filter normalization in flux calculation

Use AB magnitude zero-point consistently across all filters
```

Follow this style: type(scope): brief description, then detailed explanation.
```

Examples help Claude understand the desired style and level of detail more clearly than descriptions alone.
