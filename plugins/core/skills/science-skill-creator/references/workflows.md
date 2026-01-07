<!--
  Adapted from Anthropic's skill-creator template:
  https://github.com/anthropics/skills/tree/main/skills/skill-creator
  Modified with physics-specific examples and workflow components.
-->

# Workflow Patterns

## Sequential Workflows

For complex tasks, break operations into clear, sequential steps. It is often helpful to give Claude an overview of the process towards the beginning of SKILL.md:

```markdown
Fitting a cosmological model involves these steps:

1. Load data (run load_data.py)
2. Define likelihood function
3. Configure sampler (edit sampler_config.yaml)
4. Run inference (run run_nested.py)
5. Plot results (run plot_posteriors.py)
```

## Conditional Workflows

For tasks with branching logic, guide Claude through decision points:

```markdown
1. Determine the analysis type:
   **Parameter estimation?** → Follow "Inference workflow" below
   **Model comparison?** → Follow "Evidence calculation workflow" below

2. Inference workflow: [steps for MCMC/nested sampling]
3. Evidence calculation workflow: [steps for Bayes factors]
```

## Physics Workflow Components

Skills often fit into a standard physics research workflow:

```markdown
## Workflow position

This skill provides component (3) Inference in the standard workflow:

(1) Research/Brainstorm → (2) Science → **(3) Inference** → (4) Visualization

**Inputs from upstream:** Likelihood function, data, priors (from Science tools)
**Outputs to downstream:** Posterior samples, evidence (to Visualization tools)
```
