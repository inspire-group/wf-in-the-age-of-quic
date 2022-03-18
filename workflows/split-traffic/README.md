# Split traffic

This workflow simulates splitting along CoMPS with different parameters set.

Before running this workflow, ensure you have enough disk space to hold all of the various datasets. This workflow will take all possible parameter combinations listed in `config.yaml` and generate split versions of the `open_world_dataset` for each parameter combination.

To change the dataset you are simulating splitting on, you can change the parameter `open_world_dataset` in `config.yml`.

```bash
# Perform a dry-run
snakemake -n -j
# Make all targets in the workflow
snakemake -j
```


