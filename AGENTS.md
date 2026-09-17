# Repository Instructions



## Repo structure

- `processing/` contains the current analysis and data-processing workflows.
  - `processing/functions/` contains reusable processing modules.
  - `processing/notebooks/` contains analysis and figure notebooks, including `testing_codes/`, `paper_fig_new/`, `paper_fig_upload/`, and `old_code/`.
- `archive/` contains legacy acquisition and processing code. The `scanning-microphone/` subtree includes older scanner, printer, and lock-in-amplifier workflows.
- `stmpy-master/` is the bundled STMpy source tree and package distribution.
- `data/` is a local symlink to the simulation and experimental-data repository; its contents are not necessarily tracked here.
- `output/` is a local symlink used for generated analysis and figure artifacts.
- `README.md` provides the repository overview and links to the original setup instructions.
- `notes/` is a symlink to my personal obsidian vault for note-taking. Should follow the rules in the "rules for notes" section.

## Rules for testing
- Always use `~/miniconda3/envs/metamaterials/bin/python` to run Python code, scripts, notebooks, tests, or other Python-based tooling in this repository.
- Do not use the system `python`, `python3`, or a Python interpreter from another environment.
- If there is no heavy computation involved, run the notebook over with nbconvert to check the execution.

## Rules for notes
- Don't use enter to wrap text. 
- Use `$...$` for inline math and `$$...$$` for display equations on their own separate lines so they compile properly in VS Code and Obsidian.
- When starting a note, use the YAML frontmatter for the properties of the note. The author should be "codex" if generated in one go. If being prompted multiple times, should use both "jiatongy" and "codex." The rest of the properties include "project", "topic", "type", and "status". Prompt the use for what to fill in for these properties. Use only the boolean values `true` or `false` for `status`, with `false` representing an unchecked box.
- Do not use underscores in note titles or note filenames.
- Put a copy of all relevant figures in the `notes/attachments` folder. When referring to a figure from a note, use only its filename with the `![]()` notation (for example, `![](figure.png)`) so that it compiles in Obsidian. Replace spaces in image filenames with `%20` inside the brackets (for example, `![](figure%20one.png)`).
- The notes are intended to be read by humans, so try to keep things clear and concise. Avoid being repetitive and going into unecessary details unless instructed by the user.
