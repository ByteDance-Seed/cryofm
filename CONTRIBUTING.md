# Contributing to CryoFM

Thank you for your interest in contributing to CryoFM! This document provides guidelines for contributing to the project.

## Version Control

This project follows [Semantic Versioning](https://semver.org/) (SemVer) for version management.

### Version Format

Version numbers follow the format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incremented for incompatible API changes
- **MINOR**: Incremented for backward-compatible functionality additions
- **PATCH**: Incremented for backward-compatible bug fixes

### Version Number Location

The version number is stored in `src/cryofm/__init__.py`:

```python
__version__ = "0.1.0"
```

The `pyproject.toml` automatically reads the version from this file, so you only need to update it in one place.

## Commit Messages

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages.

### Commit Message Format

Commit messages should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **`feat`**: A new feature
- **`fix`**: A bug fix
- **`docs`**: Documentation only changes
- **`style`**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **`refactor`**: A code change that neither fixes a bug nor adds a feature
- **`perf`**: A code change that improves performance
- **`test`**: Adding missing tests or correcting existing tests
- **`build`**: Changes that affect the build system or external dependencies
- **`ci`**: Changes to CI configuration files and scripts
- **`chore`**: Other changes that don't modify src or test files

### Examples

```
feat: add support for new data format

feat(datasets): add EMDB data loader

fix: resolve memory leak in sampling process

fix(sampling): correct timestep calculation

docs: update installation instructions

refactor(models): simplify UNet architecture

test: add unit tests for FFT operations
```

## Branch Naming

This project recommends following the [Conventional Branch](https://conventional-branch.github.io/) specification for branch names.

### Branch Name Format

Branch names should follow the format:

```
<type>/<description>
```

### Branch Types

- **`feature/`** or **`feat/`**: For new features
- **`bugfix/`** or **`fix/`**: For bug fixes
- **`hotfix/`**: For urgent fixes
- **`release/`**: For branches preparing a release
- **`chore/`**: For non-code tasks like dependency or documentation updates

### Rules

- Use lowercase letters, numbers, and hyphens only
- Keep names descriptive yet concise
- No consecutive, leading, or trailing hyphens

### Examples

```
feature/add-emdb-support
feat/new-sampling-method
bugfix/fix-memory-leak
fix/correct-fsc-calculation
hotfix/security-patch
release/v1.2.0
chore/update-dependencies
```

## Updating User Guide

The user guide is located in the `user-guide/` directory and uses [MkDocs](https://www.mkdocs.org/) for documentation generation.

### Prerequisites

1. Install MkDocs and required plugins:
```bash
pip install mkdocs mkdocs-material
```

2. Navigate to the user guide directory:
```bash
cd user-guide
```

### Making Changes

1. Edit the Markdown files in `user-guide/docs/` directory
2. If adding a new page, update `user-guide/mkdocs.yml` to include it in the navigation
3. Preview your changes locally:
```bash
mkdocs serve
```
This will start a local server (usually at `http://127.0.0.1:8000`) where you can preview the documentation

### Example: Adding a New Guide

**Step 1:** Create a new Markdown file in the appropriate directory:
```bash
# For example, adding a new how-to guide
touch user-guide/docs/how-to/new-feature.md
```

**Step 2:** Add content to the file:
```markdown
# New Feature Guide

This guide explains how to use the new feature...

## Getting Started

...
```

**Step 3:** Update `user-guide/mkdocs.yml` to include the new page in navigation:
```yaml
nav:
  - How-to Guides:
      - how-to/working-with-emdb.md
      - how-to/new-feature.md  # Add your new page here
```

**Step 4:** Commit your changes:
```bash
git add user-guide/docs/how-to/new-feature.md user-guide/mkdocs.yml
git commit -m "docs: add guide for new feature"
```

### Example: Updating Existing Documentation

**Step 1:** Edit the existing Markdown file:
```bash
# Edit the file you want to update
code user-guide/docs/getting-started/installation.md
```

**Step 2:** Preview your changes:
```bash
cd user-guide
mkdocs serve
```

**Step 3:** Commit your changes:
```bash
git add user-guide/docs/getting-started/installation.md
git commit -m "docs: update installation instructions"
```

## Questions?

If you have questions about contributing or version management, please open an issue on GitHub.

