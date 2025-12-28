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

## Questions?

If you have questions about contributing or version management, please open an issue on GitHub.

