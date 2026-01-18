# Pre-commit Hooks Guide

## Overview

This project uses [pre-commit](https://pre-commit.com/) to automatically run quality assurance (QA) checks before each commit. This ensures that all code committed to the repository meets our quality standards.

## What Are Pre-commit Hooks?

Pre-commit hooks are scripts that run automatically before you create a commit. They check your code for issues and can even fix some problems automatically. If any check fails, the commit is blocked until you fix the issues.

## Quality Checks

The following checks run automatically on every commit:

### 1. Standard Hooks (Auto-fix)

- **Trailing whitespace**: Removes trailing whitespace from files
- **End of files**: Ensures files end with a newline
- **Mixed line endings**: Fixes mixed line endings (CRLF/LF)
- **YAML syntax**: Validates YAML files
- **TOML syntax**: Validates TOML files
- **Large files**: Prevents committing files larger than 1MB
- **Merge conflicts**: Detects unresolved merge conflict markers
- **Case conflicts**: Detects case-conflicting filenames

### 2. Ruff (Auto-fix)

- **Linting**: Fast Python linting (ignores E501 line length as per CLAUDE.md)
- **Formatting**: Code formatting (Black-compatible)

### 3. mypy (Manual fix required)

- **Type checking**: Static type analysis to catch type errors
- Checks all Python files except archive directories

### 4. pytest (Manual fix required)

- **Unit tests**: Runs the full test suite with coverage
- **Coverage**: Enforces minimum 85% code coverage (current stage)
- Tests must pass for commit to succeed

### 5. pylint (Manual fix required)

- **Code quality**: Checks code quality and style
- **Score**: Requires minimum score of 8.5/10 (current stage)
- Ignores archive directories

## Installation

### 1. Install pre-commit package

```bash
pip install -r requirements.txt
```

This installs pre-commit along with all other dependencies.

### 2. Install the git hooks

```bash
pre-commit install
```

This sets up the hooks in your `.git/hooks/` directory. You only need to do this once per repository clone.

## Usage

### Automatic Checks on Commit

Once installed, the hooks run automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
```

If any check fails:

1. The commit is blocked
2. Files that can be auto-fixed are modified
3. You'll see error messages for issues requiring manual fixes

### Manual Checks

Run checks on all files without committing:

```bash
pre-commit run --all-files
```

Run checks on staged files only:

```bash
pre-commit run
```

Run a specific hook:

```bash
pre-commit run ruff --all-files
pre-commit run mypy --all-files
pre-commit run pytest-check --all-files
```

### Skipping Hooks (Not Recommended)

In rare cases where you need to commit without running hooks:

```bash
git commit --no-verify -m "Emergency fix"
```

**WARNING**: Only use `--no-verify` for legitimate emergencies. Code that doesn't pass QA checks should not be merged to main.

## Common Issues and Solutions

### Issue: Tests fail due to environment variables

**Problem**: Tests fail because your local `.env` file has different values than the test defaults.

**Solution**: This is expected behavior. The tests are designed to run in a clean environment. Either:

1. Temporarily unset environment variables before committing, or
2. Update your environment variables to match test expectations, or
3. Ensure tests are written to handle your environment properly

### Issue: mypy can't find imports

**Problem**: `Cannot find implementation or library stub for module`

**Solution**: The module might need to be added to `additional_dependencies` in `.pre-commit-config.yaml`. Open an issue or update the config.

### Issue: Pylint score too low

**Problem**: `Your code has been rated at X.XX/10`

**Solution**: Review pylint output and fix the issues. Common fixes:

- Add docstrings to functions/classes
- Fix naming conventions
- Reduce code complexity
- Add type hints

### Issue: Coverage below threshold

**Problem**: `Required test coverage of 85% not reached`

**Solution**: Add tests for uncovered code. Use coverage report to see what's missing:

```bash
pytest tests/ --cov=src --cov-report=html
```

Then open `htmlcov/index.html` in a browser to see detailed coverage.

### Issue: Pre-commit is slow

**Problem**: Running all checks takes time

**Tips**:

- Hooks run in parallel where possible
- Run `pre-commit run` on staged files only (faster than `--all-files`)
- Consider committing more frequently with smaller changes
- Use `git commit -a` to automatically stage and commit modified tracked files (it does not stage new/untracked files)

## Updating Hooks

Pre-commit hooks can be updated to use newer versions:

```bash
pre-commit autoupdate
```

This updates the hook repositories to their latest versions. Review changes before committing the updated `.pre-commit-config.yaml`.

## Configuration

The pre-commit configuration is in `.pre-commit-config.yaml` at the repository root.

### Current Stage vs Final Stage

Quality thresholds are staged based on project completion:

| Check | Current Stage | Final Stage |
| :---: | :-----------: | :---------: |
| Coverage | ≥85% | ≥90% |
| Pylint | ≥8.5/10 | ≥9.0/10 |

When all features are complete, the thresholds in `.pre-commit-config.yaml` will be updated to final stage values.

### Customization

To modify hook behavior, edit `.pre-commit-config.yaml`. Common customizations:

**Skip expensive checks locally (not recommended for main branch):**

```yaml
# Add to specific hook:
stages: [push]  # Only runs on git push, not commit
```

**Change coverage threshold:**

```yaml
# In pytest-check hook:
args:
  - --cov-fail-under=90  # Change from 85 to 90
```

**Add file exclusions:**

```yaml
# In any hook:
exclude: ^(tests/archive/|src/archive/|scripts/)
```

## Integration with CI/CD

Pre-commit hooks run locally before committing. The same checks also run in CI/CD pipelines to catch any issues that bypass local checks (e.g., commits with `--no-verify`).

### Pre-commit.ci (Optional)

The configuration includes optional integration with [pre-commit.ci](https://pre-commit.ci), a free service that:

- Runs hooks on pull requests
- Auto-fixes issues and commits them
- Auto-updates hook versions weekly

To enable, install the pre-commit.ci GitHub app on your repository.

## Best Practices

1. **Run checks early**: Don't wait until commit time to discover issues
2. **Fix issues immediately**: Address failures as soon as they occur
3. **Keep changes small**: Smaller commits are easier to debug when hooks fail
4. **Don't skip hooks**: Use `--no-verify` only for emergencies
5. **Update regularly**: Keep hooks updated with `pre-commit autoupdate`
6. **Read the output**: Hook failure messages tell you exactly what to fix

## Troubleshooting

### Completely reinstall hooks

```bash
pre-commit uninstall
pre-commit clean
pre-commit install
```

### Clear cache

```bash
pre-commit clean
```

### Run with verbose output

```bash
pre-commit run --all-files --verbose
```

### Check hook configuration

```bash
pre-commit validate-config
```

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Pre-commit Hooks Repository](https://github.com/pre-commit/pre-commit-hooks)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Pylint Documentation](https://pylint.readthedocs.io/)

## Support

If you encounter issues with pre-commit hooks:

1. Check this guide for common solutions
2. Review the output carefully - it usually indicates the fix needed
3. Consult the CLAUDE.md for project-specific requirements
4. Open an issue if you believe there's a configuration problem

---

**Remember**: Pre-commit hooks are here to help maintain code quality. They catch issues early, before they reach code review or production.
