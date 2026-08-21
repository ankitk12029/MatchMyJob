# Contributing to MatchMyJob

Thanks for your interest in contributing.

## Reporting a bug

Please open an issue on the [GitHub Issues page](https://github.com/ankitk12029/MatchMyJob/issues), including:

- What you expected to happen vs. what actually happened
- Steps to reproduce (input job title/description, or a sample of your batch file if relevant)
- Any error output or screenshots

## Requesting a feature

Open an issue on the [GitHub Issues page](https://github.com/ankitk12029/MatchMyJob/issues) describing the use case and why the current functionality doesn't cover it.

## Setting up a dev environment

```bash
pip install -r requirements.txt
pip install pytest ruff
```

## Running tests

```bash
pytest tests/ -v
```

The test suite is unit-level only — it does not require the fine-tuned model, GPU, or model downloads, so it should run quickly on any machine.

## Pull request process

1. Fork the repository
2. Create a branch for your change
3. Make your change, and add or update tests as needed
4. Ensure the test suite passes locally (`pytest tests/ -v`) and CI passes on your PR
5. Open a pull request describing what changed and why

## Support

For questions or help, please use [GitHub Issues](https://github.com/ankitk12029/MatchMyJob/issues).

## Code of conduct

Be respectful and assume good faith. Disagreements about code or design are fine; personal attacks are not.
