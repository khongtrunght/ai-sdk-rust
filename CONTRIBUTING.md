# Contributing to AI SDK for Rust

Thank you for your interest in contributing to AI SDK for Rust! We welcome contributions from everyone.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Git
- An OpenAI API key for running integration tests (optional)

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-sdk-rust.git
   cd ai-sdk-rust
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/khongtrunght/ai-sdk-rust.git
   ```
4. Build the project:
   ```bash
   cargo build --workspace
   ```
5. Run tests:
   ```bash
   cargo test --workspace
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### Making Changes

1. **Write Clean Code**: Follow Rust idioms and best practices
2. **Format Your Code**: Run `cargo fmt` before committing
3. **Lint Your Code**: Run `cargo clippy` and address all warnings
4. **Add Tests**: Include tests for new functionality
5. **Update Documentation**: Keep documentation in sync with code changes

### Code Style

This project uses the standard Rust formatting:

```bash
# Check formatting
cargo fmt --all -- --check

# Apply formatting
cargo fmt --all
```

Run Clippy to catch common mistakes:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

### Testing

Run all tests:

```bash
cargo test --workspace
```

Some tests require API keys and are marked with `#[ignore]`. To run these:

```bash
OPENAI_API_KEY=your-key cargo test --workspace -- --ignored
```

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(provider): add support for streaming responses
fix(openai): handle rate limit errors correctly
docs(readme): update installation instructions
```

### Submitting a Pull Request

1. **Update Your Branch**: Sync with upstream before submitting
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push Your Changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**: Go to GitHub and create a PR
   - Fill out the PR template completely
   - Link any related issues
   - Add a clear description of your changes

4. **Code Review**: Address feedback from maintainers
   - Make requested changes in new commits
   - Push updates to the same branch

5. **Merge**: Once approved, a maintainer will merge your PR

## Pull Request Guidelines

- **Keep PRs Focused**: One feature or fix per PR
- **Write Clear Descriptions**: Explain what and why, not just how
- **Include Tests**: All new features must have tests
- **Update Documentation**: Keep docs current
- **Pass CI Checks**: All automated checks must pass
- **Be Responsive**: Reply to review comments promptly

## Types of Contributions

### Bug Reports

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:
- Clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Rust version, etc.)

### Feature Requests

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) and include:
- Problem statement
- Proposed solution
- Alternative approaches considered
- Code examples showing desired usage

### Code Contributions

- Start with an issue to discuss major changes
- Follow the development workflow above
- Ensure all tests pass
- Update relevant documentation

### Documentation

Documentation improvements are always welcome:
- Fix typos or clarify explanations
- Add examples
- Improve rustdoc comments
- Update guides and tutorials

## Project Structure

```
ai-sdk-rust/
├── ai-sdk-provider/      # Core trait definitions
│   ├── src/
│   └── tests/
├── ai-sdk-openai/        # OpenAI implementation
│   ├── src/
│   ├── examples/
│   └── tests/
└── .github/              # CI/CD workflows
```

## Testing Philosophy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Documentation Tests**: Ensure examples in docs work
- **Examples**: Serve as both documentation and tests

## Continuous Integration

All pull requests must pass:
- Format check (`cargo fmt`)
- Clippy lints (`cargo clippy`)
- Tests on Linux, macOS, and Windows
- Documentation build
- MSRV (Minimum Supported Rust Version) check

## Release Process

Releases are automated using [release-plz](https://release-plz.dev/):
1. Merge PRs with conventional commit messages
2. release-plz creates a release PR with version bump and changelog
3. Merge the release PR
4. Automated workflow publishes to crates.io

## Communication

- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and general discussions

## Recognition

Contributors are automatically added to the contributors list on GitHub.

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Reach out to the maintainers

Thank you for contributing!
