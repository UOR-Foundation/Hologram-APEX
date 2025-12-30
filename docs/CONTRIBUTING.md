# Contributing to MCP (Model Context Protocol)

Thank you for your interest in contributing to the MCP project! This document provides guidelines and standards to help you contribute effectively.

## Code Style and Standards

### TypeScript

- Use TypeScript strict mode for all new code
- Follow the [TypeScript coding guidelines](https://github.com/microsoft/TypeScript/wiki/Coding-guidelines)
- Use explicit types rather than relying on type inference when the type is not obvious
- Use interfaces for object shapes and types for unions, primitives, and tuples
- Use readonly modifiers where appropriate

### Formatting

- We use Prettier for code formatting
- Maximum line length is 100 characters
- Use 2 spaces for indentation
- Use single quotes for strings
- Use trailing commas in multi-line object and array literals
- Place semicolons at the end of statements

### Naming Conventions

- Use PascalCase for class, interface, type, and enum names
- Use camelCase for variable, function, and method names
- Use UPPER_CASE for constants
- Prefix interfaces with `I` only when there is a class of the same name
- Prefix private class members with `_`

### Documentation

- Document all public APIs with JSDoc comments
- Include parameter descriptions and return types
- Document complex logic with inline comments
- Keep documentation up-to-date when changing code

## Git Workflow

### Branches

- Create feature branches from `main`
- Use the naming convention: `feature/short-description` or `fix/issue-description`
- Keep branches focused on a single feature or fix

### Commits

- Write clear, concise commit messages
- Use the imperative mood ("Add feature" not "Added feature")
- Reference issue numbers in commit messages when applicable
- Keep commits atomic and focused on a single change

### Pull Requests

- Create a pull request when your feature or fix is ready for review
- Provide a clear description of the changes
- Include steps to test the changes
- Ensure all tests pass before requesting review
- Address review comments promptly

## Testing

- Write unit tests for all new features
- Maintain or improve code coverage (target: ≥85%)
- Test edge cases and error conditions
- Run the full test suite before submitting a PR

## Performance Considerations

- Be mindful of bundle size when adding dependencies
- Consider browser and Node.js compatibility
- Optimize for performance where appropriate
- Use lazy loading and code splitting when possible

## Dependency Management

- Minimize external dependencies
- Keep dependencies up-to-date
- Document why a dependency is needed when adding a new one
- Consider the license of any new dependency

## Security

- Never commit secrets or credentials
- Validate user input
- Follow security best practices
- Report security vulnerabilities privately

## Getting Help

If you have questions or need help, please:

1. Check existing documentation
2. Search for similar issues
3. Ask in the project's discussion forum
4. Reach out to the maintainers

Thank you for contributing to MCP!
