# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build project: `npm run build`
- Build client: `npm run build:client`
- Start server: `npm run start`
- Development mode: `npm run dev`
- Development client: `npm run dev:client`
- Run all tests: `npm run test`
- Run core tests only: `npm run test:core`
- Run a single test: `npx jest path/to/testfile.test.ts -t "test name"`

## Code Style
- Use TypeScript with strict type checking
- Follow interface-based design patterns using abstract classes
- Use camelCase for variables/functions, PascalCase for classes/interfaces
- Group imports by internal/external dependencies
- Abstract core functionality via interfaces before implementation
- Proper error handling with typed errors
- Document all public interfaces and methods with JSDoc comments
- Follow functional programming principles where appropriate
- Write tests for all core functionality with descriptive test names
- Maintain observer-invariant canonical representations