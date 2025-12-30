# Changelog

## [Unreleased]

### Security
- **Improved GitHub OAuth implementation for GitHub Pages deployment**
  - Removed insecure token simulation in production environments
  - Added proper token exchange proxy support with configuration options
  - Enhanced error handling and user feedback for authentication failures
  - Added comprehensive documentation for setting up secure token exchange proxies

### Documentation
- **Added token exchange proxy implementation guide**
  - Detailed examples for Netlify, Vercel, and CloudFlare Workers implementations
  - Security considerations for proxy deployment
  - Configuration instructions for integrating with the UOR MCP client

### Testing
- **Added client-side authentication tests**
  - Test coverage for OAuth flow, token exchange, and auth state management
  - JSDOM-based testing for browser environment simulation
  - Mocked implementations to ensure secure practices

### User Experience
- **Enhanced error handling in auth callback**
  - Clearer error messages for common authentication issues
  - Helpful guidance for setting up token exchange proxies
  - Better feedback for users on authentication status

### Configuration
- **Improved configuration options**
  - Added token exchange proxy URL parameter support
  - Enhanced config handling with proper default values
  - Development mode detection for localhost testing

## [0.1.0] - 2024-04-27

### Initial Release
- Core MCP protocol implementation
- GitHub authentication flow
- UOR integration with GitHub storage
- Client-side application for GitHub Pages deployment