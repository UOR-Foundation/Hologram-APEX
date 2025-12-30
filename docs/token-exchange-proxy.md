# GitHub OAuth Token Exchange Proxy

This document provides guidance on implementing a secure OAuth token exchange proxy for use with the UOR MCP GitHub Pages deployment.

## Why a Token Exchange Proxy?

When implementing OAuth with GitHub in a client-side only application (like GitHub Pages), there's a fundamental security challenge: the GitHub OAuth flow requires a client secret, but this secret cannot be safely stored in client-side code where it would be exposed to users.

The solution is to implement a token exchange proxy - a small serverless function that:
1. Receives the authorization code from the client
2. Combines it with the securely stored client secret
3. Makes the token exchange request to GitHub's API
4. Returns the token to the client

## Implementation Options

There are several serverless platforms that can host this proxy without requiring a full server:

### 1. Netlify Functions

```javascript
// netlify/functions/token-exchange.js
const axios = require('axios');

exports.handler = async function(event, context) {
  // Enable CORS
  const headers = {
    'Access-Control-Allow-Origin': 'https://your-github-pages-url.github.io',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
  };
  
  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 204,
      headers
    };
  }
  
  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }
  
  try {
    // Parse request body
    const body = JSON.parse(event.body);
    const { code, client_id, redirect_uri } = body;
    
    if (!code) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Authorization code is required' })
      };
    }
    
    // The client secret is stored as an environment variable
    const client_secret = process.env.GITHUB_CLIENT_SECRET;
    
    // Exchange the code for a token with GitHub
    const response = await axios.post(
      'https://github.com/login/oauth/access_token',
      {
        client_id,
        client_secret,
        code,
        redirect_uri
      },
      {
        headers: {
          'Accept': 'application/json'
        }
      }
    );
    
    // Return the token response to the client
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(response.data)
    };
  } catch (error) {
    console.error('Token exchange error:', error);
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: 'Failed to exchange authorization code for token',
        details: error.message
      })
    };
  }
};
```

### 2. Vercel Serverless Functions

```javascript
// api/token-exchange.js
const axios = require('axios');

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', 'https://your-github-pages-url.github.io');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const { code, client_id, redirect_uri } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Authorization code is required' });
    }
    
    // The client secret is stored as an environment variable
    const client_secret = process.env.GITHUB_CLIENT_SECRET;
    
    // Exchange the code for a token with GitHub
    const response = await axios.post(
      'https://github.com/login/oauth/access_token',
      {
        client_id,
        client_secret,
        code,
        redirect_uri
      },
      {
        headers: {
          'Accept': 'application/json'
        }
      }
    );
    
    // Return the token response
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Token exchange error:', error);
    return res.status(500).json({
      error: 'Failed to exchange authorization code for token',
      details: error.message
    });
  }
}
```

### 3. CloudFlare Workers

```javascript
// worker.js
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  // CORS headers
  const corsHeaders = {
    'Access-Control-Allow-Origin': 'https://your-github-pages-url.github.io',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
  
  // Handle OPTIONS request
  if (request.method === 'OPTIONS') {
    return new Response(null, { 
      status: 204,
      headers: corsHeaders
    });
  }
  
  // Only allow POST requests
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  }
  
  try {
    // Parse request body
    const body = await request.json();
    const { code, client_id, redirect_uri } = body;
    
    if (!code) {
      return new Response(JSON.stringify({ error: 'Authorization code is required' }), {
        status: 400,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      });
    }
    
    // The client secret is stored as an environment variable
    const client_secret = GITHUB_CLIENT_SECRET; // Cloudflare Workers environment variable
    
    // Exchange the code for a token with GitHub
    const response = await fetch('https://github.com/login/oauth/access_token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        client_id,
        client_secret,
        code,
        redirect_uri
      })
    });
    
    const data = await response.json();
    
    // Return the token response
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Failed to exchange authorization code for token',
      details: error.message
    }), {
      status: 500,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  }
}
```

## Security Considerations

1. **CORS Configuration**: Ensure the CORS settings only allow requests from your GitHub Pages URL.
2. **Client Secret**: Store the client secret as an environment variable, never hard-code it.
3. **Validation**: Validate all input parameters before making requests to GitHub.
4. **Error Handling**: Implement proper error handling to avoid exposing sensitive information.
5. **Rate Limiting**: Consider implementing rate limiting to prevent abuse.

## Deployment Instructions

### Netlify

1. Create a Netlify account and link it to your GitHub repository
2. In your repository, create a `netlify.toml` file:
   ```toml
   [build]
     functions = "netlify/functions"
   ```
3. Create the function file at `netlify/functions/token-exchange.js`
4. Add your GitHub client secret to Netlify environment variables in the Netlify dashboard
5. Deploy your site to Netlify

### Vercel

1. Create a Vercel account and link it to your GitHub repository
2. Create the API handler at `api/token-exchange.js`
3. Add your GitHub client secret to Vercel environment variables in the Vercel dashboard
4. Deploy your site to Vercel

### CloudFlare Workers

1. Sign up for a CloudFlare Workers account
2. Create a new worker with the code above
3. Configure environment variables in the CloudFlare dashboard
4. Deploy the worker

## Configuring the UOR MCP Client

Once you have deployed your token exchange proxy, you need to configure the UOR MCP client to use it:

1. In your GitHub Pages deployment, add the token exchange proxy URL to the configuration:

```html
<script>
window.MCPConfig.defaultConfig.githubOAuth.tokenExchangeProxy = 'https://your-deployed-proxy.netlify.app/.netlify/functions/token-exchange';
window.MCPConfig.defaultConfig.githubOAuth.clientId = 'your-github-client-id';
</script>
```

2. Alternatively, you can pass it as a URL parameter:

```
https://your-github-pages-url.github.io/?token_exchange_proxy=https://your-deployed-proxy.netlify.app/.netlify/functions/token-exchange&github_client_id=your-github-client-id
```

## Testing

To test your token exchange proxy:

1. Deploy it to your chosen platform
2. Use a tool like Postman to send a POST request with a valid authorization code
3. Verify that you receive a token response
4. Check that your GitHub Pages application can successfully authenticate using the proxy

Remember that a real authorization code can only be used once, so you may need to go through the OAuth flow each time you test.