# Vercel Deployment Setup Guide

**Last Updated**: 2026-01-30
**Purpose**: Step-by-step guide for deploying the frontend to Vercel

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Project Configuration](#project-configuration)
4. [Environment Variables](#environment-variables)
5. [Deployment](#deployment)
6. [Domain Configuration](#domain-configuration)
7. [Preview Deployments](#preview-deployments)
8. [Monitoring & Analytics](#monitoring--analytics)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- [ ] Vercel account (free tier available at [vercel.com](https://vercel.com))
- [ ] Git repository with frontend code (GitHub, GitLab, or Bitbucket)
- [ ] Backend API deployed and accessible (or running locally for testing)
- [ ] Code pushed to your Git repository

---

## Initial Setup

### Step 1: Create Vercel Account

1. Go to [vercel.com](https://vercel.com)
2. Click "Sign Up"
3. Choose your Git provider (GitHub recommended)
4. Authorize Vercel to access your repositories

### Step 2: Connect Git Repository

1. Log in to Vercel Dashboard
2. Click "Add New..." → "Project"
3. Select your Git provider (GitHub/GitLab/Bitbucket)
4. Find your frontend repository in the list
5. Click "Import"

**Note**: If you don't see your repository:
- Check repository permissions in your Git provider
- Ensure Vercel has access to the organization/account
- Try refreshing the repository list

---

## Project Configuration

### Step 3: Configure Build Settings

Vercel auto-detects Vite. Verify the following settings match:

| Setting | Value |
| ------- | ----- |
| **Framework Preset** | Vite |
| **Root Directory** | `.` |
| **Build Command** | `npm run build` (auto-detected) |
| **Output Directory** | `dist` (auto-detected) |
| **Install Command** | `npm install` (auto-detected) |

**Tip**: Click "Edit" next to any setting to customize it.

### Step 4: Vite SPA Build Configuration (Required)

The `grade_my_climb` frontend is a Vite SPA with React Router. A `vercel.json` is
required so that direct navigation and page refreshes work correctly:

```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options",        "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy",        "value": "strict-origin-when-cross-origin" }
      ]
    }
  ]
}
```

**Why the rewrite**: React Router uses `BrowserRouter`. Without it, navigating directly
to `/result/<id>` or refreshing any non-root page returns a Vercel 404. The catch-all
rewrite serves `index.html` for all paths and lets the client router take over.

**Common Options**:

- `rewrites`: Map incoming paths to destinations (required for SPAs)
- `headers`: Set custom HTTP response headers (security hardening)
- `redirects`: Configure URL redirects

---

## Environment Variables

### Step 5: Configure Environment Variables

Environment variables are used to configure your app for different environments (development, preview, production).

#### Required Variables

Add these in Vercel Dashboard → Project Settings → Environment Variables:

| Variable Name | Value | Environment |
|---------------|-------|-------------|
| `VITE_API_URL` | `https://<your-backend-domain>` | Production |
| `VITE_API_URL` | `http://localhost:8000` | Development |

#### How to Add Variables

1. Go to Vercel Dashboard
2. Select your project
3. Go to "Settings" → "Environment Variables"
4. Click "Add New"
5. Enter variable name (e.g., `VITE_API_URL`)
6. Enter value (e.g., `https://<your-backend-domain>`)
7. Select environments:
   - **Production**: Live site
   - **Preview**: Pull request deployments
   - **Development**: Local development (rarely used)
8. Click "Save"

**Important Notes**:
- Variables starting with `VITE_` are exposed to the browser at build time
- Never put secrets in `VITE_` variables
- Changes require redeployment to take effect

### Step 6: Update Backend CORS Settings

The backend CORS is driven by the `BA_CORS_ORIGINS` environment variable. Set it in
your backend deployment to the exact frontend origin:

```bash
BA_CORS_ORIGINS=["https://grade-my-climb.vercel.app"]
```

If a custom domain is acquired later, add it to the list:

```bash
BA_CORS_ORIGINS=["https://grade-my-climb.vercel.app","https://your-custom-domain.com"]
```

**Notes**:
- The default `["*"]` wildcard is for local development only. Always set this env var
  in production.
- `allow_credentials` is intentionally `False` — the API is stateless (no cookies or
  sessions). Do not change this.
- Local development: leave `BA_CORS_ORIGINS` unset (wildcard default) so `http://localhost:5173`
  is accepted without extra config.

---

## Deployment

### Step 7: Deploy to Production

After configuring everything:

1. Click "Deploy" in Vercel Dashboard
2. Wait for build to complete (usually 1-3 minutes)
3. Vercel will show build logs in real-time
4. Once complete, you'll see "✓ Deployment Ready"
5. Click "Visit" to view your deployed site

**Your Site URL**:
- Default: `https://your-project-name.vercel.app`
- Can be customized with a custom domain (see Step 8)

### Step 8: Verify Deployment

1. **Check Homepage**
   - Visit your Vercel URL
   - Ensure page loads correctly
   - Check browser console for errors

2. **Test API Integration**
   - Try uploading an image
   - Check that API calls work
   - Verify CORS is configured correctly

3. **Test Responsive Design**
   - View on mobile device
   - Test on different screen sizes
   - Check that all features work

4. **Performance Check**
   - Use Lighthouse in Chrome DevTools
   - Aim for scores ≥90 in all categories
   - Address any performance issues

---

## Domain Configuration

### Step 9: Add Custom Domain (Optional)

To use your own domain (e.g., `bouldering.yourdomain.com`):

1. **In Vercel Dashboard**:
   - Go to Project Settings → Domains
   - Click "Add"
   - Enter your domain (e.g., `bouldering.yourdomain.com`)
   - Click "Add"

2. **In Your DNS Provider**:
   - Add a CNAME record:
     ```
     Type: CNAME
     Name: bouldering (or @ for root domain)
     Value: cname.vercel-dns.com
     TTL: 3600 (or automatic)
     ```

3. **Wait for Verification**:
   - DNS changes can take 5-48 hours to propagate
   - Vercel will auto-verify and issue SSL certificate
   - Once verified, your site will be accessible at your custom domain

**Tip**: For root domains (e.g., `yourdomain.com`), use A records:
```
Type: A
Name: @
Value: 76.76.21.21
```

---

## Preview Deployments

### Step 10: Configure Preview Deployments

Preview deployments are automatically created for every pull request.

**How It Works**:

1. Create a new branch in Git:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Make changes and push:
   ```bash
   git add .
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```

3. Create a pull request on GitHub

4. Vercel automatically:
   - Detects the PR
   - Builds the code
   - Deploys to a unique preview URL
   - Comments on the PR with the URL

**Preview URL Format**:
```
https://your-project-git-branch-name-username.vercel.app
```

**Benefits**:
- Test changes before merging
- Share with stakeholders for feedback
- Catch issues early
- Preview URLs are temporary (deleted after PR is merged/closed)

### Step 11: Configure Deployment Protection (Optional)

For sensitive projects, you can protect deployments:

1. Go to Project Settings → Git
2. Enable "Deployment Protection"
3. Choose who can trigger deployments:
   - **Only Production Branch**: Deploy only from main branch
   - **All Branches**: Deploy from any branch
   - **Specific Branches**: Deploy from listed branches only

---

## Monitoring & Analytics

### Step 12: Enable Vercel Analytics

Vercel Analytics provides insights into your site's performance:

1. Go to Project Settings → Analytics
2. Click "Enable Analytics"
3. Choose plan (free tier includes 100k events/month)
4. Analytics will automatically start collecting data

**Metrics Provided**:
- Page views
- Unique visitors
- Top pages
- Referrers
- Countries
- Devices
- Performance metrics

### Step 13: Set Up Error Tracking (Optional)

Integrate error tracking services like Sentry:

1. **Sign up for Sentry** (free tier available)

2. **Install Sentry SDK**:

   ```bash
   npm install @sentry/react
   ```

3. **Configure Sentry**:

   ```javascript
   // src/main.tsx (before ReactDOM.createRoot)
   import * as Sentry from '@sentry/react';

   Sentry.init({
     dsn: import.meta.env.VITE_SENTRY_DSN,
     environment: import.meta.env.MODE,
     tracesSampleRate: 1.0,
   });
   ```

4. **Add Sentry DSN to Vercel**:
   - Go to Environment Variables
   - Add `VITE_SENTRY_DSN` with your Sentry DSN

### Step 14: Configure Log Drains (Optional)

For advanced logging, set up log drains:

1. Go to Project Settings → Log Drains
2. Click "Add Log Drain"
3. Choose service (Datadog, Logtail, etc.)
4. Configure endpoint and authentication

**Use Cases**:
- Centralized logging
- Advanced analytics
- Debugging production issues
- Compliance and auditing

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Build Fails

**Symptoms**: Build fails with error messages in Vercel logs

**Solutions**:

1. **Check build logs** in Vercel Dashboard
   - Go to Deployments → Select failed deployment → View logs
   - Look for specific error messages

2. **Test build locally**:
   ```bash
   npm run build
   ```
   - If it fails locally, fix the errors
   - If it works locally, check Vercel configuration

3. **Check Node.js version**:
   - Vercel uses the latest Node.js LTS version by default
   - **Recommended**: Explicitly pin a Node version for reproducible builds
   - **Option 1**: Add `engines` to `package.json` (replace `18.x` with your chosen version):
     ```json
     {
       "engines": {
         "node": "18.x"
       }
     }
     ```
   - **Option 2**: Configure Node version in Vercel Project Settings → General → Node.js Version
   - Consult [Vercel's Node.js runtime documentation](https://vercel.com/docs/functions/runtimes/node-js) for available versions

4. **Clear build cache**:
   - Go to Project Settings → General
   - Scroll to "Build & Development Settings"
   - Click "Clear Cache"
   - Trigger a new deployment

#### Issue 2: API Calls Fail with CORS Errors

**Symptoms**: Browser console shows CORS errors when calling API

**Solutions**:

1. **Check backend CORS configuration**:
   ```python
   # Ensure your backend allows Vercel domain
   allow_origins=[
       "https://your-project.vercel.app",
       "https://*.vercel.app",  # For preview deployments
   ]
   ```

2. **Verify API URL**:
   - Check `VITE_API_URL` in environment variables
   - Ensure it's the correct backend URL
   - Test API endpoint directly in browser

3. **Check credentials**:
   - This API is stateless (no cookies/sessions); `allow_credentials` is `False`
   - Do not set `credentials: 'include'` in fetch options

#### Issue 3: Environment Variables Not Working

**Symptoms**: App can't access environment variables

**Solutions**:

1. **Check variable names**:
   - Must start with `VITE_` to be accessible in the browser (Vite build-time injection)
   - Check for typos

2. **Redeploy after changes**:
   - Environment variable changes require redeployment
   - Go to Deployments → Latest → Redeploy

3. **Check environment selection**:
   - Ensure variables are set for the correct environment (Production/Preview)
   - Preview and Production can have different values

#### Issue 4: Deployment Is Slow

**Symptoms**: Builds take a long time (>5 minutes)

**Solutions**:

1. **Optimize dependencies**:
   - Remove unused packages: `npm prune`
   - Use `npm ci` instead of `npm install` for faster installs

2. **Enable caching**:
   - Vercel caches `node_modules` by default
   - Ensure `.vercel/cache` is not in `.gitignore`

3. **Reduce build output**:
   - Remove console.logs in production
   - Optimize images before committing
   - Use code splitting

#### Issue 5: Images Not Loading from Supabase

**Symptoms**: Route images fail to load

**Solutions**:

1. **Check Supabase Storage CORS**:
   - Go to Supabase Dashboard → Storage → Configuration
   - Add CORS policy for your Vercel domain:
     ```json
     [
       {
         "origin": "https://your-project.vercel.app",
         "methods": ["GET"]
       }
     ]
     ```

2. **Verify public URLs**:
   - Ensure images are in a public bucket
   - Test image URL directly in browser

3. **Use a plain img tag for external Supabase URLs**:

   ```typescript
   <img src={imageUrl} alt="Route image" style={{ maxWidth: '100%' }} />
   ```

#### Issue 6: Preview Deployments Not Creating

**Symptoms**: Pull requests don't trigger preview deployments

**Solutions**:

1. **Check Git integration**:
   - Go to Project Settings → Git
   - Ensure repository is connected
   - Re-authorize if needed

2. **Check deployment settings**:
   - Ensure "Deploy Previews" is enabled
   - Check branch configuration

3. **Check GitHub permissions**:
   - Vercel needs permission to comment on PRs
   - Go to GitHub → Settings → Applications
   - Check Vercel permissions

#### Issue 7: Production Site Shows Old Version

**Symptoms**: Changes don't appear on production site

**Solutions**:

1. **Check deployment status**:
   - Go to Deployments tab
   - Ensure latest deployment is "Ready"
   - Check if deployment was promoted to production

2. **Clear browser cache**:
   - Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
   - Or use incognito mode

3. **Check DNS**:
   - If using custom domain, verify DNS is pointing to Vercel
   - Use `dig yourdomain.com` to check DNS records

---

## Best Practices

### Security

- **Never commit secrets** to Git
- **Use environment variables** for all sensitive data
- **Enable Deployment Protection** for production
- **Configure CORS strictly** (avoid wildcards)

### Performance

- **Enable compression** (Vercel does this automatically)
- **Lazy load** heavy components
- **Monitor bundle size** with `rollup-plugin-visualizer` or `vite-bundle-analyzer`

### Reliability

- **Set up error tracking** (Sentry, etc.)
- **Monitor analytics** regularly
- **Test preview deployments** before merging
- **Use staging environment** for major changes

### Workflow

- **Use preview deployments** for all changes
- **Test locally first**: `npm run build && npm run preview`
- **Review Vercel logs** after each deployment
- **Keep dependencies updated**: `npm update`

---

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html)
- [Vercel CLI](https://vercel.com/docs/cli)
- [Vercel GitHub Integration](https://vercel.com/docs/git/vercel-for-github)
- [Vercel Support](https://vercel.com/support)

---

## Quick Reference

### Common Commands

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from CLI
vercel

# Deploy to production
vercel --prod

# List deployments
vercel ls

# View logs
vercel logs [deployment-url]

# Remove deployment
vercel rm [deployment-url]
```

### Important URLs

| Resource | URL |
| -------- | --- |
| Vercel Dashboard | [Vercel Dashboard](https://vercel.com/dashboard) |
| Your Project | [Your Project](https://vercel.com/[username]/[project]) |
| Deployment Logs | [Deployment Logs](https://vercel.com/[username]/[project]/[deployment-id]) |
| Analytics | [Analytics](https://vercel.com/[username]/[project]/analytics) |
| Settings | [Settings](https://vercel.com/[username]/[project]/settings) |

---

**Questions or issues?** Check the [Vercel Community](https://github.com/vercel/vercel/discussions) or open an issue in our repository.
