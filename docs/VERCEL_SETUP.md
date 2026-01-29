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

Vercel will auto-detect your framework. Verify the following settings:

#### For Next.js Projects

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `.` (or `frontend/` if in subdirectory) |
| **Build Command** | `npm run build` (or auto-detected) |
| **Output Directory** | `.next` (auto-detected) |
| **Install Command** | `npm install` (or auto-detected) |

#### For React Projects (Non-Next.js)

| Setting | Value |
|---------|-------|
| **Framework Preset** | Create React App (or Vite) |
| **Root Directory** | `.` |
| **Build Command** | `npm run build` |
| **Output Directory** | `build` (or `dist` for Vite) |
| **Install Command** | `npm install` |

**Tip**: Click "Edit" next to any setting to customize it.

### Step 4: Advanced Build Configuration (Optional)

For advanced use cases, you can create a `vercel.json` file:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "regions": ["iad1"],
  "github": {
    "silent": true
  }
}
```

**Common Options**:

- `regions`: Deploy to specific regions (e.g., `["iad1"]` for US East)
- `github.silent`: Disable GitHub deployment comments
- `redirects`: Configure URL redirects
- `headers`: Set custom HTTP headers

---

## Environment Variables

### Step 5: Configure Environment Variables

Environment variables are used to configure your app for different environments (development, preview, production).

#### Required Variables

Add these in Vercel Dashboard → Project Settings → Environment Variables:

| Variable Name | Value | Environment |
|---------------|-------|-------------|
| `NEXT_PUBLIC_API_URL` | `https://api.yourdomain.com` | Production |
| `NEXT_PUBLIC_API_URL` | `https://staging-api.yourdomain.com` | Preview (optional) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Development (optional) |

#### Optional Variables (If Using Supabase Directly)

| Variable Name | Value | Environment |
|---------------|-------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://your-project.supabase.co` | All |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbGc...` (anon key) | All |

#### How to Add Variables

1. Go to Vercel Dashboard
2. Select your project
3. Go to "Settings" → "Environment Variables"
4. Click "Add New"
5. Enter variable name (e.g., `NEXT_PUBLIC_API_URL`)
6. Enter value (e.g., `https://api.yourdomain.com`)
7. Select environments:
   - **Production**: Live site
   - **Preview**: Pull request deployments
   - **Development**: Local development (rarely used)
8. Click "Save"

**Important Notes**:
- Variables starting with `NEXT_PUBLIC_` are exposed to the browser
- Never put secrets in `NEXT_PUBLIC_` variables
- Changes require redeployment to take effect

### Step 6: Update Backend CORS Settings

Your backend needs to allow requests from your Vercel domain:

```python
# FastAPI backend (src/app.py)
from fastapi.middleware.cors import CORSMiddleware

app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.vercel.app",           # Production
        "https://*.vercel.app",                    # Preview deployments
        "http://localhost:3000",                   # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Security Note**: Replace `*` wildcards with specific domains in production.

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
   npm install @sentry/nextjs
   ```

3. **Configure Sentry**:
   ```javascript
   // sentry.client.config.js
   import * as Sentry from '@sentry/nextjs';

   Sentry.init({
     dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
     environment: process.env.VERCEL_ENV,
     tracesSampleRate: 1.0,
   });
   ```

4. **Add Sentry DSN to Vercel**:
   - Go to Environment Variables
   - Add `NEXT_PUBLIC_SENTRY_DSN` with your Sentry DSN

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
   - Vercel uses Node.js 18 by default
   - To change: Add `engines` to `package.json`:
     ```json
     {
       "engines": {
         "node": "18.x"
       }
     }
     ```

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
   - Check `NEXT_PUBLIC_API_URL` in environment variables
   - Ensure it's the correct backend URL
   - Test API endpoint directly in browser

3. **Check credentials**:
   - If using cookies/authentication, set `allow_credentials=True`
   - In frontend, use `credentials: 'include'` in fetch options

#### Issue 3: Environment Variables Not Working

**Symptoms**: App can't access environment variables

**Solutions**:

1. **Check variable names**:
   - Must start with `NEXT_PUBLIC_` to be accessible in browser
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

3. **Use Next.js Image component**:
   ```typescript
   import Image from 'next/image';

   <Image
     src={imageUrl}
     alt="Route image"
     width={800}
     height={600}
     loader={({ src }) => src} // For external URLs
   />
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

- **Use Next.js Image** component for automatic optimization
- **Enable compression** (Vercel does this automatically)
- **Lazy load** heavy components
- **Monitor bundle size** with `next/bundle-analyzer`

### Reliability

- **Set up error tracking** (Sentry, etc.)
- **Monitor analytics** regularly
- **Test preview deployments** before merging
- **Use staging environment** for major changes

### Workflow

- **Use preview deployments** for all changes
- **Test locally first**: `npm run build && npm start`
- **Review Vercel logs** after each deployment
- **Keep dependencies updated**: `npm update`

---

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Deployment Guide](https://nextjs.org/docs/deployment)
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
|----------|-----|
| Vercel Dashboard | https://vercel.com/dashboard |
| Your Project | https://vercel.com/[username]/[project] |
| Deployment Logs | https://vercel.com/[username]/[project]/[deployment-id] |
| Analytics | https://vercel.com/[username]/[project]/analytics |
| Settings | https://vercel.com/[username]/[project]/settings |

---

**Questions or issues?** Check the [Vercel Community](https://github.com/vercel/vercel/discussions) or open an issue in our repository.
