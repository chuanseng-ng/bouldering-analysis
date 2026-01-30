# Frontend Development Workflow

**Last Updated**: 2026-01-30
**Purpose**: Guide for developing the bouldering route analysis frontend

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Lovable Development](#phase-1-lovable-development)
3. [Phase 2: Code Export & Refinement](#phase-2-code-export--refinement)
4. [Phase 3: Vercel Deployment](#phase-3-vercel-deployment)
5. [API Integration](#api-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The frontend development follows a **three-phase approach** designed to balance rapid prototyping with production-grade quality:

```text
Phase 1: Lovable Prototype
    ↓ (rapid UI development)
Phase 2: Code Export & Enhancement
    ↓ (refinement with Claude Code)
Phase 3: Vercel Deployment
    ↓ (production hosting)
```

### Why This Approach?

- **Rapid Prototyping**: Lovable enables fast UI development without writing code
- **Flexibility**: Export to standard React/Next.js for unlimited customization
- **Production Ready**: Vercel provides enterprise-grade hosting and deployment
- **Cost Effective**: Minimize development time while maintaining code quality

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Prototyping | [Lovable](https://lovable.dev) | Visual UI builder |
| Framework | React/Next.js | Frontend framework |
| Enhancement | Claude Code | AI-assisted refinement |
| Hosting | [Vercel](https://vercel.com) | Serverless deployment |
| API | FastAPI (Backend) | REST API |

---

## Phase 1: Lovable Development

### Goal

Create a working prototype with all core functionality using Lovable's no-code platform.

### Prerequisites

- Lovable account (free tier available)
- Backend API running locally or deployed
- Basic understanding of UI/UX principles

### Step 1: Project Setup

1. **Create New Lovable Project**
   - Log in to [Lovable](https://lovable.dev)
   - Click "New Project"
   - Choose a template (or start blank)
   - Name: "Bouldering Route Analysis"

2. **Configure Project Settings**
   - Theme: Dark/Light (user preference)
   - Color scheme: Climbing-themed (blues, greens, earth tones)
   - Fonts: Modern, readable
   - Responsive breakpoints: Mobile-first

3. **Set Backend API URL**
   - Add environment variable: `API_URL`
   - Development: `http://localhost:8000`
   - Production: Your deployed backend URL

### Step 2: Build Core Components

#### 2.1 Image Upload Interface

**Features**:
- Drag-and-drop file upload
- File type validation (JPEG, PNG only)
- File size validation (max 10MB)
- Image preview before upload
- Upload progress indicator
- Error handling with user-friendly messages

**API Integration**:
```javascript
// Upload image
POST /api/v1/routes/upload
Content-Type: multipart/form-data

Response:
{
  "file_id": "uuid",
  "public_url": "https://...",
  "file_size": 1048576,
  "content_type": "image/jpeg",
  "uploaded_at": "2026-01-30T12:00:00Z"
}
```

**UI Elements**:
- Large drop zone with icon
- "Browse Files" button
- Supported formats indicator
- Upload status (idle, uploading, success, error)

#### 2.2 Route Image Display

**Features**:
- Display uploaded route image
- Zoom and pan controls
- Overlay detected holds (after analysis)
- Interactive hold selection
- Hold type indicators (color-coded)

**UI Elements**:
- Full-width image container
- Zoom controls (+/-/reset)
- Pan functionality (drag to move)
- Hold overlays (circles/rectangles)
- Legend for hold types

#### 2.3 Hold Annotation Tools

**Features**:
- Mark start holds (can select multiple)
- Mark finish hold (single selection)
- Clear/reset annotations
- Visual feedback on selection
- Confirm annotations button

**API Integration**:
```javascript
// Set constraints
PUT /api/v1/routes/{id}/constraints
Content-Type: application/json

Body:
{
  "start_hold_ids": [1, 2],
  "finish_hold_id": 10
}
```

**UI Elements**:
- "Mark Start" button (toggle mode)
- "Mark Finish" button (toggle mode)
- "Clear" button
- Selected holds highlighted
- Submit button

#### 2.4 Grade Prediction Display

**Features**:
- Show predicted grade (V0-V17)
- Display confidence/uncertainty
- Show explanation text
- Highlight key contributing factors
- Visualize hold importance

**API Integration**:
```javascript
// Get prediction
GET /api/v1/routes/{id}/prediction

Response:
{
  "grade": "V5",
  "confidence": 0.75,
  "uncertainty": 0.15,
  "explanation": "This route is graded V5 because...",
  "key_factors": [
    {"feature": "max_reach", "importance": 0.8},
    {"feature": "crimp_count", "importance": 0.6}
  ]
}
```

**UI Elements**:
- Large grade display (V-scale)
- Confidence bar/meter
- Explanation card
- Feature importance chart
- Contributing holds highlighted

#### 2.5 Feedback Submission Form

**Features**:
- User can submit their assessment
- Rate prediction accuracy
- Add comments/notes
- Optional email for follow-up

**API Integration**:
```javascript
// Submit feedback
POST /api/v1/routes/{id}/feedback
Content-Type: application/json

Body:
{
  "user_grade": "V6",
  "is_accurate": false,
  "comments": "Feels harder than V5..."
}
```

**UI Elements**:
- Grade dropdown (V0-V17)
- "Was this accurate?" toggle
- Comments textarea
- Submit button
- Thank you message

#### 2.6 Route History/Gallery

**Features**:
- List previously analyzed routes
- Thumbnail previews
- Grade and date
- Click to view details
- Pagination or infinite scroll

**API Integration**:
```javascript
// List routes
GET /api/v1/routes?page=1&limit=20

Response:
{
  "routes": [
    {
      "id": "uuid",
      "image_url": "https://...",
      "grade": "V5",
      "created_at": "2026-01-30T12:00:00Z"
    },
    ...
  ],
  "total": 100,
  "page": 1,
  "pages": 5
}
```

**UI Elements**:
- Grid layout (responsive)
- Route cards with thumbnails
- Pagination controls
- Filter/sort options

### Step 3: Design System

Establish consistent design patterns:

#### Colors

```text
Primary: #2563eb (blue)
Secondary: #10b981 (green)
Accent: #f59e0b (orange)
Error: #ef4444 (red)
Success: #22c55e (green)
Background: #f9fafb (light) / #1f2937 (dark)
Text: #111827 (light) / #f9fafb (dark)
```

#### Typography

```text
Headings: Inter, system-ui (bold)
Body: Inter, system-ui (regular)
Code: Fira Code, monospace
```

#### Spacing

```text
Base unit: 4px
xs: 4px
sm: 8px
md: 16px
lg: 24px
xl: 32px
2xl: 48px
```

#### Components

- Buttons: Rounded, shadow on hover
- Cards: White background, subtle shadow
- Inputs: Border, focus ring
- Modals: Centered, backdrop blur

### Step 4: User Testing

1. **Internal Testing**
   - Test all user flows
   - Check responsive design (mobile, tablet, desktop)
   - Validate API integration
   - Test error scenarios

2. **External Testing**
   - Share prototype with 3-5 users
   - Observe usage patterns
   - Collect feedback
   - Identify pain points

3. **Iterate**
   - Fix critical issues
   - Improve confusing flows
   - Polish interactions
   - Update documentation

### Deliverables

- [ ] Working Lovable prototype
- [ ] Design system documentation
- [ ] API integration documentation
- [ ] User testing feedback summary
- [ ] Screenshots/demo video

---

## Phase 2: Code Export & Refinement

### Goal

Export the Lovable prototype to code and enhance it with advanced features using Claude Code.

### Prerequisites

- Completed Phase 1 (Lovable prototype)
- Git repository set up
- Node.js and npm/yarn installed
- Claude Code CLI installed

### Step 1: Export from Lovable

1. **Export Project**
   - In Lovable, go to Project Settings
   - Click "Export to Code"
   - Choose format: Next.js (recommended) or React
   - Download ZIP file

2. **Extract and Initialize**
   ```bash
   # Extract project
   unzip bouldering-analysis-frontend.zip
   cd bouldering-analysis-frontend

   # Initialize Git
   git init
   git add .
   git commit -m "Initial commit from Lovable export"

   # Install dependencies
   npm install
   ```

3. **Verify Local Setup**
   ```bash
   # Start development server
   npm run dev

   # Open browser
   open http://localhost:3000
   ```

### Step 2: Code Review & Cleanup

Use Claude Code to review and optimize the exported code:

1. **Run Initial Assessment**
   ```bash
   # In Claude Code
   /plan Review exported Lovable code for quality and optimization opportunities
   ```

2. **Code Quality Improvements**
   - Remove unused dependencies
   - Optimize imports
   - Fix ESLint warnings
   - Add TypeScript types (if not present)
   - Improve component structure

3. **Performance Optimizations**
   - Add code splitting for large components
   - Implement lazy loading for images
   - Optimize bundle size
   - Add caching strategies
   - Use React.memo for expensive components

### Step 3: Advanced Features

Enhance the prototype with features that are difficult in no-code:

#### 3.1 Advanced Interactions

```typescript
// Keyboard shortcuts
import { useHotkeys } from 'react-hotkeys-hook';

function AnnotationTools() {
  useHotkeys('s', () => toggleStartMode());
  useHotkeys('f', () => toggleFinishMode());
  useHotkeys('escape', () => clearSelection());
  useHotkeys('enter', () => submitAnnotations());

  // ... component logic
}
```

#### 3.2 Touch Gestures

```typescript
// Pinch-to-zoom for mobile
import { useGesture } from '@use-gesture/react';

function RouteImage({ src }) {
  const [scale, setScale] = useState(1);

  const bind = useGesture({
    onPinch: ({ offset: [scale] }) => setScale(scale),
  });

  return <img {...bind()} style={{ transform: `scale(${scale})` }} />;
}
```

#### 3.3 Improved Error Handling

```typescript
// Error boundary with retry logic
class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  retry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <ErrorFallback
          error={this.state.error}
          onRetry={this.retry}
        />
      );
    }
    return this.props.children;
  }
}
```

#### 3.4 Accessibility Improvements

```typescript
// Add ARIA labels and keyboard navigation
<button
  onClick={handleUpload}
  aria-label="Upload route image"
  aria-describedby="upload-help"
>
  Upload
</button>
<div id="upload-help" className="sr-only">
  Supported formats: JPEG, PNG. Max size: 10MB.
</div>
```

### Step 4: Testing

Add comprehensive test suite:

#### 4.1 Unit Tests (Jest/Vitest)

```bash
# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest
```

```typescript
// Example: UploadButton.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { UploadButton } from './UploadButton';

describe('UploadButton', () => {
  it('calls onUpload when file is selected', () => {
    const onUpload = jest.fn();
    render(<UploadButton onUpload={onUpload} />);

    const input = screen.getByLabelText(/upload/i);
    const file = new File(['content'], 'route.jpg', { type: 'image/jpeg' });

    fireEvent.change(input, { target: { files: [file] } });

    expect(onUpload).toHaveBeenCalledWith(file);
  });
});
```

#### 4.2 Integration Tests

```typescript
// Test API integration
import { render, screen, waitFor } from '@testing-library/react';
import { RoutePrediction } from './RoutePrediction';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  rest.get('/api/v1/routes/:id/prediction', (req, res, ctx) => {
    return res(ctx.json({
      grade: 'V5',
      confidence: 0.75,
      explanation: 'Test explanation'
    }));
  })
);

beforeAll(() => server.listen());
afterAll(() => server.close());

test('displays prediction from API', async () => {
  render(<RoutePrediction routeId="123" />);

  await waitFor(() => {
    expect(screen.getByText('V5')).toBeInTheDocument();
    expect(screen.getByText(/Test explanation/)).toBeInTheDocument();
  });
});
```

#### 4.3 E2E Tests (Playwright)

```bash
# Install Playwright
npm install --save-dev @playwright/test
```

```typescript
// e2e/upload-route.spec.ts
import { test, expect } from '@playwright/test';

test('complete route analysis flow', async ({ page }) => {
  // Navigate to app
  await page.goto('http://localhost:3000');

  // Upload image
  await page.setInputFiles('input[type="file"]', 'test-route.jpg');
  await expect(page.locator('.image-preview')).toBeVisible();

  // Mark start holds
  await page.click('button:has-text("Mark Start")');
  await page.click('.hold[data-id="1"]');
  await page.click('.hold[data-id="2"]');

  // Mark finish hold
  await page.click('button:has-text("Mark Finish")');
  await page.click('.hold[data-id="10"]');

  // Submit and wait for prediction
  await page.click('button:has-text("Analyze")');
  await expect(page.locator('.grade-result')).toBeVisible();
  await expect(page.locator('.grade-result')).toContainText(/V\d+/);
});
```

### Step 5: Documentation

Document the codebase for future developers:

```markdown
# Frontend Documentation

## Project Structure

```
frontend/
├── src/
│   ├── components/         # React components
│   │   ├── upload/         # Upload-related components
│   │   ├── annotation/     # Annotation tools
│   │   ├── prediction/     # Prediction display
│   │   └── common/         # Shared components
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utilities and helpers
│   ├── api/                # API client
│   ├── types/              # TypeScript types
│   └── styles/             # Global styles
├── public/                 # Static assets
├── tests/                  # Test files
└── e2e/                    # E2E tests
```

## Running Locally

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Run tests
npm test

# Run E2E tests
npm run test:e2e

# Build for production
npm run build
```
```

### Deliverables

- [ ] Cleaned, optimized codebase
- [ ] Test suite with ≥80% coverage
- [ ] Performance benchmarks documented
- [ ] Developer documentation
- [ ] TypeScript types (if applicable)

---

## Phase 3: Vercel Deployment

### Goal

Deploy the frontend to Vercel with continuous deployment from Git.

See [VERCEL_SETUP.md](VERCEL_SETUP.md) for detailed deployment instructions.

### Quick Summary

1. **Connect Git Repository**
   - Push code to GitHub/GitLab/Bitbucket
   - Import project in Vercel

2. **Configure Build**
   - Framework: Next.js
   - Build command: `npm run build`
   - Output directory: `.next`

3. **Set Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://api.yourdomain.com
   ```

4. **Deploy**
   - Automatic deployment on push to main
   - Preview deployments for PRs

---

## API Integration

### API Client Setup

Create a reusable API client:

```typescript
// src/lib/api.ts
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptors for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
    }
    return Promise.reject(error);
  }
);
```

### API Functions

```typescript
// src/api/routes.ts
import { api } from '@/lib/api';

export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await api.post('/api/v1/routes/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return data;
}

export async function createRoute(imageUrl: string) {
  const { data } = await api.post('/api/v1/routes', {
    image_url: imageUrl,
  });
  return data;
}

export async function getRoute(id: string) {
  const { data } = await api.get(`/api/v1/routes/${id}`);
  return data;
}

export async function getPrediction(id: string) {
  const { data } = await api.get(`/api/v1/routes/${id}/prediction`);
  return data;
}

export async function submitFeedback(id: string, feedback: any) {
  const { data } = await api.post(`/api/v1/routes/${id}/feedback`, feedback);
  return data;
}
```

### Error Handling

```typescript
// src/hooks/useApi.ts
import { useState } from 'react';

export function useApi<T>(apiFunc: (...args: any[]) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = async (...args: any[]) => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiFunc(...args);
      setData(result);
      return result;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, execute };
}
```

---

## Best Practices

### 1. Component Organization

- **One component per file**
- **Colocate related files** (component, styles, tests)
- **Use barrel exports** (index.ts files)
- **Keep components focused** (single responsibility)

### 2. State Management

- **Use React hooks** for local state
- **Use context** for global state (theme, user)
- **Consider Zustand/Redux** for complex state
- **Minimize prop drilling**

### 3. Performance

- **Use React.memo** for expensive components
- **Lazy load** routes and heavy components
- **Optimize images** (Next.js Image component)
- **Code splitting** for large bundles

### 4. Accessibility

- **Semantic HTML** (use proper elements)
- **ARIA labels** for interactive elements
- **Keyboard navigation** (tab order, shortcuts)
- **Screen reader support** (alt text, descriptions)

### 5. Security

- **Validate user input** on frontend and backend
- **Sanitize HTML** to prevent XSS
- **Use HTTPS** for all API calls
- **Protect sensitive data** (don't expose keys)

---

## Troubleshooting

### Common Issues

#### Issue: Lovable export is broken

**Solution**:
1. Re-export from Lovable
2. Check for syntax errors in exported code
3. Ensure all dependencies are installed
4. Update npm packages: `npm update`

#### Issue: API calls fail with CORS errors

**Solution**:
1. Configure CORS on backend:
   ```python
   # FastAPI backend
   from fastapi.middleware.cors import CORSMiddleware

   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.vercel.app"],
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

#### Issue: Build fails on Vercel

**Solution**:
1. Check build logs in Vercel dashboard
2. Ensure all environment variables are set
3. Test build locally: `npm run build`
4. Check Node.js version compatibility

#### Issue: Images not loading

**Solution**:
1. Check CORS headers on Supabase Storage
2. Verify public URL is correct
3. Use Next.js Image component for optimization
4. Check browser console for errors

#### Issue: Tests failing

**Solution**:
1. Run tests locally: `npm test`
2. Check for outdated snapshots: `npm test -- -u`
3. Ensure test environment is configured
4. Mock API calls properly

---

## Additional Resources

- [Lovable Documentation](https://docs.lovable.dev)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Vercel Documentation](https://vercel.com/docs)
- [Testing Library](https://testing-library.com)
- [Playwright](https://playwright.dev)

---

**Questions or issues?** Open an issue in the GitHub repository.
