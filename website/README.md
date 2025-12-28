# CryoFM Website

This website is built based on the [ricoui-portfolio](https://github.com/ricocc/ricoui-portfolio) template, a modern designer portfolio website template built with Astro.

## Development

### Start Development Server

```bash
npm run dev
# or
pnpm dev
```

The development server will start at `http://localhost:5200` (configured in `astro.config.mjs`).

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

### Adding Publications

Publications are stored in the `src/content/post/` directory. Each publication is a folder containing an `index.mdx` file.

### Steps to Add a New Publication

1. Create a new folder in `src/content/post/` with a descriptive name (e.g., `my-publication/`)

2. Create an `index.mdx` file inside the folder with the following frontmatter structure:

```mdx
---
title: "Your Publication Title"
publishDate: 2026-01-15
description: "A brief description of your publication"
read: 5  # Optional: estimated reading time in minutes
img: "/assets/blog/my-publication/cover.jpg"  # Optional: cover image path
img_alt: "Cover image description"  # Optional: alt text for the image
tags:  # Optional: array of tags
  - Tag1
  - Tag2
featured: false  # Optional: set to true to feature this publication
---

Your publication content in Markdown/MDX format...
```

3. Add any images or assets to `public/assets/blog/your-publication-name/`

4. The publication will automatically appear on the blog/publications page, sorted by publish date

### Local Development Testing

1. Start the development server:
   ```bash
   npm run dev
   ```

2. Navigate to `http://localhost:5200` in your browser

3. Test the following:
   - Homepage loads correctly
   - Publications page displays all publications
   - Individual publication pages render correctly
   - Navigation works
   - Responsive design on different screen sizes
   - Dark mode toggle (if applicable)
