import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import { fileURLToPath } from "url";
import path from "path";
import { readFileSync } from "fs";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Get the site URL from environment variables, or use the default value if not set
// Note: After the first deployment, be sure to set the correct PUBLIC_SITE_URL in the .env file
// GitHub Pages URL: https://bytedance-seed.github.io/cryofm/
const siteUrl = import.meta.env.PUBLIC_SITE_URL || 'https://bytedance-seed.github.io/cryofm/';

// Base path: use '/cryofm/' for production (GitHub Pages), '/' for development
// This is REQUIRED for GitHub Pages subdirectory deployment - without it, CSS/JS won't load
const basePath = import.meta.env.PUBLIC_BASE_PATH || (import.meta.env.PROD ? '/cryofm/' : '/');

// Load custom Shiki theme for better BibTeX visibility in dark mode
const customDarkTheme = JSON.parse(
  readFileSync(new URL('./shiki-theme-custom-dark.json', import.meta.url), 'utf-8')
);

// https://astro.build/config
export default defineConfig({
  site: siteUrl,
  base: basePath,
  envPrefix: 'PUBLIC_',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: customDarkTheme,
      },
      defaultColor: false, // Use CSS variables instead of inline styles for theme switching
      wrap: true,
    },
  },
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      }
    }
  },

  server: {
    port: 5200,
  },

  integrations: [mdx(), sitemap()],
});