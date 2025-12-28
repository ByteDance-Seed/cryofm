// 1. Import utility functions from `astro:content`
import { defineCollection, z } from 'astro:content';

// 2. Import loader
import { glob } from 'astro/loaders';

// 3. Define your collection
const post = defineCollection({
	// Use glob loader to load all .mdx files from src/content/post directory
	loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/post' }),
	// Type-check frontmatter using a schema
	schema: z.object({
		title: z.string(),
		description: z.string(),
		publishDate: z.coerce.date(),
		read: z.number().optional(),
		tags: z.array(z.string()).optional(),
		img: z.string().optional(),
		img_alt: z.string().optional(),
		featured: z.boolean().optional(),
	}),
});

// 4. Export a `collections` object to register your collection
export const collections = { post };

