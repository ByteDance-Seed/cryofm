/**
 * Get the base URL from environment
 * Safely handles cases where BASE_URL might not be available during build
 */
export function getBaseUrl(): string {
	try {
		// In Astro, BASE_URL is always available, but we add a fallback for safety
		const base = import.meta.env.BASE_URL;
		return base || '/';
	} catch {
		return '/';
	}
}

/**
 * Resolve a path relative to the base URL
 * @param path - The path to resolve (should start with /)
 * @returns The resolved path with base URL
 */
export function resolveUrl(path: string): string {
	// Normalize input path - ensure it starts with /
	const normalizedPath = path.startsWith('/') ? path : `/${path}`;
	
	// Get base URL
	const base = getBaseUrl();
	
	// In development, base is '/', so return path as-is
	if (base === '/' || !base) {
		return normalizedPath;
	}
	
	// In production, combine base and path
	// Remove trailing slash from base if present, but keep at least one slash
	const basePath = base.endsWith('/') ? base.slice(0, -1) : base;
	
	// Combine: basePath should not be empty at this point
	return basePath ? `${basePath}${normalizedPath}` : normalizedPath;
}

