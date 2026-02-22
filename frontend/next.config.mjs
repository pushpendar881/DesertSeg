/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Allow cross-origin requests in development (for accessing from other devices on network)
  allowedDevOrigins: [
    '172.31.214.51', // Your local network IP
    'localhost',
    '127.0.0.1',
  ],
}

export default nextConfig
