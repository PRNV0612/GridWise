/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to the FastAPI service so the browser needs no CORS setup.
  async rewrites() {
    const api = process.env.API_URL || "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${api}/:path*` }];
  },
};

export default nextConfig;
