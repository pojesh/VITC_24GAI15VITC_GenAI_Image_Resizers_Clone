# Dockerfile for Next.js application

# --- Builder Stage ---
# This stage installs dependencies, builds the app, and prunes dev dependencies.
FROM node:20-slim AS builder

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install all dependencies (including devDependencies for the build)
RUN npm install --include=dev

# Copy the rest of the application source code
COPY . .

# Build the Next.js application
RUN npm run build

# Prune development dependencies
RUN npm prune --omit=dev


# --- Runner Stage ---
# This stage creates the final, lightweight production image.
FROM node:20-slim AS runner

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files from the builder stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Switch to the non-root user
USER nextjs

# Expose the port the app runs on
EXPOSE 3000

# Disable Next.js telemetry
ENV NEXT_TELEMETRY_DISABLED 1

# Start the Next.js app
CMD ["npm", "start"]
