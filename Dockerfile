# Use official Node.js runtime as a parent image
FROM node:20

# Set working directory
WORKDIR /usr/src/app

# Install pnpm
RUN npm install -g pnpm

# Copy package.json and pnpm-lock.yaml
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Remove overriding ENV PORT and set COGNITIVE_MESH_PORT
# ENV PORT=8080
ENV COGNITIVE_MESH_PORT=8080

# Copy the rest of the application code
COPY . .

# Use curl for HEALTHCHECK
HEALTHCHECK CMD curl --fail http://localhost:${COGNITIVE_MESH_PORT}/health || exit 1

# Command to run the application
CMD ["node", "openclaw-gateway.js"]