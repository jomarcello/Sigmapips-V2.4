#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Setting up Railway integration..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}Railway CLI not found. Installing...${NC}"
    curl -fsSL https://railway.app/install.sh | sh
fi

# Check if logged in to Railway
if ! railway whoami &> /dev/null; then
    echo -e "${RED}Not logged in to Railway. Please login:${NC}"
    railway login
fi

# Link the project to Railway
echo -e "${GREEN}Linking project to Railway...${NC}"
railway link

# Set up environment variables
echo -e "${GREEN}Setting up environment variables...${NC}"
railway run env | grep -v "^#" > .env

# Update Railway configuration
echo -e "${GREEN}Updating Railway configuration...${NC}"
cat > railway.json << EOL
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "/app/start.sh",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "startupProbe": {
      "path": "/health",
      "initialDelaySeconds": 60,
      "periodSeconds": 20,
      "timeoutSeconds": 10,
      "successThreshold": 1,
      "failureThreshold": 15
    },
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOL

# Make the script executable
chmod +x start.sh

echo -e "${GREEN}Railway setup complete!${NC}"
echo -e "${GREEN}You can now use the following commands:${NC}"
echo "  railway up    - Deploy your application"
echo "  railway logs  - View application logs"
echo "  railway vars  - Manage environment variables" 