#!/bin/bash
# Production deployment script for Hunter Trading Bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
DOCKER_COMPOSE_FILE="docker-compose.yml"

echo -e "${BLUE}🚀 Hunter Trading Bot Deployment Script${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Version: ${VERSION}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose is installed"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi
print_status "Python 3 is installed"

# Check if required files exist
required_files=("config.yaml" "requirements.txt" "Dockerfile" "docker-compose.yml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file $file not found."
        exit 1
    fi
done
print_status "All required files found"

# Check if .env file exists
if [ ! -f "system/.env" ]; then
    print_warning ".env file not found. Creating from example..."
    if [ -f "system/.env.example" ]; then
        cp system/.env.example system/.env
        print_warning "Please edit system/.env with your configuration before continuing."
        read -p "Press Enter to continue after editing .env file..."
    else
        print_error ".env.example file not found. Please create system/.env manually."
        exit 1
    fi
fi
print_status "Environment configuration found"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p data logs cache system
print_status "Directories created"

# Set permissions
echo -e "${BLUE}🔐 Setting permissions...${NC}"
chmod 755 scripts/*.sh
chmod 644 config.yaml
chmod 600 system/.env
print_status "Permissions set"

# Install Python dependencies locally (for development)
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
python3 -m pip install --user -r requirements.txt
print_status "Python dependencies installed"

# Build Docker image
echo -e "${BLUE}🐳 Building Docker image...${NC}"
docker build -t hunter-trading-bot:$VERSION .
print_status "Docker image built"

# Stop existing containers
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
docker-compose -f $DOCKER_COMPOSE_FILE down || true
print_status "Existing containers stopped"

# Start new containers
echo -e "${BLUE}🚀 Starting new containers...${NC}"
docker-compose -f $DOCKER_COMPOSE_FILE up -d
print_status "Containers started"

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 30

# Check if services are running
echo -e "${BLUE}🔍 Checking service health...${NC}"

# Check main application
if curl -f http://localhost:8765/health &> /dev/null; then
    print_status "Main application is healthy"
else
    print_warning "Main application health check failed"
fi

# Check Redis (if enabled)
if docker ps | grep -q hunter-redis; then
    if docker exec hunter-redis redis-cli ping &> /dev/null; then
        print_status "Redis is healthy"
    else
        print_warning "Redis health check failed"
    fi
fi

# Check Prometheus (if enabled)
if docker ps | grep -q hunter-prometheus; then
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        print_status "Prometheus is healthy"
    else
        print_warning "Prometheus health check failed"
    fi
fi

# Check Grafana (if enabled)
if docker ps | grep -q hunter-grafana; then
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        print_status "Grafana is healthy"
    else
        print_warning "Grafana health check failed"
    fi
fi

# Display service URLs
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Service URLs:${NC}"
echo -e "  • Real-time Dashboard: http://localhost:8765"
if docker ps | grep -q hunter-prometheus; then
    echo -e "  • Prometheus: http://localhost:9090"
fi
if docker ps | grep -q hunter-grafana; then
    echo -e "  • Grafana: http://localhost:3000 (admin/admin)"
fi
echo ""
echo -e "${BLUE}📋 Useful commands:${NC}"
echo -e "  • View logs: docker-compose logs -f"
echo -e "  • Stop services: docker-compose down"
echo -e "  • Restart services: docker-compose restart"
echo -e "  • Update services: ./scripts/deploy.sh $ENVIRONMENT $VERSION"
echo ""
echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo -e "  • Configure your API keys in system/.env"
echo -e "  • Set up proper monitoring and alerting"
echo -e "  • Review security settings for production"
echo ""
