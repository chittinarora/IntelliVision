#!/bin/bash

# =====================================
# IntelliVision Log Viewer
# =====================================
# Script to view logs from Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 IntelliVision Log Viewer${NC}"
echo "=================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  web              - View web container logs"
    echo "  celery           - View celery container logs"
    echo "  all              - View all container logs"
    echo "  files            - View log files from host"
    echo "  progress         - View video analytics progress logs"
    echo "  errors           - View error logs only"
    echo "  follow           - Follow logs in real-time"
    echo ""
    echo "Examples:"
    echo "  $0 web follow    - Follow web logs"
    echo "  $0 celery        - View celery logs"
    echo "  $0 progress      - View progress logs"
    echo "  $0 files         - View log files"
}

# Function to view container logs
view_container_logs() {
    local service=$1
    local follow=$2

    if [ "$follow" = "follow" ]; then
        echo -e "${GREEN}📋 Following $service logs (Ctrl+C to stop)...${NC}"
        docker-compose logs -f $service
    else
        echo -e "${GREEN}📋 Viewing $service logs...${NC}"
        docker-compose logs --tail=100 $service
    fi
}

# Function to view log files
view_log_files() {
    echo -e "${GREEN}📁 Viewing log files from host...${NC}"

    if [ ! -d "./logs" ]; then
        echo -e "${RED}❌ Logs directory not found. Make sure containers are running.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}📊 Available log files:${NC}"
    find ./logs -name "*.log" -type f | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  📄 $file ($size)"
    done

    echo ""
    echo -e "${YELLOW}📋 Recent log entries:${NC}"

    # Show recent entries from main log files
    for log_file in ./logs/django.log ./logs/celery/celery.log ./logs/performance/performance.log; do
        if [ -f "$log_file" ]; then
            echo -e "${BLUE}📄 $log_file (last 10 lines):${NC}"
            tail -10 "$log_file" | sed 's/^/  /'
            echo ""
        fi
    done
}

# Function to view progress logs
view_progress_logs() {
    echo -e "${GREEN}🎯 Viewing video analytics progress logs...${NC}"

    # View progress logs from celery container
    echo -e "${YELLOW}📊 Progress logs from Celery:${NC}"
    docker-compose logs --tail=50 celery | grep -E "(Progress|Job|processing|completed)" || echo "No progress logs found"

    # View progress logs from files
    if [ -f "./logs/celery/celery.log" ]; then
        echo -e "${YELLOW}📄 Progress logs from file:${NC}"
        grep -E "(Progress|Job|processing|completed)" ./logs/celery/celery.log | tail -20 || echo "No progress logs in file"
    fi
}

# Function to view error logs
view_error_logs() {
    echo -e "${GREEN}❌ Viewing error logs...${NC}"

    # View error logs from containers
    echo -e "${YELLOW}📊 Error logs from containers:${NC}"
    docker-compose logs --tail=50 | grep -i error || echo "No error logs found"

    # View error log files
    if [ -f "./logs/errors.log" ]; then
        echo -e "${YELLOW}📄 Error logs from file:${NC}"
        tail -20 ./logs/errors.log || echo "No error logs in file"
    fi
}

# Main script logic
case "${1:-help}" in
    "web")
        view_container_logs "web" "$2"
        ;;
    "celery")
        view_container_logs "celery" "$2"
        ;;
    "all")
        echo -e "${GREEN}📋 Viewing all container logs...${NC}"
        docker-compose logs --tail=50
        ;;
    "files")
        view_log_files
        ;;
    "progress")
        view_progress_logs
        ;;
    "errors")
        view_error_logs
        ;;
    "follow")
        echo -e "${GREEN}📋 Following all logs (Ctrl+C to stop)...${NC}"
        docker-compose logs -f
        ;;
    "help"|*)
        show_usage
        ;;
esac
