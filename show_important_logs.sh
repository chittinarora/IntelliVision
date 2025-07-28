#!/bin/bash

# =====================================
# IntelliVision Important Logs Viewer
# =====================================
# Shows only important logs (video analytics, errors, security)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 IntelliVision Important Logs${NC}"
echo "====================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  video           - Show only video analytics logs"
    echo "  errors          - Show only error logs"
    echo "  security        - Show only security logs"
    echo "  system          - Show only system events (containers, health, etc.)"
    echo "  important       - Show all important logs (default)"
    echo "  follow          - Follow logs in real-time"
    echo ""
    echo "Examples:"
    echo "  $0 video follow    - Follow video analytics logs"
    echo "  $0 system          - Show system events"
    echo "  $0 important       - Show all important logs"
    echo "  $0 errors          - Show error logs"
}

# Function to show video analytics logs
show_video_logs() {
    local follow=$1
    echo -e "${GREEN}🎬 Video Analytics Logs${NC}"

    if [ "$follow" = "follow" ]; then
        echo "📋 Following video analytics logs (Ctrl+C to stop)..."
        docker-compose logs -f | grep -E "(video|analytics|Progress|Job|processing|completed|GPU|memory|cuda|YOLO|detection|\[#+\-*\]|Progress.*%|Done:|Left:|Avg|FPS|Images/s|Batches/s)" || true
    else
        echo "📋 Recent video analytics logs:"
        docker-compose logs --tail=100 | grep -E "(video|analytics|Progress|Job|processing|completed|GPU|memory|cuda|YOLO|detection|\[#+\-*\]|Progress.*%|Done:|Left:|Avg|FPS|Images/s|Batches/s)" || true
    fi
}

# Function to show error logs
show_error_logs() {
    local follow=$1
    echo -e "${RED}❌ Error Logs${NC}"

    if [ "$follow" = "follow" ]; then
        echo "📋 Following error logs (Ctrl+C to stop)..."
        docker-compose logs -f | grep -i error || true
    else
        echo "📋 Recent error logs:"
        docker-compose logs --tail=100 | grep -i error || true
    fi
}

# Function to show security logs
show_security_logs() {
    local follow=$1
    echo -e "${YELLOW}🔒 Security Logs${NC}"

    if [ "$follow" = "follow" ]; then
        echo "📋 Following security logs (Ctrl+C to stop)..."
        docker-compose logs -f | grep -E "(security|auth|login|permission|access)" || true
    else
        echo "📋 Recent security logs:"
        docker-compose logs --tail=100 | grep -E "(security|auth|login|permission|access)" || true
    fi
}

# Function to show system events
show_system_logs() {
    local follow=$1
    echo -e "${BLUE}🖥️  System Events${NC}"

    if [ "$follow" = "follow" ]; then
        echo "📋 Following system events (Ctrl+C to stop)..."
        docker-compose logs -f | grep -E "(Starting|Started|Stopping|Stopped|Container|Health|Ready|Connected|Disconnected|Migration|Migrations|Database|Redis|Qdrant|NVIDIA|CUDA|Tesla|P100|GPU|Memory|Health check|Service|Worker|Task|Queue|Broker|Backend|Connection|Timeout|Exception|Critical|Fatal|WARNING|WARN)" || true
    else
        echo "📋 Recent system events:"
        docker-compose logs --tail=100 | grep -E "(Starting|Started|Stopping|Stopped|Container|Health|Ready|Connected|Disconnected|Migration|Migrations|Database|Redis|Qdrant|NVIDIA|CUDA|Tesla|P100|GPU|Memory|Health check|Service|Worker|Task|Queue|Broker|Backend|Connection|Timeout|Exception|Critical|Fatal|WARNING|WARN)" || true
    fi
}

# Function to show all important logs
show_important_logs() {
    local follow=$1
    echo -e "${GREEN}🎯 All Important Logs${NC}"

    if [ "$follow" = "follow" ]; then
        echo "📋 Following important logs (Ctrl+C to stop)..."
        docker-compose logs -f | grep -E "(video|analytics|Progress|Job|processing|completed|GPU|memory|cuda|YOLO|detection|ERROR|error|security|auth|login|permission|access|Starting|Started|Stopping|Stopped|Container|Health|Ready|Connected|Disconnected|Migration|Migrations|Database|Redis|Qdrant|NVIDIA|CUDA|Tesla|P100|GPU|Memory|Health check|Service|Worker|Task|Queue|Broker|Backend|Connection|Timeout|Exception|Critical|Fatal|WARNING|WARN|\[#+\-*\]|Progress.*%|Done:|Left:|Avg|FPS|Images/s|Batches/s)" || true
    else
        echo "📋 Recent important logs:"
        docker-compose logs --tail=100 | grep -E "(video|analytics|Progress|Job|processing|completed|GPU|memory|cuda|YOLO|detection|ERROR|error|security|auth|login|permission|access|Starting|Started|Stopping|Stopped|Container|Health|Ready|Connected|Disconnected|Migration|Migrations|Database|Redis|Qdrant|NVIDIA|CUDA|Tesla|P100|GPU|Memory|Health check|Service|Worker|Task|Queue|Broker|Backend|Connection|Timeout|Exception|Critical|Fatal|WARNING|WARN|\[#+\-*\]|Progress.*%|Done:|Left:|Avg|FPS|Images/s|Batches/s)" || true
    fi
}

# Main script logic
case "${1:-important}" in
    "video")
        show_video_logs "$2"
        ;;
    "errors")
        show_error_logs "$2"
        ;;
    "security")
        show_security_logs "$2"
        ;;
    "system")
        show_system_logs "$2"
        ;;
    "important")
        show_important_logs "$2"
        ;;
    "follow")
        show_important_logs "follow"
        ;;
    "help"|*)
        show_usage
        ;;
esac
