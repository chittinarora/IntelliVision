#!/bin/bash

# =====================================
# IntelliVision Log Filter
# =====================================
# Filters logs to show only important events

# Function to filter important logs
filter_important_logs() {
    # Use a more efficient regex pattern including progress bars
    grep -E "(video|analytics|Progress|Job|processing|completed|GPU|memory|cuda|YOLO|detection|ERROR|error|security|auth|login|permission|access|Starting|Started|Stopping|Stopped|Container|Health|Ready|Connected|Disconnected|Migration|Migrations|Database|Redis|Qdrant|NVIDIA|CUDA|Tesla|P100|GPU|Memory|Health check|Service|Worker|Task|Queue|Broker|Backend|Connection|Timeout|Exception|Critical|Fatal|WARNING|WARN|🚀|✅|⚠️|❌|⚡|🌐|🗄️|🔴|🔍|🎯|📋|📊|🎬|🔒|🖥️|\[#+\-*\]|Progress.*%|Done:|Left:|Avg|FPS|Images/s|Batches/s)" || true
}

# If called directly, filter stdin
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    filter_important_logs
fi
