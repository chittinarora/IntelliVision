#!/bin/bash

# Exit on any error
set -e

echo "Setting up VM environment for IntelliVision..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    netstat -tuln | grep ":$1 " >/dev/null 2>&1
}

# Update package list
echo "Updating package list..."
apt-get update

# Install netstat if not present
apt-get install -y net-tools

# Check what's using port 80
if port_in_use 80; then
    echo "Port 80 is already in use. Checking what's using it..."
    netstat -tulpn | grep ":80 "
    echo "Would you like to stop the process using port 80? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        # Try to stop Apache if it's running
        systemctl stop apache2 2>/dev/null || true
        # If something else is using the port, we need to handle it manually
        echo "Please stop the process using port 80 manually and run this script again."
        exit 1
    else
        echo "Please free up port 80 and run this script again."
        exit 1
    fi
fi

# Install Apache and WSGI module
echo "Installing Apache and mod_wsgi..."
apt-get install -y apache2 libapache2-mod-wsgi-py3

# Configure Apache to use port 8080 instead of 80 if needed
if [ -f "/etc/apache2/ports.conf" ]; then
    sed -i 's/Listen 80/Listen 8080/' /etc/apache2/ports.conf
    echo "Configured Apache to listen on port 8080"
fi

# Install Redis
echo "Installing Redis..."
apt-get install -y redis-server
systemctl enable redis-server
systemctl start redis-server

# Install Redis CLI
echo "Installing Redis CLI..."
apt-get install -y redis-tools

# Install pip if not present
apt-get install -y python3-pip

# Install Celery
echo "Installing Celery..."
pip3 install celery redis

# Set up Celery service
echo "Setting up Celery service..."
if [ ! -f "/etc/systemd/system/celery.service" ]; then
    cat > /etc/systemd/system/celery.service << EOL
[Unit]
Description=Celery Service
After=network.target redis.service

[Service]
Type=simple
User=www-data
Group=www-data
EnvironmentFile=/etc/default/celeryd
WorkingDirectory=/var/www/intellivision
ExecStart=/usr/local/bin/celery -A intellivision worker --loglevel=info
Restart=always

[Install]
WantedBy=multi-user.target
EOL

    # Create celery environment file
    cat > /etc/default/celeryd << EOL
# Names of nodes to start
CELERYD_NODES="worker1"

# Absolute or relative path to the 'celery' command:
CELERY_BIN="/usr/local/bin/celery"

# App instance to use
CELERY_APP="intellivision"

# Where to chdir at start.
CELERYD_CHDIR="/var/www/intellivision"

# Extra command-line arguments to the worker
CELERYD_OPTS="--time-limit=300 --concurrency=8"

# %n will be replaced with the first part of the nodename.
CELERYD_LOG_FILE="/var/log/intellivision/celery/%n%I.log"
CELERYD_PID_FILE="/var/run/celery/%n.pid"

# Workers should run as an unprivileged user.
CELERYD_USER="www-data"
CELERYD_GROUP="www-data"

# If enabled pid and log directories will be created if missing,
# and owned by the userid/group configured.
CELERY_CREATE_DIRS=1
EOL

    # Create required directories for Celery
    mkdir -p /var/run/celery
    mkdir -p /var/log/intellivision/celery
    chown -R www-data:www-data /var/run/celery
    chown -R www-data:www-data /var/log/intellivision/celery

    systemctl daemon-reload
fi

# Create necessary directories with correct permissions
echo "Creating necessary directories..."
mkdir -p /var/log/intellivision/{api,celery,security,performance}
mkdir -p /var/www/intellivision/media/outputs

# Set correct permissions
echo "Setting permissions..."
chown -R www-data:www-data /var/log/intellivision
chown -R www-data:www-data /var/www/intellivision
chmod -R 755 /var/log/intellivision
chmod -R 755 /var/www/intellivision

# Enable required Apache modules
echo "Enabling Apache modules..."
a2enmod wsgi
a2enmod ssl
a2enmod rewrite

# Restart services
echo "Restarting services..."
systemctl restart apache2 || {
    echo "Apache failed to start. Check error logs:"
    cat /var/log/apache2/error.log
}

systemctl restart redis-server || {
    echo "Redis failed to start. Check error logs:"
    journalctl -u redis-server
}

systemctl restart celery || {
    echo "Celery failed to start. Check error logs:"
    journalctl -u celery
}

echo "Environment setup complete!"
echo
echo "Service Status:"
echo "=============="
systemctl status apache2 --no-pager
echo
systemctl status redis-server --no-pager
echo
systemctl status celery --no-pager
echo
echo "You can now run the tests using:"
echo "sudo python3 test_in_vm.py"
echo
echo "Note: Apache is configured to run on port 8080 instead of 80"
echo "Make sure your application's configuration matches this port number."
