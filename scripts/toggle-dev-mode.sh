#!/bin/bash

# Function to display usage
show_usage() {
    echo "Usage: $0 [on|off]"
    echo "  on  - Enable local development mode (use localhost URLs)"
    echo "  off - Disable local development mode (use production URLs)"
    exit 1
}

# Check if argument is provided
if [ $# -ne 1 ]; then
    show_usage
fi

# Get the current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Function to update environment variables
update_env() {
    local env_file="$PROJECT_ROOT/.env"
    
    # Remove existing KANDC_BACKEND_URL and KANDC_FRONTEND_URL if they exist
    if [ -f "$env_file" ]; then
        sed -i.bak '/^KANDC_BACKEND_URL=/d' "$env_file"
        sed -i.bak '/^KANDC_FRONTEND_URL=/d' "$env_file"
        rm -f "${env_file}.bak"
    else
        touch "$env_file"
    fi
    
    # Add new values
    echo "$1" >> "$env_file"
    echo "$2" >> "$env_file"
}

case "$1" in
    "on")
        echo "🔧 Enabling local development mode..."
        update_env "KANDC_BACKEND_URL=http://localhost:8000" "KANDC_FRONTEND_URL=http://localhost:3000"
        echo "✅ Local development mode enabled"
        echo "Backend URL: http://localhost:8000"
        echo "Frontend URL: http://localhost:3000"
    ;;
    "off")
        echo "🔧 Disabling local development mode..."
        # Remove the environment variables to use default production URLs
        if [ -f "$PROJECT_ROOT/.env" ]; then
            sed -i.bak '/^KANDC_BACKEND_URL=/d' "$PROJECT_ROOT/.env"
            sed -i.bak '/^KANDC_FRONTEND_URL=/d' "$PROJECT_ROOT/.env"
            rm -f "${PROJECT_ROOT}/.env.bak"
        fi
        echo "✅ Local development mode disabled"
        echo "Using production URLs:"
        echo "Backend URL: https://api.keysandcaches.com"
        echo "Frontend URL: https://keysandcaches.com"
    ;;
    *)
        show_usage
    ;;
esac

echo "🎉 Done!"
