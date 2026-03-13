#!/bin/bash
# =============================================================================
# Omni-Help — Nginx + Certbot deployment script
# =============================================================================
# Installs Nginx, copies the reverse proxy config, and prints the Certbot
# command needed to finish HTTPS setup.
#
# Run from the repo root with:
#   sudo bash scripts/deploy_ssl.sh
#
# Prerequisites:
#   - Docker Compose containers must be running (port 8501 must be listening)
#   - saving.fit DNS A record must point to this server's public IP
#   - Port 80 and 443 must be open in the EC2 security group
# =============================================================================

set -e  # Exit immediately on any error

NGINX_CONF_SRC="infrastructure/nginx/omnihelp.conf"
NGINX_CONF_DEST="/etc/nginx/sites-available/omnihelp"
NGINX_ENABLED="/etc/nginx/sites-enabled/omnihelp"
DOMAIN="saving.fit"
WWW_DOMAIN="www.saving.fit"

# -----------------------------------------------------------------------------
# 1. Install Nginx and Certbot
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Installing Nginx and Certbot..."
echo "============================================================"
apt-get update -q
apt-get install -y nginx certbot python3-certbot-nginx

# -----------------------------------------------------------------------------
# 2. Deploy Nginx configuration
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Copying Nginx configuration..."
echo "============================================================"

# Ensure script is run from repo root
if [ ! -f "$NGINX_CONF_SRC" ]; then
    echo ""
    echo "  ERROR: $NGINX_CONF_SRC not found."
    echo "  Run this script from the repository root:"
    echo "    cd ~/omnihelp && sudo bash scripts/deploy_ssl.sh"
    exit 1
fi

cp "$NGINX_CONF_SRC" "$NGINX_CONF_DEST"
echo "  Copied $NGINX_CONF_SRC -> $NGINX_CONF_DEST"

# Create symlink to enable the site
ln -sf "$NGINX_CONF_DEST" "$NGINX_ENABLED"
echo "  Symlink created: $NGINX_ENABLED"

# Remove default site to avoid conflicts
rm -f /etc/nginx/sites-enabled/default
echo "  Removed default Nginx site."

# -----------------------------------------------------------------------------
# 3. Validate and restart Nginx
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Testing Nginx configuration..."
echo "============================================================"
nginx -t

echo ""
echo "  Restarting Nginx..."
systemctl restart nginx
systemctl enable nginx
echo "  Nginx is running and enabled on boot."

# -----------------------------------------------------------------------------
# 4. Print the Certbot command to finish SSL setup
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo ""
echo "  Nginx is live at http://$DOMAIN"
echo ""
echo "  ACTION REQUIRED — Run this command to enable HTTPS:"
echo ""
echo "    certbot --nginx -d $DOMAIN -d $WWW_DOMAIN"
echo ""
echo "  Certbot will:"
echo "    1. Verify domain ownership (requires DNS to be propagated)"
echo "    2. Obtain a Let's Encrypt SSL certificate"
echo "    3. Automatically update this Nginx config for HTTPS"
echo "    4. Set up auto-renewal via a systemd timer"
echo ""
echo "  After Certbot completes, your app will be live at:"
echo "    https://$DOMAIN"
echo ""
echo "============================================================"
echo ""
