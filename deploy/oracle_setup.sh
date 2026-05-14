#!/bin/bash
# ============================================================
# AegisQuant — Oracle Cloud Free Tier Setup Script
# Run this ONCE on a fresh Ubuntu 22.04 ARM instance.
# Usage: bash oracle_setup.sh
# ============================================================
set -e

echo "=== [1/7] System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git curl wget htop unzip ufw fail2ban

echo "=== [2/7] Install Docker ==="
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

echo "=== [3/7] Install Docker Compose ==="
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "=== [4/7] Firewall — allow SSH + dashboard port ==="
sudo ufw allow OpenSSH
sudo ufw allow 8000/tcp   # AegisQuant dashboard
sudo ufw --force enable

echo "=== [5/7] Clone AegisQuant ==="
cd ~
git clone https://github.com/sriv144/AegisQuant.git
cd AegisQuant

echo "=== [6/7] Create .env from template ==="
cp .env.example .env
echo ""
echo ">>> IMPORTANT: Edit .env with your Alpaca keys before continuing:"
echo "    nano .env"
echo ""
echo "    Set:"
echo "      MARKET=US"
echo "      BROKER=alpaca"
echo "      ALPACA_API_KEY=your_key"
echo "      ALPACA_SECRET_KEY=your_secret"
echo "      ALPACA_BASE_URL=https://paper-api.alpaca.markets"
echo "      INITIAL_CAPITAL=100000"
echo "      ENABLE_MOCK_DATA=False"
echo "      AEGIS_PASSWORD=choose_a_strong_password"
echo ""
read -p "Press ENTER once you've saved .env to continue..."

echo "=== [7/7] Build and start AegisQuant ==="
# Apply new group membership without re-login
newgrp docker << DOCKEREOF
docker-compose build
docker-compose up -d
DOCKEREOF

echo ""
echo "=== Setup complete! ==="
echo "Trader daemon:  docker-compose logs -f trader"
echo "Dashboard:      http://$(curl -s ifconfig.me):8000"
echo ""
echo "Useful commands:"
echo "  docker-compose ps              # check both services"
echo "  docker-compose logs -f trader  # watch live trading logs"
echo "  docker-compose restart trader  # restart if stuck"
echo "  docker-compose pull && docker-compose up -d  # update to latest"
