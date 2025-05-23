docker compose down --volumes --remove-orphans
docker system prune -f
docker compose up --build
