# Self-hosted LiveKit for Zindryx group calls

## 1. Rent a server
Any VPS with a public IP, 2 vCPU / 4 GB RAM is fine to start (Hetzner, DigitalOcean, Contabo...). Ubuntu 22.04+.
Pick a location close to most of your users.

## 2. Point a domain at it
Make an A record, e.g. `lk.yourdomain.com` -> the server's IP.

## 3. Open the firewall (these exact ports)
- 80/tcp and 443/tcp  (HTTPS + certificate)
- 7881/tcp            (media fallback)
- 3478/udp, 5349/tcp  (TURN relay)
- 50000-51000/udp     (media)

## 4. Install Docker, copy this folder, edit two files
```
curl -fsSL https://get.docker.com | sh
```
- `livekit.yaml`: change APIKEY_CHANGE_ME and SECRET_CHANGE_ME... (remember them) and `domain:`.
- `Caddyfile`: change `lk.yourdomain.com`.

## 5. Start
```
docker compose up -d
docker compose logs -f livekit     # should say it is listening
```

## 6. Tell the backend (Render -> Environment)
```
LIVEKIT_URL=wss://lk.yourdomain.com
LIVEKIT_API_KEY=<your key>
LIVEKIT_API_SECRET=<your secret>
```
Redeploy. Open `https://chadivavirtuals.name.ng/livekit/config` — it must show `{"enabled":true,...}`.
From then on every new group call uses the media server; if the server is ever down the app falls back to the old way.

## Notes
- One 2 vCPU box comfortably carries several 8-person video rooms or one 40-person room; watch CPU in `docker stats`.
- Voice-only rooms are very cheap.
- Updating: `docker compose pull && docker compose up -d`.
