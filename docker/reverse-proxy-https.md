# Домен + HTTPS: главный сайт PavFGS (Streamlit) + Brush для iframe

**Главный сайт** — это приложение Streamlit [`src/interface/main_page.py`](../src/interface/main_page.py): его отдаём на **основном имени** домена (например `https://pavfgs.твой-домен.ru` или корень `https://твой-домен.ru`).

**Brush** (статика в Docker на nginx) — **отдельный поддомен** (например `https://brush.твой-домен.ru`), потому что:
- в страницу встроен **iframe** с URL Brush;
- WebGPU в браузере требует **HTTPS** и нормальный **secure context**;
- так проще, чем ужимать всё в один Streamlit с путём `/brush`.

Схема:

| DNS (A → IP VPS) | Куда прокси (на сервере) | Зачем |
|------------------|---------------------------|--------|
| `pavfgs` или `@` | `127.0.0.1:8501` | Streamlit — **основной сайт** |
| `brush` | `127.0.0.1:8080` | Docker **brush-web** — вьювер в iframe |

---

## 1. DNS

Две записи **A** (или одна на `@` и одна на `brush`):

- `pavfgs.твой-домен.ru` → `212.233.94.88` (или **`@`** / **`www`** если хочешь сайт прямо на корне домена)
- `brush.твой-домен.ru` → тот же IP

```bash
dig +short pavfgs.твой-домен.ru brush.твой-домен.ru
```

---

## 2. Порты и сервисы на VPS

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

- **Brush** (только localhost, наружу порт 8080 не обязателен):

```bash
docker rm -f brush-web 2>/dev/null
docker run -d --name brush-web --restart unless-stopped -p 127.0.0.1:8080:80 brush-web
```

- **Streamlit** — слушать **только localhost**, наружу пусть ходит только Caddy/nginx:

```bash
cd ~/vkr/PavFGS
# venv при необходимости: source .venv/bin/activate
export PAVFGS_BRUSH_URL="https://brush.твой-домен.ru"
streamlit run src/interface/main_page.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Долгий запуск: **systemd** unit, **screen/tmux**, или `nohup` — как удобнее (чтобы после перезагрузки VPS поднять снова).

---

## 3. Caddy — основной вариант

Установка (Ubuntu): см. блок установки в конце, или [официальная инструкция](https://caddyserver.com/docs/install#debian-ubuntu-raspbian).

**`/etc/caddy/Caddyfile`**: два сайта (замени домены — один под Streamlit, второй под Brush):

```caddyfile
# Главный сайт — PavFGS (Streamlit)
pavfgs.твой-домен.ru {
    reverse_proxy 127.0.0.1:8501
    encode gzip
}

# Вьювер для iframe + WebGPU
brush.твой-домен.ru {
    reverse_proxy 127.0.0.1:8080
    encode gzip
}
```

Если **корень домена** без поддомена `pavfgs`, первая строка блока: `твой-домен.ru {` с тем же `reverse_proxy` на 8501.

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Открой: **`https://pavfgs.твой-домен.ru`** (или твой хост) — должен открываться **интерфейс из `main_page.py`**.  
`PAVFGS_BRUSH_URL` должен совпадать с **`https://brush.твой-домен.ru`** (без слэша в конце или с ним — как задашь, лучше единообразно).

---

## 4. nginx + certbot (альтернатива)

Готовый пример для **`pavfgs.ru`**: [`nginx-pavfgs.example.conf`](nginx-pavfgs.example.conf) (Streamlit + Brush, `Upgrade` для WebSocket). Установка:

```bash
sudo cp docker/nginx-pavfgs.example.conf /etc/nginx/sites-available/pavfgs
sudo ln -sf /etc/nginx/sites-available/pavfgs /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d pavfgs.ru -d www.pavfgs.ru -d brush.pavfgs.ru
```

См. также [доку Streamlit за reverse proxy](https://docs.streamlit.io/knowledge-base/deploy/running-ssl-streamlit).

---

## 5. Проверки

- В браузере: главный URL → страница PavFGS; блок Brush грузит iframe с **https://brush.…** (в DevTools → Network не должно быть mixed content).
- WebGPU: чёрного экрана на **https** для Brush быть не должно; при проблемах — F12 → Console.

---

## 6. Безопасность

- Наружу по возможности только **80/443**; **8501** и **8080** только `127.0.0.1`.
- SSH по ключу; при необходимости `fail2ban`.

---

Пример готового Caddy: [`Caddyfile.example`](Caddyfile.example)
