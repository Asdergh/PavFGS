# Сборка Brush (WASM + Next static export) для nginx.
#
#   docker build -f docker/Brush.web.Dockerfile -t brush-web .
#   docker run -d --name brush-web --restart unless-stopped -p 127.0.0.1:8080:80 brush-web
#
# Требование: в репозитории должен быть ПОЛНЫЙ каталог brush/ из
# https://github.com/ArthurBrussee/brush (все crates/, Cargo.toml в корне brush/).
# Только brush_nextjs/ без Rust — сборка WASM невозможна.

FROM rust:1.92-bookworm AS wasm-builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack

WORKDIR /brush
COPY brush/ .

# В build context каталог brush/ должен быть КОРНЕМ репозитория Brush (см. README: git clone … в brush/)
RUN test -f Cargo.toml || ( \
    echo "ERROR: в PavFGS/brush/ нет Cargo.toml — положи сюда полный клон https://github.com/ArthurBrussee/brush" >&2; \
    echo "Содержимое сейчас (первый уровень):" >&2; ls -la >&2; \
    echo "Поиск Cargo.toml (если пусто — файлов нет):" >&2; find . -maxdepth 3 -name Cargo.toml 2>/dev/null | head -30 >&2; \
    exit 1)
RUN test -f crates/brush-app/Cargo.toml || ( \
    echo "ERROR: нет crates/brush-app/Cargo.toml — не полный воркспейс Rust." >&2; exit 1)

# Собираем из корня воркспейса (как в npm script build:wasm-release)
RUN wasm-pack build crates/brush-app --release --target bundler --out-dir brush_nextjs/pkg

FROM node:22-bookworm-slim AS next-builder
WORKDIR /app

COPY brush/brush_nextjs/package.json brush/brush_nextjs/package-lock.json* ./
RUN npm install

COPY brush/brush_nextjs/ ./
COPY --from=wasm-builder /brush/brush_nextjs/pkg ./pkg

ENV NEXT_TELEMETRY_DISABLED=1
# next.config.js в upstream с output: 'export' → каталог out/
# Без basePath — отдаём с корня nginx (для подпути задайте NEXT_PUBLIC_BASE_PATH при сборке)
RUN npx cross-env NEXT_PUBLIC_BASE_PATH= next build --turbopack

FROM nginx:alpine AS runtime
COPY --from=next-builder /app/out /usr/share/nginx/html
RUN echo 'server { root /usr/share/nginx/html; location / { try_files $uri $uri/ /index.html; } }' > /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
