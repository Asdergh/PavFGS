#   docker build -f docker/Brush.web.Dockerfile -t brush-web .
#   docker run --rm -v "$(pwd)/brush-out:/out" brush-web
# Содержимое brush-out/ выложить на любой статический хостинг

# сборка WASM 
FROM rust:1.92-bookworm AS wasm-builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack

WORKDIR /brush
COPY brush/ .

WORKDIR /brush/crates/brush-app
RUN wasm-pack build --release --target bundler --out-dir ../../brush_nextjs/pkg

# сборка Next.js статики
FROM node:22-bookworm-slim AS next-builder
WORKDIR /app

COPY brush/brush_nextjs/package.json brush/brush_nextjs/package-lock.json* ./
RUN npm install

COPY brush/brush_nextjs/ ./
COPY --from=wasm-builder /brush/brush_nextjs/pkg ./pkg

RUN npm run build:static-only

# образ для отдачи статики 
FROM nginx:alpine AS runtime
COPY --from=next-builder /app/out /usr/share/nginx/html
RUN echo 'server { root /usr/share/nginx/html; location / { try_files $uri $uri/ /index.html; } }' > /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
