# Сборка и выкладка своей версии Brush

Наша версия Brush лежит в `brush/` и содержит ползунок времени для анимаций. Её можно собрать в контейнере и либо запускать в Docker, либо выложить статику на свой сайт (чтобы не использовать публичное демо).

## Локальный запуск (без Docker) — для разработки

Чтобы **не пересобирать образ** при каждом изменении в `brush/`, запускай Brush локально:

**Требования:** [Rust](https://rustup.rs/) (1.92+), Node.js 20+. Плюс wasm-pack:
`cargo install wasm-pack`

```bash
# Из корня репозитория (PavFGS). Если не запускается: chmod +x scripts/run-brush-local.sh
./scripts/run-brush-local.sh
```

Или вручную:

```bash
cd brush/brush_nextjs
npm install
npm run dev
```

Brush откроется на **http://localhost:3000**. В приложении Streamlit в поле «Brush URL» укажи `http://localhost:3000`. После правок в коде Rust (`brush/crates/...`) перезапусти `npm run dev` — он заново соберёт WASM и поднимет сервер. Docker для итерации не нужен.

## Сборка образа (Docker)

Из **корня репозитория** (PavFGS):

```bash
docker build -f docker/Brush.web.Dockerfile -t brush-web .
```

Сборка может занять несколько минут (Rust WASM + Next.js).

## Вариант 1: Запуск Brush в контейнере

```bash
docker run -p 8080:80 brush-web
```

Открой в браузере: **http://localhost:8080**  
В приложении в поле «Brush URL» укажи: `http://localhost:8080`

## Вариант 2: Скопировать статику на свой сайт

Получить папку со статикой (например, для GitHub Pages, Netlify или своего сервера):

```bash
docker create --name brush-tmp brush-web
mkdir -p brush-out
docker cp brush-tmp:/usr/share/nginx/html/. brush-out/
docker rm brush-tmp
```

Дальше выложи содержимое **brush-out/** на любой статический хостинг:

- **GitHub Pages**: залей в ветку `gh-pages` или через Actions в папку `brush-demo/`.
- **Netlify / Vercel**: укажи каталог `brush-out` как корень сайта.
- **Свой сервер**: скопируй `brush-out/` в каталог nginx/apache.

В приложении в поле «Brush URL» укажи URL этой выкладки (например `https://твой-сайт.ru/brush-demo/`).

## Переменная окружения в приложении

Чтобы по умолчанию подставлять свою выкладку Brush, можно задать URL при запуске Streamlit:

```bash
export PAVFGS_BRUSH_URL="https://твой-сайт.ru/brush-demo/"
streamlit run src/interface/main_page.py
```

В коде можно использовать `os.environ.get("PAVFGS_BRUSH_URL", BRUSH_VIEWER_URL)` как значение по умолчанию для поля Brush URL.
