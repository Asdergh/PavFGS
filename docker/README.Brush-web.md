# Сборка и выкладка своей версии Brush

Наша версия Brush лежит в `brush/` и содержит ползунок времени для анимаций. Её можно собрать в контейнере и либо запускать в Docker, либо выложить статику на свой сайт (чтобы не использовать публичное демо).

## Папка `brush` со стрелочкой на GitHub = git submodule

**Стрелка** в интерфейсе GitHub значит: `brush` — **не** обычная вложенная папка, а **вложенный репозиторий (submodule)**. Родитель (PavFGS) хранит только **конкретный commit** из репо Brush, плюс URL в корневом `.gitmodules` (на `origin` он может отличаться от несинхронизированного локального файла).

- Обычный `git clone …PavFGS` **сам не подтянет** полное содержимое submodule, пока не выполнить:
  ```bash
  git submodule update --init --recursive
  ```
  либо клонировать сразу: `git clone --recurse-submodules …`.

- `modified: brush (modified content)` в `git status` в **корне** PavFGS значит: внутри `brush/` другое состояние, чем тот commit, на который ссылается родитель. Нужны **два уровня**:
  1. `cd brush` → `git status` → **commit** и **push** в ваш remote (форк);
  2. в корне PavFGS: `git add brush` → `git commit` → `git push` (родитель **запоминает новый** hash).

### Почему «не подтянулся мой» Brush

После `git pull` / `git clone` подставляется **тот** commit, который **залит** в репо PavFGS, а `brush` указывает на URL и hash из **submodule** — чаще всего upstream, пока вы не **запушили** коммиты в **свой** репо и не **обновили** указатель в PavFGS (шаг 2–3 ниже). Локальные незалитые правки **никуда сами** не мигрируют.

### Свой форк вместо публичного upstream

1. Создайте на GitHub **форк** [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) (или залейте **полный** клон в свой репозиторий: в корне обязан быть `Cargo.toml` и `crates/`).
2. В `PavFGS/brush` привяжите `origin` к форку и отправьте ветку с доработками:
   ```bash
   cd brush
   git remote -v
   # при необходимости: git remote set-url origin https://github.com/<вы>/brush.git
   git push -u origin main
   ```
3. В **корне** PavFGS зафиксируйте новый commit submodule:
   ```bash
   cd /path/to/PavFGS
   git add brush
   git commit -m "chore: update brush"
   git push
   ```
4. **Сервер:** `git pull` в PavFGS, затем `git submodule update --init --recursive`, потом `docker build …`

Если в `.gitmodules` ещё записан `ArthurBrussee/brush`, смена на форк:

```bash
cd /path/to/PavFGS
git config -f .gitmodules submodule.brush.url https://github.com/<вы>/brush.git
git submodule sync
git submodule update --init --remote brush
git add .gitmodules brush
git commit -m "chore: point brush submodule to fork"
git push
```

Команда `rm -rf brush && git clone` публичного репо **ломает** привязку submodule в git — надежнее: не удалять `brush` вручную, а обновлять его через `git submodule` (или вручную клон в форк и `git add brush` с нужным commit).

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

## Что должно лежать в `brush/`

Нужен **полный** репозиторий [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush): в корне `brush/Cargo.toml`, папка `brush/crates/brush-app/` с `Cargo.toml`, плюс `brush/brush_nextjs/`.

Проверка:

```bash
test -f brush/Cargo.toml && test -f brush/crates/brush-app/Cargo.toml && echo OK
```

Если `brush/` содержит только `brush_nextjs/` (без Rust), `wasm-pack` **не соберётся** — клонируй весь Brush (или подтяни submodule), затем пересобери образ.

### Ошибка `brush/ must be the repo root with workspace Cargo.toml`

На хосте в **корне PavFGS** сейчас **нет** файла `brush/Cargo.toml`. Самый прямой фикс — заменить каталог целиком:

```bash
cd ~/vkr/PavFGS
rm -rf brush
git clone --depth 1 https://github.com/ArthurBrussee/brush.git brush
test -f brush/Cargo.toml && test -f brush/crates/brush-app/Cargo.toml && echo "OK, можно docker build"
```

Если у вас **свой форк** с доработками, клонируйте **его** в `brush/`, а не обрезанную копию. После `git pull` в форке снова проверьте `test -f brush/Cargo.toml`.

### Публичный клон затёр локальные правки

Команда `rm -rf brush && git clone … ArthurBrussee/brush` подменяет каталог **целиком**. Восстановление:

- если доработки были **в коммитах** на другой машине / в форке — `git clone <url-вашего-репо> brush` или залейте ветку на GitHub и стяните её;
- если правки были **только локально и без бэкапа** — откат только из резервной копии или из истории IDE/сервера;
- дальше держите **полный** репозиторий Brush (как у upstream) и свои изменения в **ветке** или **форке**, чтобы `docker build` всегда видел `Cargo.toml` + `crates/` + ваши правки.

## Сборка образа (Docker)

Из **корня репозитория** (PavFGS):

```bash
docker build -f docker/Brush.web.Dockerfile -t brush-web .
```

Сборка может занять несколько минут (Rust WASM + Next.js). `DOCKER_BUILDKIT=1` не обязателен: если появляется *buildx component is missing*, либо установи `docker-buildx-plugin` (`apt install docker-buildx-plugin`), либо собери **без** `DOCKER_BUILDKIT=1` — обычного `docker build` достаточно.

**Образ сам по себе контейнер не запускает.** После `Successfully tagged brush-web:latest` нужно явно стартовать:

```bash
docker rm -f brush-web 2>/dev/null   # если имя уже занято старым контейнером
docker run -d --name brush-web --restart unless-stopped -p 127.0.0.1:8080:80 brush-web
docker ps   # должен появиться brush-web
```

Проверка: `curl -sI http://127.0.0.1:8080 | head -1` — ожидается `200` или `301/302`.

## Вариант 1: Запуск Brush в контейнере

```bash
docker run -d --name brush-web --restart unless-stopped -p 127.0.0.1:8080:80 brush-web
```

Для доступа с другой машины открой порт на интерфейсе, например `-p 8080:80` (без `127.0.0.1:`) и настрой firewall.

Открой в браузере: **http://<сервер>:8080**  
В Streamlit в поле «Brush URL» укажи тот же URL (или задай `PAVFGS_BRUSH_URL`).

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

## Домен + HTTPS (WebGPU, «по красоте»)

**Основной сайт** — Streamlit [`src/interface/main_page.py`](../src/interface/main_page.py) на своём хосте (порт 8501 за reverse proxy); **Brush** — отдельный поддомен. Подробно: **[reverse-proxy-https.md](reverse-proxy-https.md)**, пример Caddy: **[Caddyfile.example](Caddyfile.example)**.
