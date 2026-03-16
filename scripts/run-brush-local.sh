set -e
cd "$(dirname "$0")/.."

if ! command -v wasm-pack &>/dev/null; then
  echo "❌ wasm-pack не найден. Установи Rust (https://rustup.rs), затем:"
  echo "   cargo install wasm-pack"
  echo "После установки перезапусти этот скрипт."
  exit 1
fi

cd brush/brush_nextjs
echo "→ Установка зависимостей (при необходимости)..."
npm install
echo "→ Сборка WASM и запуск Brush на http://localhost:3000 ..."
npm run dev
