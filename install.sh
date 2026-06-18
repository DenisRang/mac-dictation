#!/bin/bash
# Установщик «Диктовки»: создаёт окружение, собирает приложение, ставит в
# автозапуск и запускает. Запусти из папки проекта:  ./install.sh
set -euo pipefail

APP_NAME="Диктовка"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$HOME/Applications/$APP_NAME.app"
SUPPORT="$HOME/Library/Application Support/$APP_NAME"
VENV="$SUPPORT/venv"
MLX_WHISPER_VERSION="0.4.3"   # проверенная версия (ставится без torch)

echo "== $APP_NAME: установка =="

# 1. Только Apple Silicon
if [ "$(uname -m)" != "arm64" ]; then
  echo "Нужен Mac на Apple Silicon (M1/M2/M3/...). Прерываю."
  exit 1
fi

# 2. Найти Python 3.10+
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])' 2>/dev/null || echo 0)"
    if [ "$v" -ge 310 ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "Не найден Python 3.10 или новее."
  echo "Установи его одним из способов и запусти ./install.sh снова:"
  echo "  • https://www.python.org/downloads/macos/  (кнопка Download)"
  echo "  • или в терминале:  brew install python"
  exit 1
fi
echo "Python: $("$PY" --version)"

# pip-обёртка: тихо при успехе; при ошибке показать причину и выйти.
# (mlx-whisper в метаданных просит torch, но он не используется — его намеренно
#  нет, и единственное «предупреждение» об этом мы прячем.)
pip_install() {
  if ! "$VENV/bin/pip" install --quiet "$@" 2>/tmp/dictation_pip_err.log; then
    echo "Ошибка установки зависимостей:"; cat /tmp/dictation_pip_err.log; exit 1
  fi
}

# 3. Останавливаем прежнюю копию, если запущена
pkill -f "$APP_NAME.app/Contents/Resources/dictation.py" 2>/dev/null || true

# 4. Виртуальное окружение и зависимости
mkdir -p "$SUPPORT"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip 2>/dev/null || true
echo "Ставлю зависимости (несколько минут, скачивается ~300 МБ)…"
pip_install --no-deps "mlx-whisper==$MLX_WHISPER_VERSION"
pip_install -r "$SRC_DIR/requirements.txt"

# 5. Сборка .app (тонкая обёртка вокруг venv — без копирования Python)
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$SRC_DIR/dictation.py" "$APP_DIR/Contents/Resources/dictation.py"

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>com.dictation.local</string>
  <key>CFBundleExecutable</key><string>dictation</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Диктовка распознаёт твою речь локально, на устройстве.</string>
</dict>
</plist>
PLIST

cat > "$APP_DIR/Contents/MacOS/dictation" <<'LAUNCH'
#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/Library/Application Support/Диктовка/venv"
mkdir -p "$HOME/Library/Logs"
exec "$VENV/bin/python" "$HERE/../Resources/dictation.py" \
  >> "$HOME/Library/Logs/Диктовка.log" 2>&1
LAUNCH
chmod +x "$APP_DIR/Contents/MacOS/dictation"

# Зарегистрировать в Launch Services (Spotlight / Launchpad)
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -f "$APP_DIR" 2>/dev/null || true

# 6. Автозапуск при входе (без дублей)
osascript -e "tell application \"System Events\" to delete (every login item whose name is \"$APP_NAME\")" >/dev/null 2>&1 || true
osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP_DIR\", hidden:false, name:\"$APP_NAME\"}" >/dev/null 2>&1 || true

# 7. Запуск
open "$APP_DIR" || true

cat <<DONE

== Готово! ==
Приложение установлено и запущено (значок 🎙Дикт в строке меню сверху).

ОСТАЛОСЬ ВЫДАТЬ 3 РАЗРЕШЕНИЯ (один раз):
System Settings → Privacy & Security, и включи «Python» в трёх разделах:
  1. Microphone        — слышать тебя
  2. Accessibility     — вставлять текст
  3. Input Monitoring  — ловить правый Option
(пункт называется «Python» — это и есть Диктовка)

После включения Accessibility/Input Monitoring перезапусти приложение:
закрой через значок 🎙Дикт → «Выход» и снова открой из Launchpad («Диктовка»).

КАК ПОЛЬЗОВАТЬСЯ:
  • держи правый Option, говори, отпусти — текст вставится;
  • либо двойной тап правого Option → говори → одиночный тап.
Первая диктовка скачает модель (~1.5 ГБ), нужен интернет один раз.
DONE
