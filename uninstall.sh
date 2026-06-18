#!/bin/bash
# Полное удаление «Диктовки».  Запусти:  ./uninstall.sh
set -uo pipefail

APP_NAME="Диктовка"
APP_DIR="$HOME/Applications/$APP_NAME.app"
SUPPORT="$HOME/Library/Application Support/$APP_NAME"

echo "Удаляю $APP_NAME…"

# остановить процесс
pkill -f "$APP_NAME.app/Contents/Resources/dictation.py" 2>/dev/null || true

# убрать из автозапуска
osascript -e "tell application \"System Events\" to delete (every login item whose name is \"$APP_NAME\")" >/dev/null 2>&1 || true

# удалить приложение и окружение
rm -rf "$APP_DIR"
rm -rf "$SUPPORT"
rm -f "$HOME/Library/Logs/$APP_NAME.log"

echo "Готово. Разрешения «Python» можно убрать вручную в System Settings → Privacy & Security."
echo "Скачанную модель (если нужно освободить место) ищи в ~/.cache/huggingface."
