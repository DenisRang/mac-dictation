#!/usr/bin/env python3
"""
Диктовка — локальная голосовая диктовка для macOS (Apple Silicon).

Говоришь — текст появляется в активном поле любого приложения. Всё работает
локально: без интернета (кроме разовой загрузки модели), без облаков и без API.
Языки: русский и английский.

Управление правым Option:
  - Удержание: держишь — пишет, отпустил — распознаёт и вставляет (push-to-talk).
  - Двойной короткий тап: старт записи hands-free; одиночный короткий тап — стоп.

Значок в строке меню показывает состояние и даёт сменить язык / выйти.

Конфиденциальность: в коде ноль сетевых вызовов. Аудио живёт только в
оперативной памяти, на диск не пишется. Текст вставляется локально через Cmd+V.
Нужны разрешения macOS: Микрофон, Accessibility (вставка), Input Monitoring
(горячая клавиша).
"""

import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import pyperclip
import mlx_whisper
import rumps
from pynput import keyboard
from Quartz import (
    CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags,
    kCGEventFlagMaskCommand, kCGHIDEventTap,
)

# --- настройки ---
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"  # быстрая модель, RU/EN
SAMPLE_RATE = 16000
LANGUAGE = None                  # None = авто RU/EN; можно зашить "ru" или "en"
TRIGGER_KEY = keyboard.Key.alt_r  # правый Option
HOLD_SEC = 0.3                   # держишь дольше -> удержание; короче -> тап
DOUBLE_TAP_SEC = 0.4             # окно между двумя тапами для двойного тапа
_V_KEYCODE = 9                   # физическая клавиша V (Cmd+V = вставка)

TITLES = {"idle": "🎙Дикт", "recording": "🔴REC", "transcribing": "⏳…"}
STATUS = {"idle": "Готов", "recording": "● Запись…", "transcribing": "… Распознаю"}
LANG_CYCLE = [None, "ru", "en"]
LANG_LABEL = {None: "авто", "ru": "ru", "en": "en"}


class Recorder:
    """Запись с микрофона и распознавание через mlx-whisper. Аудио только в RAM."""

    def __init__(self, language=LANGUAGE):
        self.frames = []
        self.recording = False
        self.stream = None
        self.target_app = None    # приложение, активное в момент начала записи
        self.language = language
        self.lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        with self.lock:
            if self.recording:
                return
            self.frames = []
            self.recording = True
            self.target_app = frontmost_app()
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
        print(f"● запись... (вставлю в: {self.target_app})", file=sys.stderr)

    def stop_and_transcribe(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            frames = self.frames
            self.frames = []
            target = self.target_app

        if not frames:
            return
        audio = np.concatenate(frames, axis=0).flatten()
        print("… распознаю", file=sys.stderr)
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL_REPO, language=self.language,
        )
        text = result["text"].strip()
        if text:
            paste(text, target)
            where = f" → {target}" if target else ""
            print(f"✓{where} {text}", file=sys.stderr)
        else:
            print("(пусто)", file=sys.stderr)

    def cancel(self):
        """Прервать запись без распознавания (для слишком коротких тапов)."""
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.frames = []


def frontmost_app():
    """Имя процесса активного приложения (для лога). Тихо вернёт None при сбое."""
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of '
             'first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _press_cmd_v():
    """Нажать Cmd+V одним событием Quartz с явным флагом Command — мгновенно и
    надёжно (без гонок модификатора, из-за которых составной Cmd+V не доходил)."""
    try:
        down = CGEventCreateKeyboardEvent(None, _V_KEYCODE, True)
        CGEventSetFlags(down, kCGEventFlagMaskCommand)
        up = CGEventCreateKeyboardEvent(None, _V_KEYCODE, False)
        CGEventSetFlags(up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
    except Exception as e:
        print(f"[paste] не удалось вставить: {e}", file=sys.stderr)


def paste(text, target=None):
    """Мгновенная вставка: текст в буфер и сразу Cmd+V, затем вернуть прежний
    буфер. Без пауз. Нужен только Accessibility. target — лишь для лога."""
    try:
        prev = pyperclip.paste()
    except Exception:
        prev = ""
    pyperclip.copy(text)
    _press_cmd_v()
    threading.Timer(0.5, lambda: pyperclip.copy(prev)).start()


class DictateApp(rumps.App):
    """Значок в строке меню + горячая клавиша. Логика записи в Recorder."""

    def __init__(self):
        super().__init__(TITLES["idle"], quit_button="Выход")
        self.rec = Recorder()
        self.state = "idle"
        self.pressed = set()
        self.last_tap = 0.0
        self.press_time = 0.0
        self.rec_mode = None      # как начата запись: "hold" (удержание) / "toggle"
        self.lang_idx = 0
        self._announced = False   # баннер «запущена» показать один раз при старте

        self.status_item = rumps.MenuItem(STATUS["idle"])
        self.record_item = rumps.MenuItem(
            "Записать  (Option: держать / 2×)", callback=self.on_record_click)
        self.lang_item = rumps.MenuItem(
            f"Язык: {LANG_LABEL[None]}", callback=self.on_lang_click)
        self.menu = [self.status_item, None, self.record_item, self.lang_item, None]

        keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release).start()
        rumps.Timer(self.refresh_ui, 0.3).start()

    # --- горячая клавиша: удержание = push-to-talk; короткий двойной тап = старт
    #     hands-free, короткий одиночный тап во время hands-free = стоп ---
    def on_press(self, key):
        if key in self.pressed:
            return                          # игнор автоповтора при удержании
        self.pressed.add(key)
        if key != TRIGGER_KEY or self.state == "transcribing":
            return
        if self.state == "recording" and self.rec_mode == "toggle":
            self.stop()                     # одиночный тап во время hands-free -> стоп
            return
        if self.state == "idle":
            # начать запись сразу (нужно для удержания); тип решим на отпускании
            self.press_time = time.monotonic()
            self.rec_mode = "hold"
            self.start()

    def on_release(self, key):
        self.pressed.discard(key)
        if key != TRIGGER_KEY:
            return
        if self.state == "recording" and self.rec_mode == "hold":
            held = time.monotonic() - self.press_time
            if held >= HOLD_SEC:
                self.stop()                 # держал -> распознать и вставить
            else:
                self.cancel()               # короткий тап -> отбросить запись
                now = time.monotonic()
                if now - self.last_tap <= DOUBLE_TAP_SEC:
                    self.last_tap = 0.0
                    self.rec_mode = "toggle"   # второй тап -> hands-free запись
                    self.start()
                else:
                    self.last_tap = now

    # --- клики по меню ---
    def on_record_click(self, _):
        if self.state == "idle":
            self.rec_mode = "toggle"
            self.start()
        elif self.state == "recording":
            self.stop()

    def on_lang_click(self, _):
        self.lang_idx = (self.lang_idx + 1) % len(LANG_CYCLE)
        lang = LANG_CYCLE[self.lang_idx]
        self.rec.language = lang
        self.lang_item.title = f"Язык: {LANG_LABEL[lang]}"

    # --- запись ---
    def start(self):
        self.rec.start()
        self.state = "recording"

    def stop(self):
        self.state = "transcribing"         # распознавание в фоне, UI не виснет
        threading.Thread(target=self._finish, daemon=True).start()

    def _finish(self):
        try:
            self.rec.stop_and_transcribe()
        finally:
            self.state = "idle"

    def cancel(self):
        self.rec.cancel()
        self.state = "idle"

    # --- обновление значка (главный поток, через таймер) ---
    def refresh_ui(self, _):
        if not self._announced:
            self._announced = True
            try:
                rumps.notification(
                    "Диктовка запущена", "",
                    "Правый Option: держи — говори — отпусти. Или двойной тап.")
            except Exception:
                pass
        self.title = TITLES[self.state]
        self.status_item.title = STATUS[self.state]


def main():
    # Утилита строки меню: без иконки в Dock.
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory)
    except Exception:
        pass
    # Попросить Accessibility (нужно для вставки): покажет системный диалог и
    # добавит приложение в список, останется включить галку.
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions
        AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})
    except Exception:
        pass
    DictateApp().run()


if __name__ == "__main__":
    main()
