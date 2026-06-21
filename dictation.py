#!/usr/bin/env python3
"""
Диктовка — локальная голосовая диктовка для macOS (Apple Silicon).

Говоришь — текст появляется в активном поле любого приложения. Всё работает
локально: без интернета (кроме разовой загрузки модели), без облаков и без API.
Языки: русский и английский.

Управление правым Option:
  - Удержание: держишь — пишет, отпустил — распознаёт и вставляет (push-to-talk).
  - Двойной короткий тап: старт записи hands-free; одиночный короткий тап — стоп.

Надёжность (чтобы НИЧЕГО не терялось):
  - Речь распознаётся по ходу дела: каждый кусок после паузы сразу распознаётся,
    вставляется и СРАЗУ пишется в ~/Documents/Диктовка.txt. Даже если приложение
    закрыть/убить во время записи, всё уже сказанное останется в этом файле.
  - hands-free сам останавливается через 5 минут, чтобы микрофон не висел вечно.

Конфиденциальность: в коде ноль сетевых вызовов. Аудио живёт только в памяти,
на диск не пишется (на диск идёт только РАСПОЗНАННЫЙ текст в твой файл). Текст
вставляется локально через Cmd+V. Нужны разрешения macOS: Микрофон,
Accessibility (вставка), Input Monitoring (горячая клавиша).
"""

import os
import queue
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

# распознавание по ходу речи
SILENCE_LEVEL = 0.01             # ниже этого уровня считаем тишиной
SILENCE_GAP_SEC = 0.7            # пауза, после которой кусок уходит в распознавание
MIN_FLUSH_SEC = 1.5             # не дробить слишком короткие куски
MAX_CHUNK_SEC = 25               # жёсткий предел длины куска (если говорят без пауз)
AUTO_STOP_SEC = 300              # авто-стоп hands-free (чтобы микрофон не висел)

# постоянный лог всего распознанного — страховка от потери
TRANSCRIPT_FILE = os.path.expanduser("~/Documents/Диктовка.txt")

TITLES = {"idle": "🎙Дикт", "recording": "🔴REC", "transcribing": "⏳…"}
STATUS = {"idle": "Готов", "recording": "● Запись…", "transcribing": "… Распознаю"}
LANG_CYCLE = [None, "ru", "en"]
LANG_LABEL = {None: "авто", "ru": "ru", "en": "en"}


def save_transcript(text):
    """Дописать распознанный текст в постоянный файл. Вызывается ДО вставки,
    чтобы текст сохранился, даже если вставка не удастся или процесс убьют."""
    try:
        os.makedirs(os.path.dirname(TRANSCRIPT_FILE), exist_ok=True)
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {text}\n")
    except Exception as e:
        print(f"[save] не смог записать в лог: {e}", file=sys.stderr)


class Recorder:
    """Запись с микрофона и распознавание. Речь распознаётся кусками по ходу дела
    (на паузах), поэтому ничего не копится в памяти и ничего не теряется."""

    def __init__(self, language=LANGUAGE, on_text=None):
        self.on_text = on_text or (lambda text, target: None)
        self.language = language
        self.frames = []
        self.samples = 0
        self.recording = False
        self.stream = None
        self.target_app = None
        self.last_level = 0.0
        self.lock = threading.Lock()
        self._stop_flusher = threading.Event()
        self._flusher = None
        # единственный поток распознавания: куски обрабатываются строго по очереди
        self.work = queue.Queue()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    # --- аудио ---
    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            with self.lock:
                self.frames.append(indata.copy())
                self.samples += frames
            self.last_level = float(np.abs(indata).max())

    def start(self):
        with self.lock:
            if self.recording:
                return
            self.frames = []
            self.samples = 0
            self.recording = True
            self.target_app = frontmost_app()
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
        self._stop_flusher.clear()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()
        print(f"● запись... (вставлю в: {self.target_app})", file=sys.stderr)

    def _drain(self):
        """Забрать накопленный звук, оставив буфер пустым (запись продолжается)."""
        with self.lock:
            frames = self.frames
            self.frames = []
            self.samples = 0
        if not frames:
            return None
        return np.concatenate(frames, axis=0).flatten()

    def _enqueue(self, audio):
        if audio is not None and len(audio):
            self.work.put((audio, self.target_app))

    def _flush_loop(self):
        """Пока идёт запись — отрезать куски на паузах (или по жёсткому пределу)
        и отправлять в распознавание, не прерывая запись."""
        silence = 0.0
        while not self._stop_flusher.is_set():
            time.sleep(0.15)
            if not self.recording:
                continue
            acc = self.samples / SAMPLE_RATE
            silence = silence + 0.15 if self.last_level < SILENCE_LEVEL else 0.0
            if (acc >= MIN_FLUSH_SEC and silence >= SILENCE_GAP_SEC) or acc >= MAX_CHUNK_SEC:
                self._enqueue(self._drain())
                silence = 0.0

    def stop(self):
        """Завершить запись и отправить остаток в распознавание."""
        self._stop_flusher.set()
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
            self.samples = 0
            target = self.target_app
        if frames:
            self.work.put((np.concatenate(frames, axis=0).flatten(), target))

    def cancel(self):
        """Прервать запись без распознавания (для слишком коротких тапов)."""
        self._stop_flusher.set()
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.frames = []
            self.samples = 0

    def busy(self):
        return not self.work.empty()

    # --- распознавание (один поток, строго по очереди) ---
    def _worker_loop(self):
        while True:
            audio, target = self.work.get()
            try:
                result = mlx_whisper.transcribe(
                    audio, path_or_hf_repo=MODEL_REPO, language=self.language,
                )
                text = result["text"].strip()
                if text:
                    save_transcript(text)        # сначала на диск — потом вставка
                    self.on_text(text, target)
            except Exception as e:
                print(f"[transcribe] ошибка: {e}", file=sys.stderr)
            finally:
                self.work.task_done()


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
        self.rec = Recorder(on_text=self._on_text)
        self.recording = False
        self.rec_mode = None      # как начата запись: "hold" (удержание) / "toggle"
        self.rec_start = 0.0
        self.pressed = set()
        self.last_tap = 0.0
        self.press_time = 0.0
        self.lang_idx = 0
        self._announced = False

        self.status_item = rumps.MenuItem(STATUS["idle"])
        self.record_item = rumps.MenuItem(
            "Записать  (Option: держать / 2×)", callback=self.on_record_click)
        self.lang_item = rumps.MenuItem(
            f"Язык: {LANG_LABEL[None]}", callback=self.on_lang_click)
        self.log_item = rumps.MenuItem(
            "Открыть лог расшифровок", callback=self.on_open_log)
        self.menu = [self.status_item, None, self.record_item, self.lang_item,
                     self.log_item, None]

        keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release).start()
        rumps.Timer(self.refresh_ui, 0.3).start()

    def _on_text(self, text, target):
        where = f" → {target}" if target else ""
        print(f"✓{where} {text}", file=sys.stderr)
        paste(text, target)

    # --- горячая клавиша: удержание = push-to-talk; короткий двойной тап = старт
    #     hands-free, короткий одиночный тап во время hands-free = стоп ---
    def on_press(self, key):
        if key in self.pressed:
            return                          # игнор автоповтора при удержании
        self.pressed.add(key)
        if key != TRIGGER_KEY:
            return
        if self.recording and self.rec_mode == "toggle":
            self.stop()                     # одиночный тап во время hands-free -> стоп
            return
        if not self.recording:
            self.press_time = time.monotonic()
            self.rec_mode = "hold"
            self.start()

    def on_release(self, key):
        self.pressed.discard(key)
        if key != TRIGGER_KEY:
            return
        if self.recording and self.rec_mode == "hold":
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
        if self.recording:
            self.stop()
        else:
            self.rec_mode = "toggle"
            self.start()

    def on_lang_click(self, _):
        self.lang_idx = (self.lang_idx + 1) % len(LANG_CYCLE)
        lang = LANG_CYCLE[self.lang_idx]
        self.rec.language = lang
        self.lang_item.title = f"Язык: {LANG_LABEL[lang]}"

    def on_open_log(self, _):
        subprocess.run(["open", TRANSCRIPT_FILE], check=False)

    # --- запись ---
    def start(self):
        self.rec.start()
        self.recording = True
        self.rec_start = time.monotonic()

    def stop(self):
        self.recording = False
        self.rec.stop()

    def cancel(self):
        self.recording = False
        self.rec.cancel()

    # --- значок (главный поток, через таймер) ---
    def refresh_ui(self, _):
        if not self._announced:
            self._announced = True
            try:
                rumps.notification(
                    "Диктовка запущена", "",
                    "Правый Option: держи — говори — отпусти. Или двойной тап.")
            except Exception:
                pass
        # авто-стоп hands-free, чтобы микрофон не висел вечно
        if self.recording and time.monotonic() - self.rec_start > AUTO_STOP_SEC:
            self.stop()
            try:
                rumps.notification("Диктовка: авто-стоп", "",
                                   "Прошло 5 минут. Дважды нажми Option, чтобы продолжить.")
            except Exception:
                pass
        if self.recording:
            state = "recording"
        elif self.rec.busy():
            state = "transcribing"
        else:
            state = "idle"
        self.title = TITLES[state]
        self.status_item.title = STATUS[state]


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
