import { createContext, createElement, useContext, useState } from "react";

export const LANGS = ["ru", "en"];

const messages = {
  ru: {
    // Navigation
    "nav.home": "Главная",
    "nav.capabilities": "Возможности",
    "nav.integrations": "Подключения",
    "nav.console": "Диагностика",
    "nav.settings": "Настройки",
    "nav.about": "О Sens",

    // View headers
    "view.home.kicker": "Система восприятия",
    "view.home.title": "Sens готов",
    "view.home.desc": "Зрение и слух доступны подключённой модели.",
    "view.capabilities.kicker": "Возможности",
    "view.capabilities.title": "Чувства модели",
    "view.capabilities.desc": "Выберите чувство, чтобы настроить его точнее.",
    "view.integrations.kicker": "Подключения",
    "view.integrations.title": "Подключённые модели",
    "view.integrations.desc": "Один локальный мост для Z-Code и других клиентов.",
    "view.console.kicker": "Диагностика",
    "view.console.title": "Система стабильна",
    "view.console.desc": "Broker и MCP отвечают без критических ошибок.",
    "view.settings.kicker": "Настройки",
    "view.settings.title": "Поведение Sens",
    "view.settings.desc": "Автозапуск, приватность и управление ресурсами.",
    "view.about.kicker": "Sens 1.1",
    "view.about.title": "Больше, чем зрение",
    "view.about.desc": "Расширяемая система чувств для любой модели.",

    // Capabilities
    "cap.sight.name": "Зрение",
    "cap.sight.description": "Анализ изображений, OCR, поиск деталей и визуальная самопроверка.",
    "cap.sight.settingsDescription": "Провайдер, модель, качество и лимиты визуального анализа.",
    "cap.sight.openSettings": "Открыть настройки зрения",
    "cap.sight.access": "Доступ к зрению",
    "cap.sight.pageKicker": "Возможности / Зрение",
    "cap.sight.pageTitle": "Настройка зрения",
    "cap.hearing.name": "Слух",
    "cap.hearing.description": "Локальное распознавание аудиофайлов без доступа модели к микрофону.",
    "cap.hearing.settingsDescription": "Модель распознавания, устройство, точность и подготовка текста.",
    "cap.hearing.openSettings": "Открыть настройки слуха",
    "cap.hearing.access": "Доступ к слуху",
    "cap.hearing.pageKicker": "Возможности / Слух",
    "cap.hearing.pageTitle": "Настройка слуха",

    // Status labels
    "state.ready": "готово",
    "state.readyUpper": "Готово",
    "state.off": "выключен",
    "state.offUpper": "Выключено",
    "state.active": "Активно",
    "state.listening": "слушает",
    "state.needStart": "нужен запуск",
    "state.needEnable": "включите движок",
    "state.error": "ошибка",
    "status.connected": "Подключено",
    "status.enabled": "Включено",
    "integ.addClient": "Добавить клиент",

    // Home
    "home.connectCta": "Подключить модель или приложение",
    "home.activeCapabilities": "Активные возможности",
    "home.allCapabilities": "Все возможности",
    "home.currentConnection": "Текущее подключение",
    "home.local": "Локально",
    "home.lastAction": "Последнее действие: анализ изображения",

    // Capabilities page
    "future.title": "Будущие чувства",
    "future.desc": "Новые модули появятся здесь после установки — отдельно от подключения моделей.",
    "future.badge": "Roadmap",

    // Connect modal
    "modal.kicker": "Новое подключение",
    "modal.title": "Дать модели чувства",
    "modal.desc": "Выберите клиент. Sens подготовит локальное MCP-подключение без облачного посредника.",
    "modal.clientLabel": "Клиент",
    "modal.otherClient": "Другой MCP-клиент",
    "modal.connect": "Подключить {client}",
    "modal.close": "Закрыть",

    // Window controls
    "win.minimize": "Свернуть",
    "win.maximize": "Развернуть",
    "win.close": "Закрыть",

    // Capability settings
    "settings.back": "Все возможности",
    "dictation.kicker": "Диктовка · {state}",
    "dictation.title": "Говорите — Sens вставит текст",
    "dictation.desc": "Удерживайте {hotkey}, говорите и отпустите клавиши. Текст появится в активном поле.",
    "dictation.hotkeyLabel": "Комбинация клавиш",
    "dictation.hotkeyRecommended": "Ctrl + Win · рекомендуется",
    "dictation.start": "Запустить слух",
    "dictation.running": "Служба работает в фоне",
    "group.model": "Модель",
    "group.model.sightSub": "Кто будет разбирать изображение",
    "group.model.hearingSub": "Кто будет распознавать аудио",
    "field.provider": "Провайдер",
    "field.visionModel": "Vision-модель",
    "field.visionModelHint": "Можно указать любую совместимую модель выбранного провайдера.",
    "field.recognitionModel": "Модель распознавания",
    "field.device": "Устройство",
    "device.auto": "Автоматически",
    "device.cpu": "CPU",
    "device.cuda": "NVIDIA CUDA",
    "group.quality": "Качество и ресурсы",
    "group.quality.sub": "Баланс скорости и точности",
    "field.detail": "Детализация",
    "detail.quick": "Quick · быстро",
    "detail.normal": "Normal · баланс",
    "detail.deep": "Deep · максимум деталей",
    "field.maxCalls": "Максимум vision-вызовов",
    "maxCalls.hint": "Жёсткий лимит на одну визуальную задачу.",
    "field.beam": "Beam size",
    "beam.hint": "Больше — точнее, но медленнее.",
    "field.vad": "Чувствительность к речи",
    "vad.high": "Высокая",
    "vad.balanced": "Сбалансированная",
    "vad.confident": "Только уверенная речь",
    "toggle.accessDesc": "Разрешить подключённым моделям использовать эту возможность",
    "toggle.cache": "Кэшировать результаты",
    "toggle.cacheDesc": "Повторный анализ того же изображения не расходует vision-вызовы",
    "toggle.verify": "Перепроверять ответ",
    "toggle.verifyDesc": "Двойной проход: сверка результата с изображением и исправление ошибок (больше времени и расходов)",
    "toggle.preload": "Предзагружать модель",
    "toggle.preloadDesc": "Быстрее первый ответ, но больше памяти в фоне",
    "toggle.postprocess": "Обрабатывать текст",
    "toggle.postprocessDesc": "Пунктуация, пробелы и очистка результата",
    "toggle.clipboard": "Копировать в буфер обмена",
    "toggle.clipboardDesc": "Сохранять расшифровку, даже если активное поле недоступно",
    "toggle.paste": "Вставлять в активное поле",
    "toggle.pasteDesc": "После отпускания клавиш вставлять готовый текст туда, где находится курсор",
    "actions.hintSight": "Изменения применятся к следующей задаче.",
    "actions.hintHearing": "Настройки диктовки применятся сразу после сохранения.",
    "actions.save": "Сохранить настройки",
    "actions.saving": "Сохраняю…",

    // Update card
    "update.label": "Обновления",
    "update.check": "Проверить обновления",
    "update.checking": "Проверяю…",
    "update.install": "Обновить до {version}",
    "update.idle": "Sens проверит подписанный релиз на GitHub.",
    "update.checkingStatus": "Проверяем канал обновлений…",
    "update.current": "У вас последняя версия.",
    "update.available": "Доступна версия {version}.",
    "update.downloading": "Загрузка обновления{progress}…",
    "update.installing": "Установка обновления…",
    "update.errorFallback": "Не удалось проверить обновления.",

    // Detail rows
    "status.ready": "Готово",
    "status.waiting": "Ожидание",
    "integ.zcode.title": "Z-Code",
    "integ.zcode.meta": "MCP · stdio",
    "integ.zcode.desc": "15 инструментов Sens доступны модели",
    "integ.broker.title": "Локальный broker",
    "integ.broker.meta": "Named pipe",
    "integ.broker.desc": "Один процесс обслуживает все клиенты",
    "console.broker.title": "Broker",
    "console.broker.meta": "Rust",
    "console.broker.desc": "Среднее время ответа 14 мс",
    "console.sight.title": "Sight worker",
    "console.sight.meta": "Node.js",
    "console.sight.desc": "Запускается только при запросе",
    "console.hearing.title": "Hearing worker",
    "console.hearing.meta": "Python",
    "console.hearing.desc": "Изолирован и завершает дочерние процессы",
    "settings.autostart.title": "Запускать вместе с Windows",
    "settings.autostart.meta": "Система",
    "settings.autostart.desc": "Sens появляется в трее без открытия окна",
    "settings.privacy.title": "Локальная обработка",
    "settings.privacy.meta": "Приватность",
    "settings.privacy.desc": "Данные не покидают выбранного провайдера",
    "about.sens.title": "Sens",
    "about.sens.meta": "Версия {version}",
    "about.sens.desc": "Одна точка подключения для чувств модели",
    "about.arch.title": "Архитектура",
    "about.arch.meta": "Rust + sidecars",
    "about.arch.desc": "Новые чувства подключаются как модули",

    // Tray
    "tray.background": "Sens работает в фоне",
    "tray.starting": "Sens запускается",
    "tray.systemReady": "Система готова",
    "tray.diagnostics": "Нужна диагностика",
    "tray.open": "Открыть",
    "tray.quit": "Выйти",
    "tray.vision": "Зрение",
    "tray.hearing": "Слух",
    "tray.zcode": "Z-Code",
    "restore.window": "Окно Sens свёрнуто — открыть",

    // Toasts
    "toast.connected": "{client} подключён к Sens",
    "toast.connectFailed": "Не удалось подключить {client}: {error}",
    "toast.saved": "{name}: настройки сохранены",
    "toast.saveFailed": "Не удалось сохранить: {error}",
    "toast.speechStarted": "Слух запущен. Удерживайте Ctrl + Win, чтобы говорить.",
    "toast.speechFailed": "Не удалось запустить слух: {error}",

    // Providers and models
    "provider.mimo": "MiMo",
    "provider.openai": "OpenAI",
    "provider.custom": "Свой провайдер",
    "model.parakeet": "Parakeet · быстрая",
    "model.parakeet.desc": "600M, мультиязычная, быстрая на CPU",
    "model.whisperRu": "Whisper RU · точная",
    "model.whisperRu.desc": "RU + EN code-switching, высокая точность",
    "model.gigaam": "GigaAM v3 · русский",
    "model.gigaam.desc": "230M, локальная русская модель с пунктуацией",

    // Language
    "lang.label": "Язык интерфейса",
    "lang.ru": "Русский",
    "lang.en": "English",
  },

  en: {
    "nav.home": "Home",
    "nav.capabilities": "Capabilities",
    "nav.integrations": "Integrations",
    "nav.console": "Diagnostics",
    "nav.settings": "Settings",
    "nav.about": "About Sens",

    "view.home.kicker": "Perception system",
    "view.home.title": "Sens is ready",
    "view.home.desc": "Vision and hearing are available to the connected model.",
    "view.capabilities.kicker": "Capabilities",
    "view.capabilities.title": "Model senses",
    "view.capabilities.desc": "Choose a sense to fine-tune it.",
    "view.integrations.kicker": "Integrations",
    "view.integrations.title": "Connected models",
    "view.integrations.desc": "One local bridge for Z-Code and other clients.",
    "view.console.kicker": "Diagnostics",
    "view.console.title": "System is stable",
    "view.console.desc": "Broker and MCP respond without critical errors.",
    "view.settings.kicker": "Settings",
    "view.settings.title": "Sens behavior",
    "view.settings.desc": "Startup, privacy, and resource management.",
    "view.about.kicker": "Sens 1.1",
    "view.about.title": "More than vision",
    "view.about.desc": "An extensible system of senses for any model.",

    "cap.sight.name": "Vision",
    "cap.sight.description": "Image analysis, OCR, detail search, and visual self-check.",
    "cap.sight.settingsDescription": "Provider, model, quality, and limits of visual analysis.",
    "cap.sight.openSettings": "Open vision settings",
    "cap.sight.access": "Vision access",
    "cap.sight.pageKicker": "Capabilities / Vision",
    "cap.sight.pageTitle": "Vision settings",
    "cap.hearing.name": "Hearing",
    "cap.hearing.description": "Local audio transcription without model access to the microphone.",
    "cap.hearing.settingsDescription": "Recognition model, device, accuracy, and text processing.",
    "cap.hearing.openSettings": "Open hearing settings",
    "cap.hearing.access": "Hearing access",
    "cap.hearing.pageKicker": "Capabilities / Hearing",
    "cap.hearing.pageTitle": "Hearing settings",

    "state.ready": "ready",
    "state.readyUpper": "Ready",
    "state.off": "off",
    "state.offUpper": "Off",
    "state.active": "Active",
    "state.listening": "listening",
    "state.needStart": "needs start",
    "state.needEnable": "enable the engine",
    "state.error": "error",
    "status.connected": "Connected",
    "status.enabled": "Enabled",
    "integ.addClient": "Add client",

    "home.connectCta": "Connect a model or app",
    "home.activeCapabilities": "Active capabilities",
    "home.allCapabilities": "All capabilities",
    "home.currentConnection": "Current connection",
    "home.local": "Local",
    "home.lastAction": "Last action: image analysis",

    "future.title": "Future senses",
    "future.desc": "New modules will appear here after installation — separate from model connections.",
    "future.badge": "Roadmap",

    "modal.kicker": "New connection",
    "modal.title": "Give the model senses",
    "modal.desc": "Choose a client. Sens will set up a local MCP connection with no cloud intermediary.",
    "modal.clientLabel": "Client",
    "modal.otherClient": "Other MCP client",
    "modal.connect": "Connect {client}",
    "modal.close": "Close",

    "win.minimize": "Minimize",
    "win.maximize": "Maximize",
    "win.close": "Close",

    "settings.back": "All capabilities",
    "dictation.kicker": "Dictation · {state}",
    "dictation.title": "Speak — Sens will insert the text",
    "dictation.desc": "Hold {hotkey}, speak, and release. The text will appear in the active field.",
    "dictation.hotkeyLabel": "Hotkey",
    "dictation.hotkeyRecommended": "Ctrl + Win · recommended",
    "dictation.start": "Start hearing",
    "dictation.running": "Service runs in the background",
    "group.model": "Model",
    "group.model.sightSub": "Who will analyze the image",
    "group.model.hearingSub": "Who will recognize the audio",
    "field.provider": "Provider",
    "field.visionModel": "Vision model",
    "field.visionModelHint": "Any compatible model of the selected provider.",
    "field.recognitionModel": "Recognition model",
    "field.device": "Device",
    "device.auto": "Automatic",
    "device.cpu": "CPU",
    "device.cuda": "NVIDIA CUDA",
    "group.quality": "Quality and resources",
    "group.quality.sub": "Balance of speed and accuracy",
    "field.detail": "Detail",
    "detail.quick": "Quick · fast",
    "detail.normal": "Normal · balanced",
    "detail.deep": "Deep · maximum detail",
    "field.maxCalls": "Max vision calls",
    "maxCalls.hint": "Hard limit per visual task.",
    "field.beam": "Beam size",
    "beam.hint": "More is more accurate but slower.",
    "field.vad": "Speech sensitivity",
    "vad.high": "High",
    "vad.balanced": "Balanced",
    "vad.confident": "Confident speech only",
    "toggle.accessDesc": "Allow connected models to use this capability",
    "toggle.cache": "Cache results",
    "toggle.cacheDesc": "Re-analyzing the same image does not spend vision calls",
    "toggle.verify": "Re-verify the answer",
    "toggle.verifyDesc": "Two-pass: cross-check the result against the image and fix errors (more time and cost)",
    "toggle.preload": "Preload the model",
    "toggle.preloadDesc": "Faster first response, but more memory in the background",
    "toggle.postprocess": "Post-process text",
    "toggle.postprocessDesc": "Punctuation, spacing, and result cleanup",
    "toggle.clipboard": "Copy to clipboard",
    "toggle.clipboardDesc": "Keep the transcript even if the active field is unavailable",
    "toggle.paste": "Paste into the active field",
    "toggle.pasteDesc": "Insert the finished text where the cursor is after releasing the keys",
    "actions.hintSight": "Changes apply to the next task.",
    "actions.hintHearing": "Dictation settings apply immediately after saving.",
    "actions.save": "Save settings",
    "actions.saving": "Saving…",

    "update.label": "Updates",
    "update.check": "Check for updates",
    "update.checking": "Checking…",
    "update.install": "Update to {version}",
    "update.idle": "Sens will check the signed GitHub release.",
    "update.checkingStatus": "Checking the update channel…",
    "update.current": "You are on the latest version.",
    "update.available": "Version {version} is available.",
    "update.downloading": "Downloading update{progress}…",
    "update.installing": "Installing the update…",
    "update.errorFallback": "Could not check for updates.",

    "status.ready": "Ready",
    "status.waiting": "Waiting",
    "integ.zcode.title": "Z-Code",
    "integ.zcode.meta": "MCP · stdio",
    "integ.zcode.desc": "15 Sens tools available to the model",
    "integ.broker.title": "Local broker",
    "integ.broker.meta": "Named pipe",
    "integ.broker.desc": "One process serves all clients",
    "console.broker.title": "Broker",
    "console.broker.meta": "Rust",
    "console.broker.desc": "Average response time 14 ms",
    "console.sight.title": "Sight worker",
    "console.sight.meta": "Node.js",
    "console.sight.desc": "Starts only on request",
    "console.hearing.title": "Hearing worker",
    "console.hearing.meta": "Python",
    "console.hearing.desc": "Isolated and terminates child processes",
    "settings.autostart.title": "Launch with Windows",
    "settings.autostart.meta": "System",
    "settings.autostart.desc": "Sens appears in the tray without opening a window",
    "settings.privacy.title": "Local processing",
    "settings.privacy.meta": "Privacy",
    "settings.privacy.desc": "Data does not leave the selected provider",
    "about.sens.title": "Sens",
    "about.sens.meta": "Version {version}",
    "about.sens.desc": "One connection point for model senses",
    "about.arch.title": "Architecture",
    "about.arch.meta": "Rust + sidecars",
    "about.arch.desc": "New senses plug in as modules",

    "tray.background": "Sens is running in the background",
    "tray.starting": "Sens is starting",
    "tray.systemReady": "System ready",
    "tray.diagnostics": "Diagnostics needed",
    "tray.open": "Open",
    "tray.quit": "Quit",
    "tray.vision": "Vision",
    "tray.hearing": "Hearing",
    "tray.zcode": "Z-Code",
    "restore.window": "Sens window is minimized — open",

    "toast.connected": "{client} connected to Sens",
    "toast.connectFailed": "Could not connect {client}: {error}",
    "toast.saved": "{name}: settings saved",
    "toast.saveFailed": "Could not save: {error}",
    "toast.speechStarted": "Hearing started. Hold Ctrl + Win to speak.",
    "toast.speechFailed": "Could not start hearing: {error}",

    "provider.mimo": "MiMo",
    "provider.openai": "OpenAI",
    "provider.custom": "Custom provider",
    "model.parakeet": "Parakeet · fast",
    "model.parakeet.desc": "600M, multilingual, fast on CPU",
    "model.whisperRu": "Whisper RU · accurate",
    "model.whisperRu.desc": "RU + EN code-switching, high accuracy",
    "model.gigaam": "GigaAM v3 · Russian",
    "model.gigaam.desc": "230M, local Russian model with punctuation",

    "lang.label": "Interface language",
    "lang.ru": "Русский",
    "lang.en": "English",
  },
};

export function detectLanguage() {
  try {
    const saved = localStorage.getItem("sens.lang");
    if (saved === "ru" || saved === "en") return saved;
    if ((navigator.language || "").toLowerCase().startsWith("en")) return "en";
  } catch {
    // Storage unavailable; fall through to the default.
  }
  return "ru";
}

export function saveLanguage(lang) {
  try {
    localStorage.setItem("sens.lang", lang);
  } catch {
    // Storage unavailable; the choice is not persisted.
  }
}

export function translate(lang, key, vars) {
  let text = messages[lang]?.[key] ?? messages.ru[key] ?? key;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      text = text.split(`{${name}}`).join(String(value));
    }
  }
  return text;
}

export const LanguageContext = createContext({ lang: "ru", setLang: () => {} });

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(detectLanguage);
  const setLang = (next) => {
    setLangState(next);
    saveLanguage(next);
  };
  return createElement(LanguageContext.Provider, { value: { lang, setLang } }, children);
}

export function useLanguage() {
  return useContext(LanguageContext);
}

export function useT() {
  const { lang } = useLanguage();
  return (key, vars) => translate(lang, key, vars);
}
