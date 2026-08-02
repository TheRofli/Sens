import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  IconActivity,
  IconAdjustmentsHorizontal,
  IconArrowLeft,
  IconBox,
  IconChevronRight,
  IconClock,
  IconDeviceFloppy,
  IconDownload,
  IconEye,
  IconHeadphones,
  IconHome,
  IconInfoCircle,
  IconKeyboard,
  IconMinus,
  IconMicrophone,
  IconPlugConnected,
  IconPointFilled,
  IconPlus,
  IconPrompt,
  IconPuzzle,
  IconRefresh,
  IconSettings,
  IconSparkles,
  IconSquare,
  IconPlayerPlay,
  IconPower,
  IconX,
} from "@tabler/icons-react";

const navItems = [
  { id: "home", label: "Главная", icon: IconHome },
  { id: "capabilities", label: "Возможности", icon: IconBox },
  { id: "integrations", label: "Подключения", icon: IconPuzzle },
  { id: "console", label: "Диагностика", icon: IconPrompt },
  { id: "settings", label: "Настройки", icon: IconSettings },
  { id: "about", label: "О Sens", icon: IconInfoCircle },
];

const viewCopy = {
  home: ["Система восприятия", "Sens готов", "Зрение и слух доступны подключённой модели."],
  capabilities: ["Возможности", "Чувства модели", "Выберите чувство, чтобы настроить его точнее."],
  integrations: ["Подключения", "Подключённые модели", "Один локальный мост для Z-Code и других клиентов."],
  console: ["Диагностика", "Система стабильна", "Broker и MCP отвечают без критических ошибок."],
  settings: ["Настройки", "Поведение Sens", "Автозапуск, приватность и управление ресурсами."],
  about: ["Sens 1.1", "Больше, чем зрение", "Расширяемая система чувств для любой модели."],
};

const defaultCapabilitySettings = {
  sight: {
    enabled: true,
    provider: "mimo",
    model: "mimo-v2.5",
    detail: "normal",
    cache: true,
    maxCallsPerImage: 8,
    verify: false,
  },
  hearing: {
    enabled: true,
    model: "gigaam",
    device: "cpu",
    hotkey: "ctrl+win",
    copyToClipboard: true,
    pasteToActiveInput: true,
    suppressHotkey: false,
    preloadModel: true,
    beamSize: 5,
    postprocessText: true,
    vadSensitivity: 0.02,
  },
  sightProviders: [
    { value: "mimo", label: "MiMo", model: "mimo-v2.5" },
    { value: "openai", label: "OpenAI", model: "gpt-4.1-mini" },
    { value: "custom", label: "Свой провайдер", model: "vision-model-name" },
  ],
  hearingModels: [
    { value: "parakeet", label: "Parakeet · быстрая", description: "600M, мультиязычная, быстрая на CPU" },
    { value: "whisper-ru", label: "Whisper RU · точная", description: "RU + EN code-switching, высокая точность" },
    { value: "gigaam", label: "GigaAM v3 · русский", description: "230M, локальная русская модель с пунктуацией" },
  ],
};

const capabilityMeta = {
  sight: {
    name: "Зрение",
    genitive: "зрения",
    dative: "зрению",
    source: "Eye",
    description: "Анализ изображений, OCR, поиск деталей и визуальная самопроверка.",
    settingsDescription: "Провайдер, модель, качество и лимиты визуального анализа.",
    icon: IconEye,
  },
  hearing: {
    name: "Слух",
    genitive: "слуха",
    dative: "слуху",
    source: "Speech",
    description: "Локальное распознавание аудиофайлов без доступа модели к микрофону.",
    settingsDescription: "Модель распознавания, устройство, точность и подготовка текста.",
    icon: IconHeadphones,
  },
};

function BrandMark({ tone = "dark", size = 48 }) {
  return <img className={`brand-mark brand-mark--${tone}`} src="/assets/sens-mark-source.png" width={size} height={size} alt="Sens" draggable="false" />;
}

function StatusDot({ tone = "ready", label }) {
  return <span className={`status-dot status-dot--${tone}`} aria-label={label}><IconPointFilled size={16} stroke={0} aria-hidden="true" /></span>;
}

function WindowControls({ onMinimize, onMaximize, onClose }) {
  return (
    <div className="window-controls" aria-label="Управление окном">
      <button type="button" onClick={onMinimize} aria-label="Свернуть"><IconMinus size={19} /></button>
      <button type="button" onClick={onMaximize} aria-label="Развернуть"><IconSquare size={16} /></button>
      <button type="button" onClick={onClose} aria-label="Закрыть"><IconX size={20} /></button>
    </div>
  );
}

function hearingUiState(enabled, runtime) {
  if (!enabled) return { label: "выключен", tone: "idle" };
  if (runtime?.transcribing) return { label: "слушает", tone: "attention" };
  if (!runtime?.running) return { label: "нужен запуск", tone: "attention" };
  if (!runtime?.enabled) return { label: "включите движок", tone: "attention" };
  return { label: "готово", tone: "ready" };
}

function formatHotkey(value = "ctrl+win") {
  const labels = { ctrl: "Ctrl", win: "Win", shift: "Shift", alt: "Alt", space: "Space" };
  return value.split("+").map((part) => labels[part] || part).join(" + ");
}

function CapabilityPill({ capability, enabled, status, onClick }) {
  const meta = capabilityMeta[capability];
  const state = status || { label: enabled ? "готово" : "выключено", tone: enabled ? "ready" : "idle" };
  return (
    <button type="button" className="capability-pill" onClick={onClick} aria-label={`Открыть настройки ${meta.genitive}`}>
      <StatusDot tone={state.tone} label={`${meta.name}: ${state.label}`} />
      <span className="capability-pill__copy"><strong>{meta.name}</strong><small>{state.label}</small></span>
    </button>
  );
}

function ConnectModal({ onClose, onConnected }) {
  const [client, setClient] = useState("Z-Code");
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="connect-modal" role="dialog" aria-modal="true" aria-labelledby="connect-title" onMouseDown={(event) => event.stopPropagation()}>
        <button className="modal-close" type="button" onClick={onClose} aria-label="Закрыть"><IconX size={22} /></button>
        <p className="section-kicker">Новое подключение</p>
        <h2 id="connect-title">Дать модели чувства</h2>
        <p>Выберите клиент. Sens подготовит локальное MCP-подключение без облачного посредника.</p>
        <label htmlFor="client">Клиент</label>
        <select id="client" value={client} onChange={(event) => setClient(event.target.value)}>
          <option>Z-Code</option><option>Claude Desktop</option><option>Cursor</option><option>Другой MCP-клиент</option>
        </select>
        <button className="primary-button" type="button" onClick={() => onConnected(client)}><IconPlugConnected size={20} />Подключить {client}</button>
      </section>
    </div>
  );
}

function HomeContent({ settings, speechRuntime, openCapability, openCapabilities, openConnect }) {
  const hearingState = hearingUiState(settings.hearing.enabled, speechRuntime);
  return (
    <>
      <section className="home-grid" aria-label="Главная панель">
        <button className="connect-banner" type="button" onClick={openConnect}>
          <IconPlus size={31} stroke={1.7} /><span>Подключить модель или приложение</span>
        </button>
        <section className="capability-chamber">
          <p>Активные возможности</p>
          <div className="capability-row">
            <CapabilityPill capability="sight" enabled={settings.sight.enabled} onClick={() => openCapability("sight")} />
            <span className="capability-divider" />
            <CapabilityPill capability="hearing" enabled={settings.hearing.enabled} status={hearingState} onClick={() => openCapability("hearing")} />
            <span className="capability-divider" />
            <button className="capability-overview" type="button" onClick={openCapabilities}><IconAdjustmentsHorizontal size={19} />Все возможности</button>
          </div>
        </section>
        <aside className="connection-card">
          <p>Текущее подключение</p><strong>Z-Code</strong>
          <div className="connection-line"><StatusDot tone="attention" label="MCP активен" /><span>MCP</span></div>
          <div className="connection-line"><StatusDot label="Локальное подключение готово" /><span>Локально</span></div>
        </aside>
      </section>
      <footer className="activity-bar"><IconClock size={28} stroke={1.8} /><span>Последнее действие: анализ изображения</span></footer>
    </>
  );
}

function CapabilityCard({ capability, enabled, settings, runtimeStatus, onOpen }) {
  const meta = capabilityMeta[capability];
  const Icon = meta.icon;
  const summary = capability === "sight" ? `${settings.provider} · ${settings.model}` : `${settings.model} · ${settings.device.toUpperCase()}`;
  const state = capability === "hearing" ? hearingUiState(enabled, runtimeStatus) : { label: enabled ? "Готово" : "Выключено", tone: enabled ? "ready" : "idle" };
  return (
    <button className="detail-card capability-card" type="button" onClick={onOpen} aria-label={`Открыть настройки ${meta.genitive}`}>
      <div className="detail-card__icon"><Icon size={28} /></div>
      <div><p>{meta.source} · {summary}</p><h3>{meta.name}</h3><span>{meta.description}</span></div>
      <div className="capability-card__tail">
        <span className="detail-card__status"><StatusDot tone={state.tone} label={state.label} />{state.label}</span>
        <IconChevronRight size={22} aria-hidden="true" />
      </div>
    </button>
  );
}

function CapabilitiesContent({ settings, speechRuntime, onOpenCapability }) {
  return (
    <section className="detail-panel" aria-label="Каталог возможностей">
      <div className="detail-list">
        <CapabilityCard capability="sight" enabled={settings.sight.enabled} settings={settings.sight} onOpen={() => onOpenCapability("sight")} />
        <CapabilityCard capability="hearing" enabled={settings.hearing.enabled} settings={settings.hearing} runtimeStatus={speechRuntime} onOpen={() => onOpenCapability("hearing")} />
      </div>
      <aside className="future-sense-card">
        <div className="future-sense-card__icon"><IconSparkles size={24} /></div>
        <div><strong>Будущие чувства</strong><span>Новые модули появятся здесь после установки — отдельно от подключения моделей.</span></div>
        <span className="future-sense-card__badge">Roadmap</span>
      </aside>
    </section>
  );
}

function SettingsToggle({ label, description, checked, onChange }) {
  return (
    <label className="setting-toggle">
      <span><strong>{label}</strong><small>{description}</small></span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function CapabilitySettingsContent({ capability, data, speechRuntime, onStartSpeech, onBack, onSave, saving }) {
  const meta = capabilityMeta[capability];
  const Icon = meta.icon;
  const current = data[capability];
  const [draft, setDraft] = useState(current);

  useEffect(() => setDraft(current), [current, capability]);

  const update = (key, value) => setDraft((previous) => ({ ...previous, [key]: value }));
  const providerChanged = (provider) => {
    const option = data.sightProviders.find((item) => item.value === provider);
    setDraft((previous) => ({ ...previous, provider, model: option?.model || previous.model }));
  };

  return (
    <section className="capability-settings" aria-label={`Настройки ${meta.genitive}`}>
      <button className="back-button" type="button" onClick={onBack}><IconArrowLeft size={19} />Все возможности</button>
      <div className="capability-summary">
        <div className="capability-summary__icon"><Icon size={34} /></div>
        <div><p>{meta.source}</p><strong>{meta.name}</strong><span>{meta.description}</span></div>
        <span className={`capability-state${draft.enabled ? "" : " capability-state--idle"}`}><StatusDot tone={draft.enabled ? "ready" : "idle"} label={draft.enabled ? "Активно" : "Выключено"} />{draft.enabled ? "Активно" : "Выключено"}</span>
      </div>

      {capability === "hearing" ? (
        <section className="dictation-card" aria-label="Диктовка голосом">
          <div className="dictation-card__intro">
            <div className="dictation-card__icon"><IconMicrophone size={27} /></div>
            <div>
              <div className="dictation-card__eyebrow"><StatusDot tone={hearingUiState(draft.enabled, speechRuntime).tone} label={hearingUiState(draft.enabled, speechRuntime).label} />Диктовка · {hearingUiState(draft.enabled, speechRuntime).label}</div>
              <h2>Говорите — Sens вставит текст</h2>
              <p>Удерживайте <kbd>{formatHotkey(draft.hotkey)}</kbd>, говорите и отпустите клавиши. Текст появится в активном поле.</p>
            </div>
          </div>
          <div className="dictation-card__controls">
            <label className="setting-field"><span><IconKeyboard size={17} />Комбинация клавиш</span>
              <select value={draft.hotkey} onChange={(event) => update("hotkey", event.target.value)}>
                <option value="ctrl+win">Ctrl + Win · рекомендуется</option>
                <option value="ctrl+shift">Ctrl + Shift</option>
                <option value="alt+space">Alt + Space</option>
              </select>
            </label>
            {!speechRuntime?.running ? <button className="secondary-button start-speech" type="button" onClick={onStartSpeech}><IconPlayerPlay size={18} />Запустить слух</button> : <span className="speech-running"><StatusDot label="Служба диктовки запущена" />Служба работает в фоне</span>}
          </div>
          {speechRuntime?.error && !speechRuntime.running ? <p className="dictation-card__error">{speechRuntime.error}</p> : null}
        </section>
      ) : null}

      <div className="settings-grid">
        <section className="settings-group">
          <div className="settings-group__heading"><span>01</span><div><h2>Модель</h2><p>{capability === "sight" ? "Кто будет разбирать изображение" : "Кто будет распознавать аудио"}</p></div></div>
          {capability === "sight" ? (
            <>
              <label className="setting-field">Провайдер
                <select value={draft.provider} onChange={(event) => providerChanged(event.target.value)}>
                  {data.sightProviders.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              </label>
              <label className="setting-field">Vision-модель
                <input value={draft.model} onChange={(event) => update("model", event.target.value)} spellCheck="false" />
                <small>Можно указать любую совместимую модель выбранного провайдера.</small>
              </label>
            </>
          ) : (
            <>
              <label className="setting-field">Модель распознавания
                <select value={draft.model} onChange={(event) => update("model", event.target.value)}>
                  {data.hearingModels.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <small>{data.hearingModels.find((item) => item.value === draft.model)?.description}</small>
              </label>
              <label className="setting-field">Устройство
                <select value={draft.device} onChange={(event) => update("device", event.target.value)}>
                  <option value="auto">Автоматически</option><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option>
                </select>
              </label>
            </>
          )}
        </section>

        <section className="settings-group">
          <div className="settings-group__heading"><span>02</span><div><h2>Качество и ресурсы</h2><p>Баланс скорости и точности</p></div></div>
          {capability === "sight" ? (
            <>
              <label className="setting-field">Детализация
                <select value={draft.detail} onChange={(event) => update("detail", event.target.value)}>
                  <option value="quick">Quick · быстро</option><option value="normal">Normal · баланс</option><option value="deep">Deep · максимум деталей</option>
                </select>
              </label>
              <label className="setting-field">Максимум vision-вызовов
                <input type="number" min="1" max="32" value={draft.maxCallsPerImage} onChange={(event) => update("maxCallsPerImage", Number(event.target.value))} />
                <small>Жёсткий лимит на одну визуальную задачу.</small>
              </label>
            </>
          ) : (
            <>
              <label className="setting-field">Beam size
                <input type="number" min="1" max="10" value={draft.beamSize} onChange={(event) => update("beamSize", Number(event.target.value))} />
                <small>Больше — точнее, но медленнее.</small>
              </label>
              <label className="setting-field">Чувствительность к речи
                <select value={String(draft.vadSensitivity)} onChange={(event) => update("vadSensitivity", Number(event.target.value))}>
                  <option value="0.01">Высокая</option><option value="0.02">Сбалансированная</option><option value="0.04">Только уверенная речь</option>
                </select>
              </label>
            </>
          )}
        </section>
      </div>

      <section className="settings-toggles">
        <SettingsToggle label={`Доступ к ${meta.dative}`} description="Разрешить подключённым моделям использовать эту возможность" checked={draft.enabled} onChange={(value) => update("enabled", value)} />
        {capability === "sight" ? (
          <>
            <SettingsToggle label="Кэшировать результаты" description="Повторный анализ того же изображения не расходует vision-вызовы" checked={draft.cache} onChange={(value) => update("cache", value)} />
            <SettingsToggle label="Перепроверять ответ" description="Двойной проход: сверка результата с изображением и исправление ошибок (больше времени и расходов)" checked={draft.verify} onChange={(value) => update("verify", value)} />
          </>
        ) : (
          <>
            <SettingsToggle label="Предзагружать модель" description="Быстрее первый ответ, но больше памяти в фоне" checked={draft.preloadModel} onChange={(value) => update("preloadModel", value)} />
            <SettingsToggle label="Обрабатывать текст" description="Пунктуация, пробелы и очистка результата" checked={draft.postprocessText} onChange={(value) => update("postprocessText", value)} />
            <SettingsToggle label="Копировать в буфер обмена" description="Сохранять расшифровку, даже если активное поле недоступно" checked={draft.copyToClipboard} onChange={(value) => update("copyToClipboard", value)} />
            <SettingsToggle label="Вставлять в активное поле" description="После отпускания клавиш вставлять готовый текст туда, где находится курсор" checked={draft.pasteToActiveInput} onChange={(value) => update("pasteToActiveInput", value)} />
          </>
        )}
      </section>

      <div className="settings-actions">
        <span>{capability === "hearing" ? "Настройки диктовки применятся сразу после сохранения." : "Изменения применятся к следующей задаче."}</span>
        <button className="primary-button save-settings" type="button" disabled={saving} onClick={() => onSave(capability, draft)}><IconDeviceFloppy size={20} />{saving ? "Сохраняю…" : "Сохранить настройки"}</button>
      </div>
    </section>
  );
}

function UpdateCard({ state, version, onCheck, onInstall }) {
  const available = state.phase === "available";
  const installing = state.phase === "downloading" || state.phase === "installing";
  const statusCopy = {
    idle: "Sens проверит подписанный релиз на GitHub.",
    checking: "Проверяем канал обновлений…",
    current: "У вас последняя версия.",
    available: `Доступна версия ${state.version}.`,
    downloading: `Загрузка обновления${state.progress ? ` · ${state.progress}%` : ""}…`,
    installing: "Установка обновления…",
    error: state.error || "Не удалось проверить обновления.",
  }[state.phase] || "Sens проверит подписанный релиз на GitHub.";
  return (
    <section className="update-card" aria-label="Обновления Sens">
      <div className="update-card__icon"><IconDownload size={25} /></div>
      <div><p>Обновления</p><h3>Sens{version ? ` ${version}` : ""}</h3><span>{statusCopy}</span></div>
      {available ? (
        <button className="primary-button update-button" type="button" onClick={onInstall}><IconDownload size={18} />Обновить до {state.version}</button>
      ) : (
        <button className="secondary-button update-button" type="button" disabled={state.phase === "checking" || installing} onClick={onCheck}><IconRefresh size={18} />{state.phase === "checking" ? "Проверяю…" : "Проверить обновления"}</button>
      )}
    </section>
  );
}

function DetailContent({ view, settings, speechRuntime, runtimeStatus, updateState, onCheckUpdate, onInstallUpdate, onOpenCapability, openConnect }) {
  if (view === "capabilities") return <CapabilitiesContent settings={settings} speechRuntime={speechRuntime} onOpenCapability={onOpenCapability} />;
  const data = {
    integrations: [["Z-Code", "MCP · stdio", "Подключено", "15 инструментов Sens доступны модели"], ["Локальный broker", "Named pipe", "Готово", "Один процесс обслуживает все клиенты"]],
    console: [["Broker", "Rust", "Готово", "Среднее время ответа 14 мс"], ["Sight worker", "Node.js", "Ожидание", "Запускается только при запросе"], ["Hearing worker", "Python", "Ожидание", "Изолирован и завершает дочерние процессы"]],
    settings: [["Запускать вместе с Windows", "Система", "Включено", "Sens появляется в трее без открытия окна"], ["Локальная обработка", "Приватность", "Включено", "Данные не покидают выбранного провайдера"]],
    about: [["Sens", "Версия 1.1.0", "Локально", "Одна точка подключения для чувств модели"], ["Архитектура", "Rust + sidecars", "Расширяемо", "Новые чувства подключаются как модули"]],
  };
  return (
    <section className="detail-panel">
      <div className="detail-list">
        {(data[view] ?? []).map(([title, meta, status, description]) => (
          <article className="detail-card" key={title}>
            <div className="detail-card__icon">{view === "console" ? <IconActivity size={28} /> : <IconBox size={28} />}</div>
            <div><p>{meta}</p><h3>{title}</h3><span>{description}</span></div>
            <div className="detail-card__status"><StatusDot tone={status === "Ожидание" ? "idle" : "ready"} label={status} />{status}</div>
          </article>
        ))}
      </div>
      {view === "settings" ? <UpdateCard state={updateState} version={runtimeStatus?.version} onCheck={onCheckUpdate} onInstall={onInstallUpdate} /> : null}
      {view === "integrations" ? <button className="primary-button detail-cta" type="button" onClick={openConnect}><IconPlus size={20} />Добавить клиент</button> : null}
    </section>
  );
}

function TrayPanel({ minimized, onOpen, onDiagnostics, onQuit, runtimeStatus, runtimeError, speechRuntime, capabilitySettings }) {
  const runtimeReady = runtimeStatus?.state === "ready";
  const sight = runtimeStatus?.capabilities?.find((item) => item.id === "sight");
  const capabilityLabel = (capability, enabled) => !enabled ? "выключено" : capability?.state === "error" ? "ошибка" : "готово";
  const sightLabel = capabilityLabel(sight, capabilitySettings.sight.enabled);
  const hearingState = hearingUiState(capabilitySettings.hearing.enabled, speechRuntime);
  const systemTone = runtimeError || hearingState.tone === "attention" ? "attention" : "ready";
  return (
    <aside className="tray-panel" aria-label="Панель Sens в трее">
      <header><div className="tray-brand"><BrandMark tone="light" size={44} /><strong>Sens</strong></div><StatusDot tone={systemTone} label="Состояние Sens" /></header>
      <p className="tray-state">{runtimeError ? "Нужна диагностика" : minimized ? "Sens работает в фоне" : runtimeStatus && !runtimeReady ? "Sens запускается" : "Система готова"}</p>
      <div className="tray-separator" />
      <div className="tray-status-row"><StatusDot tone={sightLabel === "готово" ? "ready" : sightLabel === "выключено" ? "idle" : "attention"} label={`Зрение: ${sightLabel}`} /><span>Зрение · {sightLabel}</span></div>
      <div className="tray-status-row"><StatusDot tone={hearingState.tone} label={`Слух: ${hearingState.label}`} /><span>Слух · {hearingState.label}{speechRuntime?.running ? ` · ${formatHotkey(speechRuntime.hotkey || capabilitySettings.hearing.hotkey)}` : ""}</span></div>
      <div className="tray-separator" />
      <div className="tray-client-row"><span>Z-Code</span><StatusDot label="Z-Code подключён" /></div>
      <button className="tray-open" type="button" onClick={onOpen}>Открыть</button>
      <div className="tray-secondary-actions">
        <button className="tray-diagnostics" type="button" onClick={onDiagnostics}>Диагностика</button>
        <button className="tray-quit" type="button" onClick={onQuit}><IconPower size={16} />Выйти</button>
      </div>
    </aside>
  );
}

export function App() {
  const nativeRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  const trayView = nativeRuntime && new URLSearchParams(window.location.search).get("view") === "tray";
  const [view, setView] = useState("home");
  const [selectedCapability, setSelectedCapability] = useState(null);
  const [capabilitySettings, setCapabilitySettings] = useState(defaultCapabilitySettings);
  const [connectOpen, setConnectOpen] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);
  const [toast, setToast] = useState("");
  const [runtimeStatus, setRuntimeStatus] = useState(null);
  const [runtimeError, setRuntimeError] = useState("");
  const [speechRuntime, setSpeechRuntime] = useState(nativeRuntime ? null : {
    running: true,
    managed: true,
    enabled: true,
    hotkey: "ctrl+win",
    model: "gigaam",
    modelState: "ready",
  });
  const [updateState, setUpdateState] = useState({ phase: "idle", version: "", progress: 0, error: "" });
  const [pendingUpdate, setPendingUpdate] = useState(null);
  const copy = useMemo(() => selectedCapability ? [
    `Возможности / ${capabilityMeta[selectedCapability].name}`,
    `Настройка ${capabilityMeta[selectedCapability].genitive}`,
    capabilityMeta[selectedCapability].settingsDescription,
  ] : viewCopy[view] ?? viewCopy.home, [view, selectedCapability]);

  useEffect(() => {
    if (!nativeRuntime) return undefined;
    let active = true;
    const refresh = async () => {
      try {
        const status = await invoke("sens_status");
        if (active) {
          setRuntimeStatus(status);
          setRuntimeError("");
        }
      } catch (error) {
        if (active) setRuntimeError(String(error));
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => { active = false; window.clearInterval(timer); };
  }, [nativeRuntime]);

  useEffect(() => {
    if (!nativeRuntime) return;
    invoke("capability_settings")
      .then((settings) => setCapabilitySettings(settings))
      .catch((error) => setRuntimeError(String(error)));
  }, [nativeRuntime]);

  useEffect(() => {
    if (!nativeRuntime) return undefined;
    let active = true;
    const refreshSpeech = async () => {
      try {
        const status = await invoke("speech_runtime_status");
        if (active) setSpeechRuntime(status);
      } catch (error) {
        if (active) setSpeechRuntime((previous) => ({ ...previous, running: false, error: String(error) }));
      }
    };
    refreshSpeech();
    const timer = window.setInterval(refreshSpeech, 1500);
    return () => { active = false; window.clearInterval(timer); };
  }, [nativeRuntime]);

  useEffect(() => {
    if (!nativeRuntime || trayView) return undefined;
    let removeListener;
    listen("sens:navigate", (event) => {
      if (typeof event.payload === "string" && viewCopy[event.payload]) {
        setSelectedCapability(null);
        setView(event.payload);
      }
    }).then((unlisten) => { removeListener = unlisten; });
    return () => { removeListener?.(); };
  }, [nativeRuntime, trayView]);

  function showToast(message) { setToast(message); window.setTimeout(() => setToast(""), 2600); }
  function navigate(nextView) { setSelectedCapability(null); setView(nextView); }
  function openCapability(capability) { setSelectedCapability(capability); setView("capabilities"); }

  async function handleConnected(client) {
    try {
      if (nativeRuntime) await invoke("connect_client", { client });
      setConnectOpen(false);
      showToast(`${client} подключён к Sens`);
    } catch (error) {
      showToast(`Не удалось подключить ${client}: ${String(error)}`);
    }
  }

  async function saveSettings(capability, settings) {
    setSavingSettings(true);
    try {
      if (nativeRuntime) {
        const saved = await invoke("save_capability_settings", { capability, settings });
        setCapabilitySettings(saved);
      } else {
        setCapabilitySettings((previous) => ({ ...previous, [capability]: settings }));
      }
      showToast(`${capabilityMeta[capability].name}: настройки сохранены`);
    } catch (error) {
      showToast(`Не удалось сохранить: ${String(error)}`);
    } finally {
      setSavingSettings(false);
    }
  }

  async function startSpeech() {
    try {
      if (!nativeRuntime) return;
      const status = await invoke("start_speech_runtime");
      setSpeechRuntime(status);
      showToast("Слух запущен. Удерживайте Ctrl + Win, чтобы говорить.");
    } catch (error) {
      setSpeechRuntime((previous) => ({ ...previous, running: false, error: String(error) }));
      showToast(`Не удалось запустить слух: ${String(error)}`);
    }
  }

  async function checkForUpdates() {
    setUpdateState({ phase: "checking", version: "", progress: 0, error: "" });
    if (!nativeRuntime) {
      window.setTimeout(() => setUpdateState({ phase: "current", version: "", progress: 0, error: "" }), 350);
      return;
    }
    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (!update) {
        setPendingUpdate(null);
        setUpdateState({ phase: "current", version: "", progress: 0, error: "" });
        return;
      }
      setPendingUpdate(update);
      setUpdateState({ phase: "available", version: update.version, progress: 0, error: "" });
    } catch (error) {
      setUpdateState({ phase: "error", version: "", progress: 0, error: String(error) });
    }
  }

  async function installUpdate() {
    if (!pendingUpdate) return;
    let downloaded = 0;
    let total = 0;
    try {
      setUpdateState((previous) => ({ ...previous, phase: "downloading", progress: 0, error: "" }));
      await pendingUpdate.downloadAndInstall((event) => {
        if (event.event === "Started") total = event.data.contentLength || 0;
        if (event.event === "Progress") {
          downloaded += event.data.chunkLength;
          const progress = total ? Math.min(99, Math.round((downloaded / total) * 100)) : 0;
          setUpdateState((previous) => ({ ...previous, phase: "downloading", progress }));
        }
        if (event.event === "Finished") setUpdateState((previous) => ({ ...previous, phase: "installing", progress: 100 }));
      });
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (error) {
      setUpdateState((previous) => ({ ...previous, phase: "error", error: String(error) }));
    }
  }

  const performWindowAction = (action) => {
    if (nativeRuntime) return invoke("window_action", { action });
    if (action === "minimize" || action === "hide" || action === "close") setMinimized(true);
    return Promise.resolve();
  };

  if (trayView) {
    return (
      <main className="native-tray-stage">
        <TrayPanel
          minimized
          runtimeStatus={runtimeStatus}
          runtimeError={runtimeError}
          speechRuntime={speechRuntime}
          capabilitySettings={capabilitySettings}
          onOpen={() => invoke("show_main", { view: "home" })}
          onDiagnostics={() => invoke("show_main", { view: "console" })}
          onQuit={() => invoke("quit_app")}
        />
      </main>
    );
  }

  return (
    <main className={`showcase-stage${nativeRuntime ? " showcase-stage--native" : ""}`}>
      {nativeRuntime ? null : <TrayPanel minimized={minimized} runtimeStatus={runtimeStatus} runtimeError={runtimeError} speechRuntime={speechRuntime} capabilitySettings={capabilitySettings} onOpen={() => setMinimized(false)} onDiagnostics={() => { setMinimized(false); navigate("console"); }} onQuit={() => setMinimized(true)} />}
      <section className={`app-window${minimized ? " app-window--minimized" : ""}`} aria-label="Sens">
        <nav className="side-rail" aria-label="Навигация Sens">
          <button className="rail-logo" type="button" onClick={() => navigate("home")} aria-label="Sens — главная"><BrandMark tone="dark" size={52} /></button>
          <div className="rail-nav">
            {navItems.map(({ id, label, icon: Icon }) => <button key={id} type="button" className={view === id ? "is-active" : ""} onClick={() => navigate(id)} aria-label={label} title={label}><Icon size={27} stroke={1.8} /></button>)}
          </div>
        </nav>
        <div className="app-surface">
          <header className="titlebar" data-tauri-drag-region>
            <div data-tauri-drag-region><span className="titlebar-product" data-tauri-drag-region>Sens</span><span className="titlebar-view" data-tauri-drag-region>{navItems.find((item) => item.id === view)?.label}</span></div>
            <WindowControls onMinimize={() => performWindowAction("minimize")} onMaximize={() => performWindowAction("maximize")} onClose={() => performWindowAction("hide")} />
          </header>
          <div className={`content-shell${view === "home" ? " content-shell--home" : ""}${selectedCapability ? " content-shell--capability" : ""}`}>
            {view === "home" ? null : <p className="section-kicker">{copy[0]}</p>}<h1>{copy[1]}</h1>{view === "home" ? null : <p className="view-description">{copy[2]}</p>}
            {view === "home" ? (
              <HomeContent settings={capabilitySettings} speechRuntime={speechRuntime} openCapability={openCapability} openCapabilities={() => navigate("capabilities")} openConnect={() => setConnectOpen(true)} />
            ) : selectedCapability ? (
              <CapabilitySettingsContent capability={selectedCapability} data={capabilitySettings} speechRuntime={speechRuntime} onStartSpeech={startSpeech} onBack={() => setSelectedCapability(null)} onSave={saveSettings} saving={savingSettings} />
            ) : (
              <DetailContent view={view} settings={capabilitySettings} speechRuntime={speechRuntime} runtimeStatus={runtimeStatus} updateState={updateState} onCheckUpdate={checkForUpdates} onInstallUpdate={installUpdate} onOpenCapability={openCapability} openConnect={() => setConnectOpen(true)} />
            )}
          </div>
        </div>
      </section>
      {minimized ? <button className="restore-window" type="button" onClick={() => setMinimized(false)}>Окно Sens свёрнуто — открыть</button> : null}
      {connectOpen ? <ConnectModal onClose={() => setConnectOpen(false)} onConnected={handleConnected} /> : null}
      {toast ? <div className="toast" role="status">{toast}</div> : null}
    </main>
  );
}
