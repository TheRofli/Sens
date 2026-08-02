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
  IconLanguage,
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
import { useLanguage, useT } from "./i18n.js";

const navItems = [
  { id: "home", icon: IconHome },
  { id: "capabilities", icon: IconBox },
  { id: "integrations", icon: IconPuzzle },
  { id: "console", icon: IconPrompt },
  { id: "settings", icon: IconSettings },
  { id: "about", icon: IconInfoCircle },
];

const KNOWN_PROVIDERS = ["mimo", "openai", "custom"];
const KNOWN_MODELS = ["parakeet", "whisper-ru", "gigaam"];

function providerDisplay(t, value, fallback = "") {
  return KNOWN_PROVIDERS.includes(value) ? t(`provider.${value}`) : fallback || value;
}

function modelDisplay(t, value, fallback = "") {
  return KNOWN_MODELS.includes(value) ? t(`model.${value}`) : fallback || value;
}

function modelDescription(t, value, fallback = "") {
  return KNOWN_MODELS.includes(value) ? t(`model.${value}.desc`) : fallback;
}

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
    source: "Eye",
    icon: IconEye,
    nameKey: "cap.sight.name",
    descriptionKey: "cap.sight.description",
    settingsDescriptionKey: "cap.sight.settingsDescription",
    openSettingsKey: "cap.sight.openSettings",
    accessKey: "cap.sight.access",
    pageKickerKey: "cap.sight.pageKicker",
    pageTitleKey: "cap.sight.pageTitle",
  },
  hearing: {
    source: "Speech",
    icon: IconHeadphones,
    nameKey: "cap.hearing.name",
    descriptionKey: "cap.hearing.description",
    settingsDescriptionKey: "cap.hearing.settingsDescription",
    openSettingsKey: "cap.hearing.openSettings",
    accessKey: "cap.hearing.access",
    pageKickerKey: "cap.hearing.pageKicker",
    pageTitleKey: "cap.hearing.pageTitle",
  },
};

function BrandMark({ tone = "dark", size = 48 }) {
  return <img className={`brand-mark brand-mark--${tone}`} src="/assets/sens-mark-source.png" width={size} height={size} alt="Sens" draggable="false" />;
}

function StatusDot({ tone = "ready", label }) {
  return <span className={`status-dot status-dot--${tone}`} aria-label={label}><IconPointFilled size={16} stroke={0} aria-hidden="true" /></span>;
}

function WindowControls({ onMinimize, onMaximize, onClose }) {
  const t = useT();
  return (
    <div className="window-controls" aria-label={t("win.minimize")}>
      <button type="button" onClick={onMinimize} aria-label={t("win.minimize")}><IconMinus size={19} /></button>
      <button type="button" onClick={onMaximize} aria-label={t("win.maximize")}><IconSquare size={16} /></button>
      <button type="button" onClick={onClose} aria-label={t("win.close")}><IconX size={20} /></button>
    </div>
  );
}

function hearingUiState(enabled, runtime) {
  if (!enabled) return { labelKey: "state.off", tone: "idle" };
  if (runtime?.transcribing) return { labelKey: "state.listening", tone: "attention" };
  if (!runtime?.running) return { labelKey: "state.needStart", tone: "attention" };
  if (!runtime?.enabled) return { labelKey: "state.needEnable", tone: "attention" };
  return { labelKey: "state.ready", tone: "ready" };
}

function formatHotkey(value = "ctrl+win") {
  const labels = { ctrl: "Ctrl", win: "Win", shift: "Shift", alt: "Alt", space: "Space" };
  return value.split("+").map((part) => labels[part] || part).join(" + ");
}

function CapabilityPill({ capability, enabled, status, onClick }) {
  const t = useT();
  const meta = capabilityMeta[capability];
  const state = status || { labelKey: enabled ? "state.ready" : "state.off", tone: enabled ? "ready" : "idle" };
  return (
    <button type="button" className="capability-pill" onClick={onClick} aria-label={t(meta.openSettingsKey)}>
      <StatusDot tone={state.tone} label={`${t(meta.nameKey)}: ${t(state.labelKey)}`} />
      <span className="capability-pill__copy"><strong>{t(meta.nameKey)}</strong><small>{t(state.labelKey)}</small></span>
    </button>
  );
}

function ConnectModal({ onClose, onConnected }) {
  const t = useT();
  const [client, setClient] = useState("Z-Code");
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="connect-modal" role="dialog" aria-modal="true" aria-labelledby="connect-title" onMouseDown={(event) => event.stopPropagation()}>
        <button className="modal-close" type="button" onClick={onClose} aria-label={t("modal.close")}><IconX size={22} /></button>
        <p className="section-kicker">{t("modal.kicker")}</p>
        <h2 id="connect-title">{t("modal.title")}</h2>
        <p>{t("modal.desc")}</p>
        <label htmlFor="client">{t("modal.clientLabel")}</label>
        <select id="client" value={client} onChange={(event) => setClient(event.target.value)}>
          <option>Z-Code</option><option>Claude Desktop</option><option>Cursor</option><option>{t("modal.otherClient")}</option>
        </select>
        <button className="primary-button" type="button" onClick={() => onConnected(client)}><IconPlugConnected size={20} />{t("modal.connect", { client })}</button>
      </section>
    </div>
  );
}

function HomeContent({ settings, speechRuntime, openCapability, openCapabilities, openConnect }) {
  const t = useT();
  const hearingState = hearingUiState(settings.hearing.enabled, speechRuntime);
  return (
    <>
      <section className="home-grid" aria-label={t("view.home.title")}>
        <button className="connect-banner" type="button" onClick={openConnect}>
          <IconPlus size={31} stroke={1.7} /><span>{t("home.connectCta")}</span>
        </button>
        <section className="capability-chamber">
          <p>{t("home.activeCapabilities")}</p>
          <div className="capability-row">
            <CapabilityPill capability="sight" enabled={settings.sight.enabled} onClick={() => openCapability("sight")} />
            <span className="capability-divider" />
            <CapabilityPill capability="hearing" enabled={settings.hearing.enabled} status={hearingState} onClick={() => openCapability("hearing")} />
            <span className="capability-divider" />
            <button className="capability-overview" type="button" onClick={openCapabilities}><IconAdjustmentsHorizontal size={19} />{t("home.allCapabilities")}</button>
          </div>
        </section>
        <aside className="connection-card">
          <p>{t("home.currentConnection")}</p><strong>Z-Code</strong>
          <div className="connection-line"><StatusDot tone="attention" label="MCP активен" /><span>MCP</span></div>
          <div className="connection-line"><StatusDot label={t("home.local")} /><span>{t("home.local")}</span></div>
        </aside>
      </section>
      <footer className="activity-bar"><IconClock size={28} stroke={1.8} /><span>{t("home.lastAction")}</span></footer>
    </>
  );
}

function CapabilityCard({ capability, enabled, settings, runtimeStatus, onOpen }) {
  const t = useT();
  const meta = capabilityMeta[capability];
  const Icon = meta.icon;
  const summary = capability === "sight"
    ? `${providerDisplay(t, settings.provider, settings.provider)} · ${settings.model}`
    : `${modelDisplay(t, settings.model, settings.model)} · ${settings.device.toUpperCase()}`;
  const state = capability === "hearing"
    ? hearingUiState(enabled, runtimeStatus)
    : { labelKey: enabled ? "state.readyUpper" : "state.offUpper", tone: enabled ? "ready" : "idle" };
  return (
    <button className="detail-card capability-card" type="button" onClick={onOpen} aria-label={t(meta.openSettingsKey)}>
      <div className="detail-card__icon"><Icon size={28} /></div>
      <div><p>{meta.source} · {summary}</p><h3>{t(meta.nameKey)}</h3><span>{t(meta.descriptionKey)}</span></div>
      <div className="capability-card__tail">
        <span className="detail-card__status"><StatusDot tone={state.tone} label={t(state.labelKey)} />{t(state.labelKey)}</span>
        <IconChevronRight size={22} aria-hidden="true" />
      </div>
    </button>
  );
}

function CapabilitiesContent({ settings, speechRuntime, onOpenCapability }) {
  const t = useT();
  return (
    <section className="detail-panel" aria-label={t("view.capabilities.title")}>
      <div className="detail-list">
        <CapabilityCard capability="sight" enabled={settings.sight.enabled} settings={settings.sight} onOpen={() => onOpenCapability("sight")} />
        <CapabilityCard capability="hearing" enabled={settings.hearing.enabled} settings={settings.hearing} runtimeStatus={speechRuntime} onOpen={() => onOpenCapability("hearing")} />
      </div>
      <aside className="future-sense-card">
        <div className="future-sense-card__icon"><IconSparkles size={24} /></div>
        <div><strong>{t("future.title")}</strong><span>{t("future.desc")}</span></div>
        <span className="future-sense-card__badge">{t("future.badge")}</span>
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
  const t = useT();
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
    <section className="capability-settings" aria-label={t(meta.openSettingsKey)}>
      <button className="back-button" type="button" onClick={onBack}><IconArrowLeft size={19} />{t("settings.back")}</button>
      <div className="capability-summary">
        <div className="capability-summary__icon"><Icon size={34} /></div>
        <div><p>{meta.source}</p><strong>{t(meta.nameKey)}</strong><span>{t(meta.descriptionKey)}</span></div>
        <span className={`capability-state${draft.enabled ? "" : " capability-state--idle"}`}><StatusDot tone={draft.enabled ? "ready" : "idle"} label={draft.enabled ? t("state.active") : t("state.offUpper")} />{draft.enabled ? t("state.active") : t("state.offUpper")}</span>
      </div>

      {capability === "hearing" ? (
        <section className="dictation-card" aria-label={t("dictation.title")}>
          <div className="dictation-card__intro">
            <div className="dictation-card__icon"><IconMicrophone size={27} /></div>
            <div>
              <div className="dictation-card__eyebrow"><StatusDot tone={hearingUiState(draft.enabled, speechRuntime).tone} label={hearingUiState(draft.enabled, speechRuntime).labelKey} />{t("dictation.kicker", { state: t(hearingUiState(draft.enabled, speechRuntime).labelKey) })}</div>
              <h2>{t("dictation.title")}</h2>
              <p>{t("dictation.desc", { hotkey: formatHotkey(draft.hotkey) })}</p>
            </div>
          </div>
          <div className="dictation-card__controls">
            <label className="setting-field"><span><IconKeyboard size={17} />{t("dictation.hotkeyLabel")}</span>
              <select value={draft.hotkey} onChange={(event) => update("hotkey", event.target.value)}>
                <option value="ctrl+win">{t("dictation.hotkeyRecommended")}</option>
                <option value="ctrl+shift">Ctrl + Shift</option>
                <option value="alt+space">Alt + Space</option>
              </select>
            </label>
            {!speechRuntime?.running ? <button className="secondary-button start-speech" type="button" onClick={onStartSpeech}><IconPlayerPlay size={18} />{t("dictation.start")}</button> : <span className="speech-running"><StatusDot label={t("dictation.running")} />{t("dictation.running")}</span>}
          </div>
          {speechRuntime?.error && !speechRuntime.running ? <p className="dictation-card__error">{speechRuntime.error}</p> : null}
        </section>
      ) : null}

      <div className="settings-grid">
        <section className="settings-group">
          <div className="settings-group__heading"><span>01</span><div><h2>{t("group.model")}</h2><p>{t(capability === "sight" ? "group.model.sightSub" : "group.model.hearingSub")}</p></div></div>
          {capability === "sight" ? (
            <>
              <label className="setting-field">{t("field.provider")}
                <select value={draft.provider} onChange={(event) => providerChanged(event.target.value)}>
                  {data.sightProviders.map((option) => <option key={option.value} value={option.value}>{providerDisplay(t, option.value, option.label)}</option>)}
                </select>
              </label>
              <label className="setting-field">{t("field.visionModel")}
                <input value={draft.model} onChange={(event) => update("model", event.target.value)} spellCheck="false" />
                <small>{t("field.visionModelHint")}</small>
              </label>
            </>
          ) : (
            <>
              <label className="setting-field">{t("field.recognitionModel")}
                <select value={draft.model} onChange={(event) => update("model", event.target.value)}>
                  {data.hearingModels.map((option) => <option key={option.value} value={option.value}>{modelDisplay(t, option.value, option.label)}</option>)}
                </select>
                <small>{modelDescription(t, draft.model, data.hearingModels.find((item) => item.value === draft.model)?.description)}</small>
              </label>
              <label className="setting-field">{t("field.device")}
                <select value={draft.device} onChange={(event) => update("device", event.target.value)}>
                  <option value="auto">{t("device.auto")}</option><option value="cpu">{t("device.cpu")}</option><option value="cuda">{t("device.cuda")}</option>
                </select>
              </label>
            </>
          )}
        </section>

        <section className="settings-group">
          <div className="settings-group__heading"><span>02</span><div><h2>{t("group.quality")}</h2><p>{t("group.quality.sub")}</p></div></div>
          {capability === "sight" ? (
            <>
              <label className="setting-field">{t("field.detail")}
                <select value={draft.detail} onChange={(event) => update("detail", event.target.value)}>
                  <option value="quick">{t("detail.quick")}</option><option value="normal">{t("detail.normal")}</option><option value="deep">{t("detail.deep")}</option>
                </select>
              </label>
              <label className="setting-field">{t("field.maxCalls")}
                <input type="number" min="1" max="32" value={draft.maxCallsPerImage} onChange={(event) => update("maxCallsPerImage", Number(event.target.value))} />
                <small>{t("maxCalls.hint")}</small>
              </label>
            </>
          ) : (
            <>
              <label className="setting-field">{t("field.beam")}
                <input type="number" min="1" max="10" value={draft.beamSize} onChange={(event) => update("beamSize", Number(event.target.value))} />
                <small>{t("beam.hint")}</small>
              </label>
              <label className="setting-field">{t("field.vad")}
                <select value={String(draft.vadSensitivity)} onChange={(event) => update("vadSensitivity", Number(event.target.value))}>
                  <option value="0.01">{t("vad.high")}</option><option value="0.02">{t("vad.balanced")}</option><option value="0.04">{t("vad.confident")}</option>
                </select>
              </label>
            </>
          )}
        </section>
      </div>

      <section className="settings-toggles">
        <SettingsToggle label={t(meta.accessKey)} description={t("toggle.accessDesc")} checked={draft.enabled} onChange={(value) => update("enabled", value)} />
        {capability === "sight" ? (
          <>
            <SettingsToggle label={t("toggle.cache")} description={t("toggle.cacheDesc")} checked={draft.cache} onChange={(value) => update("cache", value)} />
            <SettingsToggle label={t("toggle.verify")} description={t("toggle.verifyDesc")} checked={draft.verify} onChange={(value) => update("verify", value)} />
          </>
        ) : (
          <>
            <SettingsToggle label={t("toggle.preload")} description={t("toggle.preloadDesc")} checked={draft.preloadModel} onChange={(value) => update("preloadModel", value)} />
            <SettingsToggle label={t("toggle.postprocess")} description={t("toggle.postprocessDesc")} checked={draft.postprocessText} onChange={(value) => update("postprocessText", value)} />
            <SettingsToggle label={t("toggle.clipboard")} description={t("toggle.clipboardDesc")} checked={draft.copyToClipboard} onChange={(value) => update("copyToClipboard", value)} />
            <SettingsToggle label={t("toggle.paste")} description={t("toggle.pasteDesc")} checked={draft.pasteToActiveInput} onChange={(value) => update("pasteToActiveInput", value)} />
          </>
        )}
      </section>

      <div className="settings-actions">
        <span>{t(capability === "hearing" ? "actions.hintHearing" : "actions.hintSight")}</span>
        <button className="primary-button save-settings" type="button" disabled={saving} onClick={() => onSave(capability, draft)}><IconDeviceFloppy size={20} />{saving ? t("actions.saving") : t("actions.save")}</button>
      </div>
    </section>
  );
}

function UpdateCard({ state, version, onCheck, onInstall }) {
  const t = useT();
  const available = state.phase === "available";
  const installing = state.phase === "downloading" || state.phase === "installing";
  const statusCopy = {
    idle: t("update.idle"),
    checking: t("update.checkingStatus"),
    current: t("update.current"),
    available: t("update.available", { version: state.version }),
    downloading: t("update.downloading", { progress: state.progress ? ` · ${state.progress}%` : "" }),
    installing: t("update.installing"),
    error: state.error || t("update.errorFallback"),
  }[state.phase] || t("update.idle");
  return (
    <section className="update-card" aria-label={t("update.label")}>
      <div className="update-card__icon"><IconDownload size={25} /></div>
      <div><p>{t("update.label")}</p><h3>Sens{version ? ` ${version}` : ""}</h3><span>{statusCopy}</span></div>
      {available ? (
        <button className="primary-button update-button" type="button" onClick={onInstall}><IconDownload size={18} />{t("update.install", { version: state.version })}</button>
      ) : (
        <button className="secondary-button update-button" type="button" disabled={state.phase === "checking" || installing} onClick={onCheck}><IconRefresh size={18} />{state.phase === "checking" ? t("update.checking") : t("update.check")}</button>
      )}
    </section>
  );
}

function LanguageCard() {
  const t = useT();
  const { lang, setLang } = useLanguage();
  return (
    <section className="language-card" aria-label={t("lang.label")}>
      <div className="language-card__icon"><IconLanguage size={24} /></div>
      <div><p>{t("lang.label")}</p><h3>{lang === "ru" ? "Русский" : "English"}</h3></div>
      <select value={lang} onChange={(event) => setLang(event.target.value)} aria-label={t("lang.label")}>
        <option value="ru">{t("lang.ru")}</option>
        <option value="en">{t("lang.en")}</option>
      </select>
    </section>
  );
}

function DetailContent({ view, settings, speechRuntime, runtimeStatus, updateState, onCheckUpdate, onInstallUpdate, onOpenCapability, openConnect }) {
  const t = useT();
  if (view === "capabilities") return <CapabilitiesContent settings={settings} speechRuntime={speechRuntime} onOpenCapability={onOpenCapability} />;
  const appVersion = runtimeStatus?.version;
  const rows = {
    integrations: [
      ["integ.zcode.title", "integ.zcode.meta", "status.connected", "integ.zcode.desc"],
      ["integ.broker.title", "integ.broker.meta", "status.ready", "integ.broker.desc"],
    ],
    console: [
      ["console.broker.title", "console.broker.meta", "status.ready", "console.broker.desc"],
      ["console.sight.title", "console.sight.meta", "status.waiting", "console.sight.desc"],
      ["console.hearing.title", "console.hearing.meta", "status.waiting", "console.hearing.desc"],
    ],
    settings: [
      ["settings.autostart.title", "settings.autostart.meta", "status.enabled", "settings.autostart.desc"],
      ["settings.privacy.title", "settings.privacy.meta", "status.enabled", "settings.privacy.desc"],
    ],
    about: [
      ["about.sens.title", "about.sens.meta", "home.local", "about.sens.desc"],
      ["about.arch.title", "about.arch.meta", "status.ready", "about.arch.desc"],
    ],
  };
  return (
    <section className="detail-panel">
      <div className="detail-list">
        {(rows[view] ?? []).map(([titleKey, metaKey, statusKey, descriptionKey]) => (
          <article className="detail-card" key={titleKey}>
            <div className="detail-card__icon">{view === "console" ? <IconActivity size={28} /> : <IconBox size={28} />}</div>
            <div><p>{t(metaKey)}</p><h3>{t(titleKey)}</h3><span>{t(descriptionKey)}</span></div>
            <div className="detail-card__status"><StatusDot tone={statusKey === "status.waiting" ? "idle" : "ready"} label={t(statusKey)} />{t(statusKey)}</div>
          </article>
        ))}
      </div>
      {view === "settings" ? (
        <>
          <LanguageCard />
          <UpdateCard state={updateState} version={runtimeStatus?.version} onCheck={onCheckUpdate} onInstall={onInstallUpdate} />
        </>
      ) : null}
      {view === "integrations" ? <button className="primary-button detail-cta" type="button" onClick={openConnect}><IconPlus size={20} />{t("integ.addClient")}</button> : null}
    </section>
  );
}

function TrayPanel({ minimized, onOpen, onDiagnostics, onQuit, runtimeStatus, runtimeError, speechRuntime, capabilitySettings }) {
  const t = useT();
  const runtimeReady = runtimeStatus?.state === "ready";
  const sight = runtimeStatus?.capabilities?.find((item) => item.id === "sight");
  const capabilityLabel = (capability, enabled) => !enabled ? t("state.off") : capability?.state === "error" ? t("state.error") : t("state.ready");
  const sightLabel = capabilityLabel(sight, capabilitySettings.sight.enabled);
  const hearingState = hearingUiState(capabilitySettings.hearing.enabled, speechRuntime);
  const systemTone = runtimeError || hearingState.tone === "attention" ? "attention" : "ready";
  return (
    <aside className="tray-panel" aria-label={t("tray.systemReady")}>
      <header><div className="tray-brand"><BrandMark tone="light" size={44} /><strong>Sens</strong></div><StatusDot tone={systemTone} label={t("tray.systemReady")} /></header>
      <p className="tray-state">{runtimeError ? t("tray.diagnostics") : minimized ? t("tray.background") : runtimeStatus && !runtimeReady ? t("tray.starting") : t("tray.systemReady")}</p>
      <div className="tray-separator" />
      <div className="tray-status-row"><StatusDot tone={sightLabel === t("state.ready") ? "ready" : sightLabel === t("state.off") ? "idle" : "attention"} label={`${t("tray.vision")}: ${sightLabel}`} /><span>{t("tray.vision")} · {sightLabel}</span></div>
      <div className="tray-status-row"><StatusDot tone={hearingState.tone} label={`${t("tray.hearing")}: ${t(hearingState.labelKey)}`} /><span>{t("tray.hearing")} · {t(hearingState.labelKey)}{speechRuntime?.running ? ` · ${formatHotkey(speechRuntime.hotkey || capabilitySettings.hearing.hotkey)}` : ""}</span></div>
      <div className="tray-separator" />
      <div className="tray-client-row"><span>{t("tray.zcode")}</span><StatusDot label={t("tray.zcode")} /></div>
      <button className="tray-open" type="button" onClick={onOpen}>{t("tray.open")}</button>
      <div className="tray-secondary-actions">
        <button className="tray-diagnostics" type="button" onClick={onDiagnostics}>{t("tray.diagnostics")}</button>
        <button className="tray-quit" type="button" onClick={onQuit}><IconPower size={16} />{t("tray.quit")}</button>
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
  const t = useT();
  const copy = useMemo(() => selectedCapability ? [
    t(capabilityMeta[selectedCapability].pageKickerKey),
    t(capabilityMeta[selectedCapability].pageTitleKey),
    t(capabilityMeta[selectedCapability].settingsDescriptionKey),
  ] : [t(`view.${view}.kicker`), t(`view.${view}.title`), t(`view.${view}.desc`)], [view, selectedCapability, t]);

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
      if (typeof event.payload === "string" && ["home", "capabilities", "integrations", "console", "settings", "about"].includes(event.payload)) {
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
      showToast(t("toast.connected", { client }));
    } catch (error) {
      showToast(t("toast.connectFailed", { client, error: String(error) }));
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
      showToast(t("toast.saved", { name: t(capabilityMeta[capability].nameKey) }));
    } catch (error) {
      showToast(t("toast.saveFailed", { error: String(error) }));
    } finally {
      setSavingSettings(false);
    }
  }

  async function startSpeech() {
    try {
      if (!nativeRuntime) return;
      const status = await invoke("start_speech_runtime");
      setSpeechRuntime(status);
      showToast(t("toast.speechStarted"));
    } catch (error) {
      setSpeechRuntime((previous) => ({ ...previous, running: false, error: String(error) }));
      showToast(t("toast.speechFailed", { error: String(error) }));
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
      // Stop the broker first so the installer can replace sens-broker.exe;
      // the relaunch brings a fresh broker from the new version.
      if (nativeRuntime) invoke("stop_broker").catch(() => {});
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

  const content = trayView ? (
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
  ) : (
    <main className={`showcase-stage${nativeRuntime ? " showcase-stage--native" : ""}`}>
      {nativeRuntime ? null : <TrayPanel minimized={minimized} runtimeStatus={runtimeStatus} runtimeError={runtimeError} speechRuntime={speechRuntime} capabilitySettings={capabilitySettings} onOpen={() => setMinimized(false)} onDiagnostics={() => { setMinimized(false); navigate("console"); }} onQuit={() => setMinimized(true)} />}
      <section className={`app-window${minimized ? " app-window--minimized" : ""}`} aria-label="Sens">
        <nav className="side-rail" aria-label="Sens">
          <button className="rail-logo" type="button" onClick={() => navigate("home")} aria-label="Sens — home"><BrandMark tone="dark" size={52} /></button>
          <div className="rail-nav">
            {navItems.map(({ id, icon: Icon }) => <button key={id} type="button" className={view === id ? "is-active" : ""} onClick={() => navigate(id)} aria-label={t(`nav.${id}`)} title={t(`nav.${id}`)}><Icon size={27} stroke={1.8} /></button>)}
          </div>
        </nav>
        <div className="app-surface">
          <header className="titlebar" data-tauri-drag-region>
            <div data-tauri-drag-region><span className="titlebar-product" data-tauri-drag-region>Sens</span><span className="titlebar-view" data-tauri-drag-region>{navItems.find((item) => item.id === view) ? t(`nav.${view}`) : ""}</span></div>
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
      {minimized ? <button className="restore-window" type="button" onClick={() => setMinimized(false)}>{t("restore.window")}</button> : null}
      {connectOpen ? <ConnectModal onClose={() => setConnectOpen(false)} onConnected={handleConnected} /> : null}
      {toast ? <div className="toast" role="status">{toast}</div> : null}
    </main>
  );

  return content;
}
