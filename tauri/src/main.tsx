import React, { useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

// -----------------------------------------------------------------------------
// Types (mirrors the Rust commands / Python /api/* payloads)
// -----------------------------------------------------------------------------

type StatusSnapshot = {
  running: boolean;
  engineEnabled: boolean;
  modelState: string;
  model: string;
  modelLabel: string;
  modelLoaded: boolean;
  modelLoading: boolean;
  modelInstalled: boolean;
  modelSizeLabel: string;
  transcribing: boolean;
  device: string;
  backend: string;
  statusText: string;
  historyCount: number;
  speechRoot: string;
  modelSnapshot: string;
};

type ModelInfo = {
  key: string;
  label: string;
  engine: string;
  modelId: string;
  description: string;
  installed: boolean;
  sizeLabel: string;
  active: boolean;
};

type HistoryItem = {
  id: string;
  text: string;
};

type SettingsValues = {
  model: string;
  engine_enabled: boolean;
  copy_to_clipboard: boolean;
  paste_to_active_input: boolean;
  preload_model: boolean;
  device: string;
  backend: string;
  hotkey: string;
  beam_size: number;
  temperature: number;
  repetition_penalty: number;
  no_repeat_ngram_size: number;
  compression_ratio_threshold: number;
  log_prob_threshold: number;
  vad_sensitivity: number;
  postprocess_text: boolean;
};

type Section = "status" | "models" | "settings" | "history";

// -----------------------------------------------------------------------------
// App shell (Layout B: sidebar + main)
// -----------------------------------------------------------------------------

const offlineSnapshot: StatusSnapshot = {
  running: false,
  engineEnabled: false,
  modelState: "stopped",
  model: "parakeet",
  modelLabel: "Speech core offline",
  modelLoaded: false,
  modelLoading: false,
  modelInstalled: false,
  modelSizeLabel: "Not installed",
  transcribing: false,
  device: "cpu",
  backend: "auto",
  statusText: "Start Speech in a terminal with: speech",
  historyCount: 0,
  speechRoot: "",
  modelSnapshot: "",
};

const navItems: Array<{ id: Section; label: string; short: string }> = [
  { id: "status", label: "Статус", short: "Сейчас" },
  { id: "models", label: "Модели", short: "Движок" },
  { id: "settings", label: "Настройки", short: "Парам." },
  { id: "history", label: "История", short: "Текст" },
];

function App() {
  const [snapshot, setSnapshot] = useState<StatusSnapshot>(offlineSnapshot);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [settings, setSettings] = useState<SettingsValues | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>("");
  const [filter, setFilter] = useState("");
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);
  const [installingKey, setInstallingKey] = useState<string | null>(null);
  const [section, setSection] = useState<Section>(getInitialSection());

  async function refresh() {
    try {
      const [nextSnapshot, nextModels, nextSettings, nextHistory] = await Promise.all([
        invoke<StatusSnapshot>("app_snapshot"),
        invoke<ModelInfo[]>("get_models").catch(() => []),
        invoke<SettingsValues>("get_settings").catch(() => null),
        invoke<HistoryItem[]>("recent_history", { limit: 80 }).catch(() => []),
      ]);
      setSnapshot(nextSnapshot);
      setModels(nextModels);
      if (nextSettings) setSettings(nextSettings);
      setHistory(nextHistory);
      setSelectedHistoryId((current) => current || nextHistory[0]?.id || "");
      // Clear installing flag once the model reports installed.
      if (nextModels.length && installingKey) {
        const m = nextModels.find((x) => x.key === installingKey);
        if (m && m.installed) setInstallingKey(null);
      }
    } catch {
      setSnapshot(offlineSnapshot);
    }
  }

  async function runWithToast<T>(
    fn: () => Promise<T>,
    okMessage: string,
    failMessage = "Действие не удалось",
  ): Promise<T | undefined> {
    setBusy(true);
    try {
      const result = await fn();
      showToast(okMessage);
      await refresh();
      return result;
    } catch {
      showToast(failMessage);
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(""), 1800);
  }

  function chooseSection(next: Section) {
    setSection(next);
    window.location.hash = next;
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [installingKey]);

  const selectedHistory =
    history.find((item) => item.id === selectedHistoryId) || history[0];
  const filteredHistory = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) return history;
    return history.filter((item) => item.text.toLowerCase().includes(query));
  }, [filter, history]);

  return (
    <main className="app-shell">
      <aside className="shelf">
        <Brand />
        <nav className="nav-list" aria-label="Разделы Speech">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={section === item.id ? "active" : ""}
              onClick={() => chooseSection(item.id)}
            >
              <span>{item.label}</span>
              <small>{item.short}</small>
            </button>
          ))}
        </nav>
        <div className="model-ticket">
          <span>Активная модель</span>
          <strong>{snapshot.modelLabel || "—"}</strong>
          <small>
            {snapshot.modelLoading
              ? "загрузка…"
              : snapshot.modelLoaded
                ? snapshot.modelSizeLabel || "загружена"
                : snapshot.modelInstalled
                  ? "не загружена"
                  : "не установлена"}
          </small>
        </div>
      </aside>

      <section className="stage">
        <header className="topbar">
          <div>
            <p className="eyebrow">Локальная диктовка</p>
            <h1>{headlineFor(snapshot)}</h1>
          </div>
          <div className="topbar-actions">
            <StatusPill running={snapshot.running} modelState={snapshot.modelState} />
          </div>
        </header>

        <section className="page" key={section}>
          {section === "status" && (
            <StatusSection
              snapshot={snapshot}
              latest={history[0]}
              onCopyLatest={() =>
                latestCopy(history[0], () =>
                  runWithToast(() => invoke("copy_last"), "Скопировано"),
                )
              }
            />
          )}

          {section === "models" && (
            <ModelsSection
              models={models}
              snapshot={snapshot}
              installingKey={installingKey}
              busy={busy}
              onSelect={(key) =>
                runWithToast(
                  () => invoke("select_model", { key }),
                  "Модель выбрана",
                )
              }
              onLoad={() =>
                runWithToast(() => invoke("load_model"), "Загрузка модели")
              }
              onUnload={() =>
                runWithToast(() => invoke("unload_model"), "Модель выгружена")
              }
              onInstall={(key) => {
                setInstallingKey(key);
                runWithToast(
                  () => invoke("install_model", { key }),
                  "Установка началась",
                  "Не удалось запустить установку",
                );
              }}
            />
          )}

          {section === "settings" && settings && (
            <SettingsSection
              settings={settings}
              snapshot={snapshot}
              busy={busy}
              onSave={(next) =>
                runWithToast(
                  () => invoke("save_settings", { settings: toCamelSettings(next) }),
                  "Сохранено",
                )
              }
              onLocalChange={(next) => setSettings(next)}
            />
          )}

          {section === "history" && (
            <HistorySection
              filteredHistory={filteredHistory}
              filter={filter}
              selected={selectedHistory}
              selectedId={selectedHistoryId}
              onCopy={(item) =>
                runWithToast(
                  () => invoke("copy_history_item", { id: item.id }),
                  "Скопировано",
                )
              }
              onFilter={setFilter}
              onSelect={setSelectedHistoryId}
            />
          )}
        </section>
      </section>

      {toast && <div className="toast">{toast}</div>}
    </main>
  );
}

function latestCopy(item: HistoryItem | undefined, fn: () => void) {
  if (item) fn();
}

// -----------------------------------------------------------------------------
// Shared presentational components
// -----------------------------------------------------------------------------

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>
      <div>
        <h2>Speech</h2>
        <p>локальная диктовка</p>
      </div>
    </div>
  );
}

function StatusPill({ running, modelState }: { running: boolean; modelState: string }) {
  const loading = running && modelState === "loading";
  const ready = running && (modelState === "loaded" || modelState === "ready");
  return (
    <div
      className={
        loading ? "status-pill loading" : ready ? "status-pill running" : "status-pill"
      }
    >
      <span />
      {loading ? "Загрузка" : running ? "Работает" : "Выкл"}
    </div>
  );
}

function headlineFor(snapshot: StatusSnapshot): string {
  if (!snapshot.running) return "Остановлено";
  if (!snapshot.engineEnabled) return "Движок выключен";
  if (snapshot.modelState === "loading") return `Загрузка ${snapshot.modelLabel}`;
  if (snapshot.modelState === "loaded") return "Готово";
  if (snapshot.modelState === "error") return "Нужно внимание";
  return "Работает";
}

// -----------------------------------------------------------------------------
// Section: Статус
// -----------------------------------------------------------------------------

function StatusSection({
  snapshot,
  latest,
  onCopyLatest,
}: {
  snapshot: StatusSnapshot;
  latest?: HistoryItem;
  onCopyLatest: () => void;
}) {
  return (
    <div className="status-grid">
      <article className="hero-card lifted">
        <div className="hero-copy">
          <p className="eyebrow">Push-to-talk</p>
          <h3>Готово ●</h3>
          <p>
            Удерживай <kbd>Ctrl</kbd>+<kbd>Win</kbd>, говори, отпускай. Текст
            вставится в активное поле, скопируется и сохранится в истории.
          </p>
        </div>
        <WavePreview />
      </article>

      <article className="note-card latest-note">
        <div>
          <p className="eyebrow">Последний транскрипт</p>
          <p>
            {latest?.text ||
              "Нет транскрипта. Удержи хоткей и скажи что-нибудь."}
          </p>
        </div>
        <button className="ghost-button" onClick={onCopyLatest} disabled={!latest}>
          Копировать
        </button>
      </article>

      <section className="stat-grid">
        <Metric title="Устройство" value={snapshot.device.toUpperCase()} detail="режим CPU" />
        <Metric title="Хоткей" value="Ctrl+Win" detail="удерживай для записи" />
      </section>
    </div>
  );
}

function Metric({ title, value, detail }: { title: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <p className="eyebrow">{title}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

function WavePreview() {
  return (
    <div className="wave-preview" aria-hidden="true">
      {[14, 26, 38, 24, 16, 30, 20].map((height, index) => (
        <span key={index} style={{ "--bar-height": `${height}px` } as React.CSSProperties} />
      ))}
    </div>
  );
}

// -----------------------------------------------------------------------------
// Section: Модели (detailed cards, choice Y)
// -----------------------------------------------------------------------------

function ModelsSection({
  models,
  snapshot,
  installingKey,
  busy,
  onSelect,
  onLoad,
  onUnload,
  onInstall,
}: {
  models: ModelInfo[];
  snapshot: StatusSnapshot;
  installingKey: string | null;
  busy: boolean;
  onSelect: (key: string) => void;
  onLoad: () => void;
  onUnload: () => void;
  onInstall: (key: string) => void;
}) {
  return (
    <div className="models-list">
      {models.length === 0 && (
        <article className="soft-panel">
          <p className="panel-copy">Ядро не сообщило о моделях. Запущен ли Speech?</p>
        </article>
      )}
      {models.map((model) => {
        const isActive = model.active;
        const isLoaded = isActive && snapshot.modelLoaded;
        const isInstalling = installingKey === model.key;
        const badge = isLoaded
          ? "загружена"
          : model.installed
            ? `${model.sizeLabel} установлен`
            : "не установлен";
        return (
          <article key={model.key} className={isActive ? "model-card active" : "model-card"}>
            <div className="model-card-top">
              <div>
                <strong>{model.label}</strong>
                <div className="model-card-sub">
                  {model.modelId} · {engineLabel(model.engine)}
                </div>
              </div>
              <span className={model.installed ? "badge ok" : "badge warn"}>{badge}</span>
            </div>
            <p className="model-card-desc">{model.description}</p>

            {isInstalling && (
              <div className="install-progress">
                <div className="spinner" aria-hidden="true" />
                <span>Устанавливается… Скачивание + конверсия CT2 (~5 мин, 5–8 ГБ RAM).</span>
              </div>
            )}

            <div className="model-card-actions">
              {!model.installed && !isInstalling && (
                <button onClick={() => onInstall(model.key)} disabled={busy}>
                  Установить
                </button>
              )}
              {isInstalling && (
                <button disabled>Устанавливается…</button>
              )}
              {isActive ? (
                <button className="ghost-button" disabled>
                  Активна ✓
                </button>
              ) : (
                <button className="ghost-button" onClick={() => onSelect(model.key)} disabled={busy}>
                  Выбрать активной
                </button>
              )}
              {isLoaded && (
                <button className="ghost-button" onClick={onUnload} disabled={busy}>
                  Выгрузить
                </button>
              )}
              {isActive && !isLoaded && !snapshot.modelLoading && (
                <button onClick={onLoad} disabled={busy}>
                  Загрузить
                </button>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function engineLabel(engine: string): string {
  if (engine === "whisper") return "faster-whisper (CTranslate2)";
  if (engine === "parakeet") return "Transformers / NeMo";
  return engine;
}

// -----------------------------------------------------------------------------
// Section: Настройки (accordion, choice Q)
// -----------------------------------------------------------------------------

function SettingsSection({
  settings,
  snapshot,
  busy,
  onSave,
  onLocalChange,
}: {
  settings: SettingsValues;
  snapshot: StatusSnapshot;
  busy: boolean;
  onSave: (next: SettingsValues) => void;
  onLocalChange: (next: SettingsValues) => void;
}) {
  const [openGroup, setOpenGroup] = useState<string>("quality");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const isWhisper = snapshot.model === "whisper-ru";

  function update<K extends keyof SettingsValues>(key: K, value: SettingsValues[K], autoSave = true) {
    const next = { ...settings, [key]: value };
    onLocalChange(next);
    if (autoSave) onSave(next);
  }

  const groups: Array<{
    id: string;
    title: string;
    summary: string;
  }> = [
    { id: "quality", title: "Качество и очистка", summary: "VAD, постобработка, генерация" },
    { id: "engine", title: "Движок и вывод", summary: "вкл/выкл, вставка, буфер, устройство" },
    { id: "hotkey", title: "Горячие клавиши", summary: "хоткей записи" },
  ];

  return (
    <div className="settings-accordion">
      {groups.map((group) => {
        const open = openGroup === group.id;
        return (
          <article key={group.id} className="acc-section">
            <button
              className="acc-head"
              onClick={() => setOpenGroup(open ? "" : group.id)}
              aria-expanded={open}
            >
              <span>{open ? "▾" : "▸"}</span>
              <strong>{group.title}</strong>
              <small>{group.summary}</small>
            </button>
            {open && (
              <div className="acc-body">
                {group.id === "quality" && (
                  <>
                    <Toggle
                      label="Постобработка текста"
                      checked={settings.postprocess_text}
                      onChange={(v) => update("postprocess_text", v)}
                    />
                    <NumberField
                      label="VAD чувствительность"
                      value={settings.vad_sensitivity}
                      step={0.005}
                      onCommit={(v) => update("vad_sensitivity", v)}
                    />
                    <NumberField
                      label="Beam size"
                      value={settings.beam_size}
                      step={1}
                      onCommit={(v) => update("beam_size", v)}
                    />
                    <NumberField
                      label="Temperature"
                      value={settings.temperature}
                      step={0.1}
                      onCommit={(v) => update("temperature", v)}
                    />
                    <NumberField
                      label="Repetition penalty"
                      value={settings.repetition_penalty}
                      step={0.05}
                      onCommit={(v) => update("repetition_penalty", v)}
                    />
                    <button
                      className="link-button"
                      onClick={() => setShowAdvanced((s) => !s)}
                    >
                      {showAdvanced ? "Скрыть подробности" : "Подробнее"}
                    </button>
                    {showAdvanced && (
                      <>
                        <NumberField
                          label="No-repeat ngram size"
                          value={settings.no_repeat_ngram_size}
                          step={1}
                          onCommit={(v) => update("no_repeat_ngram_size", v)}
                        />
                        <NumberField
                          label="Compression ratio threshold (Whisper)"
                          value={settings.compression_ratio_threshold}
                          step={0.1}
                          onCommit={(v) => update("compression_ratio_threshold", v)}
                        />
                        <NumberField
                          label="Log-prob threshold (Whisper)"
                          value={settings.log_prob_threshold}
                          step={0.1}
                          onCommit={(v) => update("log_prob_threshold", v)}
                        />
                      </>
                    )}
                  </>
                )}

                {group.id === "engine" && (
                  <>
                    <Toggle
                      label="Движок включён"
                      checked={settings.engine_enabled}
                      onChange={(v) => update("engine_enabled", v)}
                    />
                    <Toggle
                      label="Предзагрузка при запуске"
                      checked={settings.preload_model}
                      onChange={(v) => update("preload_model", v)}
                    />
                    <Toggle
                      label="Вставлять в активное поле"
                      checked={settings.paste_to_active_input}
                      onChange={(v) => update("paste_to_active_input", v)}
                    />
                    <Toggle
                      label="Копировать в буфер"
                      checked={settings.copy_to_clipboard}
                      onChange={(v) => update("copy_to_clipboard", v)}
                    />
                    <SelectField
                      label="Устройство"
                      value={settings.device}
                      options={["cpu", "cuda", "auto"]}
                      onChange={(v) => update("device", v)}
                    />
                    <SelectField
                      label="Backend"
                      value={settings.backend}
                      options={["auto", "transformers", "nemo"]}
                      onChange={(v) => update("backend", v)}
                      disabled={isWhisper}
                      hint={isWhisper ? "только для Parakeet" : undefined}
                    />
                  </>
                )}

                {group.id === "hotkey" && (
                  <TextField
                    label="Хоткей"
                    value={settings.hotkey}
                    onCommit={(v) => update("hotkey", v)}
                  />
                )}
              </div>
            )}
          </article>
        );
      })}
      {busy && <p className="panel-copy muted">Сохранение…</p>}
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="toggle-row">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
  disabled,
  hint,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <label className="field-row">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {hint && <small className="field-hint">{hint}</small>}
    </label>
  );
}

function TextField({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: string;
  onCommit: (value: string) => void;
}) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <label className="field-row">
      <span>{label}</span>
      <input
        type="text"
        value={local}
        onChange={(event) => setLocal(event.target.value)}
        onBlur={() => onCommit(local)}
        onKeyDown={(event) => {
          if (event.key === "Enter") onCommit(local);
        }}
      />
    </label>
  );
}

function NumberField({
  label,
  value,
  step = 1,
  onCommit,
}: {
  label: string;
  value: number;
  step?: number;
  onCommit: (value: number) => void;
}) {
  const [local, setLocal] = useState(String(value));
  useEffect(() => setLocal(String(value)), [value]);
  const commit = () => {
    const parsed = Number(local);
    if (!Number.isNaN(parsed)) onCommit(parsed);
    else setLocal(String(value));
  };
  return (
    <label className="field-row">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={local}
        onChange={(event) => setLocal(event.target.value)}
        onBlur={commit}
      />
    </label>
  );
}

// Rust SettingsPayload uses camelCase field names; convert before send.
function toCamelSettings(values: SettingsValues) {
  return {
    model: values.model,
    engineEnabled: values.engine_enabled,
    copyToClipboard: values.copy_to_clipboard,
    pasteToActiveInput: values.paste_to_active_input,
    preloadModel: values.preload_model,
    device: values.device,
    backend: values.backend,
    hotkey: values.hotkey,
    beamSize: values.beam_size,
    temperature: values.temperature,
    repetitionPenalty: values.repetition_penalty,
    noRepeatNgramSize: values.no_repeat_ngram_size,
    compressionRatioThreshold: values.compression_ratio_threshold,
    logProbThreshold: values.log_prob_threshold,
    vadSensitivity: values.vad_sensitivity,
    postprocessText: values.postprocess_text,
  };
}

// -----------------------------------------------------------------------------
// Section: История
// -----------------------------------------------------------------------------

function HistorySection({
  filteredHistory,
  filter,
  selected,
  selectedId,
  onCopy,
  onFilter,
  onSelect,
}: {
  filteredHistory: HistoryItem[];
  filter: string;
  selected?: HistoryItem;
  selectedId: string;
  onCopy: (item: HistoryItem) => void;
  onFilter: (value: string) => void;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="history-layout">
      <header className="history-header">
        <div>
          <p className="eyebrow">История</p>
          <h3>{filteredHistory.length} записей</h3>
        </div>
        <input
          value={filter}
          onChange={(event) => onFilter(event.target.value)}
          placeholder="Поиск по тексту"
        />
      </header>

      <section className="history-body">
        <div className="history-feed" aria-label="История транскриптов">
          {filteredHistory.length === 0 && (
            <div className="empty">Нет подходящих записей.</div>
          )}
          {filteredHistory.map((item) => (
            <button
              className={item.id === selectedId ? "history-row selected" : "history-row"}
              key={item.id}
              onClick={() => onSelect(item.id)}
            >
              <span>{item.text}</span>
            </button>
          ))}
        </div>

        <article className="reader-card lifted">
          <div>
            <p className="eyebrow">Выбранная запись</p>
            <p>{selected?.text || "Выбери запись слева."}</p>
          </div>
          <button
            className="ghost-button"
            onClick={() => selected && onCopy(selected)}
            disabled={!selected}
          >
            Копировать
          </button>
        </article>
      </section>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function getInitialSection(): Section {
  const hash = window.location.hash.replace("#", "");
  return navItems.some((item) => item.id === hash) ? (hash as Section) : "status";
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
