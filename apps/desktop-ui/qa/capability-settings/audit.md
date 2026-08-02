# Sens capability-settings audit

## Scope and user goal

The reviewed journey begins on Home, continues through the capability catalog, and ends in detailed Sight or Hearing settings. The goal is to make it immediately clear that connecting an MCP client is different from configuring a sense, while leaving room for future Sens modules.

## Step 1 — Home

- Strong: the primary connection action is visible and accurately names its target: `Подключить модель или приложение`.
- Fixed: the former `Новая возможность` control duplicated the connection action and had no clear product meaning.
- Current behavior: the Sight and Hearing pills open their own settings; `Настроить чувства` opens the catalog.

## Step 2 — Capability catalog

- Strong: Sight and Hearing are presented as installed, working modules with current provider/model context.
- Fixed: the cards were static content and could not be used to reach detailed configuration.
- Current behavior: both cards are keyboard-accessible buttons. `Будущие чувства` is a passive roadmap note and cannot be confused with a connection CTA.

## Step 3 — Sight settings

- The screen exposes provider, model, analysis depth, per-image call limit, enable state, and cache behavior.
- The hierarchy is clear: identity/status, model section, quality/resources section, then operational switches and save.
- Saving in the native app writes only supported Eye fields; the browser prototype keeps the edit in memory.

## Step 4 — Hearing settings

- The screen exposes recognition model, device, beam size, VAD sensitivity, enable state, model preload, and text post-processing.
- Labels explain speed/accuracy and memory tradeoffs without requiring knowledge of Speech internals.
- Saving in the native app writes only supported Speech fields; the browser prototype keeps the edit in memory.

## General health

- Visual hierarchy: healthy.
- Interaction clarity: healthy after separating connection, capability settings, and roadmap concepts.
- Accessibility: controls have semantic roles and specific accessible names.
- Runtime feedback: save actions expose a status toast; disabled capability state is reflected in the tray model.
- Remaining product risk: configuration choices will need real-task validation in the planned Sens evaluation pack, especially Eye detail/call limits and Hearing beam/VAD defaults.
