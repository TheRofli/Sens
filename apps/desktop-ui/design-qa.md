# Sens desktop design QA

Final result: passed

## Comparison contract

- Reference: `design-reference.png` — the selected Sens concept.
- Candidate: `qa/capability-settings/07-final-home.png`.
- Combined comparison: `qa/capability-settings/08-reference-vs-final-home.png`.
- Viewport and state: 1487 × 1058, Home selected, main window open, tray panel visible.
- New-flow evidence: `05-updated-capabilities.png`, `04-sight-settings.png`, and `06-hearing-settings.png`.
- Required tokens: Lumen Cream `#ffffeb`, Vast Ink `#1a1a1a`, Lavender Whisper `#f0d7ff`, Forest Ink `#034f46`, Ember Glow `#ffa946`.

## Results

- Source and candidate were compared side by side at identical dimensions.
- The selected visual language is preserved: shell geometry, typography hierarchy, cream canvas, ink panels, lavender controls, green connection card, and restrained two-pixel borders.
- The intentional Home change replaces the ambiguous `Новая возможность` action with `Настроить чувства`; Sight and Hearing remain direct, compact settings entry points.
- Capability cards now behave as controls and expose provider/model context before entering a settings screen.
- Sight and Hearing settings reuse the existing system rather than introducing a new visual language.
- Browser interaction passed: navigation, select fields, numeric fields, toggles, save actions, and confirmation toasts.
- Accessibility names passed for the two direct capability entry points and the capability enable switches.
- Browser console passed with no warnings or errors from the application.

## Severity review

- P0: none.
- P1: none.
- P2: none.
- P3: the supplied reference is a raster concept, so small differences remain in font antialiasing and the 64 × 74 source logo. They do not affect hierarchy, layout, contrast, or interaction.
