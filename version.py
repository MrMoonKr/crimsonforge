"""CrimsonForge version and changelog registry.

Single source of truth for the application version. Every module that needs
the version string imports it from here. The CHANGELOG list is rendered in the
About tab so users (and developers) always see what changed.

VERSION BUMPING RULES
---------------------
- Bump PATCH (1.x.Y) for bug fixes, small tweaks, and safe improvements.
- Bump MINOR (1.X.0) for new features, new tabs, new AI providers, or
  significant workflow changes.
- Bump MAJOR (X.0.0) for breaking changes to project files, settings
  format, or game-patch pipeline.
- Always add a new entry at the TOP of CHANGELOG when changing code.
"""

__all__ = ["APP_VERSION", "APP_NAME", "CHANGELOG"]

APP_NAME = "CrimsonForge"
APP_VERSION = "1.8.0"

# Each entry: (version, date, list_of_changes)
# Newest first. `date` is YYYY-MM-DD.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    (
        "1.8.0", "2026-04-01", [
            # ── PAC Mesh Parser (complete rewrite) ──
            "[Feature] PAC mesh parser fully reverse-engineered from binary analysis — correct geometry for all character meshes",
            "[Feature] PAC section layout auto-detected from section offset table inside section 0 — works for all format variants",
            "[Feature] PAC vertex data: uint16 quantized positions dequantized with per-submesh bounding box",
            "[Feature] PAC index buffer: triangle list format with per-submesh index counts per LOD level",
            "[Feature] PAC multi-LOD support: LOD0 (highest quality) automatically selected for preview and export",
            "[Feature] PAC multi-submesh support: sword blades, guards, handles, accessories parsed as separate objects",
            "[Feature] PAC bone index padding: odd bone counts padded to even byte boundary (fixes facial/head meshes)",
            "[Feature] PAC auto-detect vertex stride from section size — handles 36, 38, 40, 42+ byte strides",
            "[Feature] PAC idx_count validation: stops reading at garbage values to prevent buffer overruns",
            "[Feature] UV coordinates extracted from float16 values in vertex records",

            # ── Explorer Export Fixes ──
            "[Fix] Export context menu now uses right-clicked row instead of selected row — no more exporting wrong file",
            "[Fix] Export output filenames include full path (e.g. character_warrior_body.obj) — no more overwrites",
            "[Fix] Lambda closure in export menu binds entry by value — prevents stale reference issues",

            # ── Format Compatibility ──
            "[Feature] 3-LOD PAC files (cd_pgw_* heads, eyebrows) now parse correctly alongside 4-LOD files",
            "[Feature] Variable section size encoding handled: u64 pairs, consecutive u32s, and mixed layouts",
            "[Feature] Unsupported PAC variants (skinnedmesh_box v4.3) gracefully skip instead of showing errors",
        ],
    ),
    (
        "1.7.0", "2026-03-31", [
            # ── Localization Tracer ──
            "[Feature] Localization Tracer: standalone tool — type any text, instantly see every screen it appears on in-game",
            "[Feature] Tracer shows the full chain for each hit: which UI screen, which element, what CSS styling, what font and color",
            "[Feature] 182 game screens mapped to readable names (Character Select, Skill Tree, World Map, Alert Popup, etc.)",
            "[Feature] Three search modes: search by displayed text, by paloc key ID, or by UI binding name",
            "[Feature] When a string appears on multiple screens, all locations are listed with descriptions",
            "[Feature] All 170 CSS, 153 HTML, and 29 template files decrypted and indexed on startup",

            # ── Game UI System ──
            "[Feature] Full game UI system reverse-engineered: HTML/CSS-based with custom localstring binding to paloc entries",
            "[Feature] Per-language CSS files identified — each language has its own font rules and line-breaking behavior",
            "[Feature] Widget template system mapped: reusable KeyGuide, Modal, ItemTooltip components with text overrides",
            "[Feature] 115 UI text bindings cataloged (Save/Load, Exit, Confirm, Cancel, menu labels, skill names, shop titles, etc.)",
            "[Feature] Runtime template variables documented: keybind display, currency icons, clickable game-term links",

            # ── 3D Mesh ──
            "[Feature] Extract and preview all 12,724 skinned character meshes (.pac) from game archives",
            "[Feature] Extract and preview 50,388 static meshes (.pam) including props, terrain, and breakable objects",
            "[Feature] Extract and preview 32,188 LOD mesh variants (.pamlod) with multiple quality levels",
            "[Feature] Export any mesh to OBJ (Wavefront) or FBX (binary 7.4) from Explorer right-click menu",
            "[Feature] FBX export auto-finds and embeds the matching skeleton with full bone hierarchy",
            "[Feature] Mesh preview shows 3D render, vertex/face counts, submesh list, materials, and textures",
            "[Feature] Breakable and destructible object meshes now extract correctly",

            # ── Textures ──
            "[Feature] Preview all 279,515 DDS textures directly in Explorer — no external tools needed",
            "[Feature] Supports all game texture formats: color, normal maps, roughness, heightmaps, distance fields",
            "[Feature] Grayscale and terrain textures render as preview instead of showing an error",

            # ── Skeleton / Animation / Havok ──
            "[Feature] Extract skeleton data (.pab): bone names, parent hierarchy, bind poses, transforms",
            "[Feature] Extract animation data (.paa): keyframes, bone rotations, frame count, duration",
            "[Feature] Extract Havok data (.hkx): bone names, skeleton hierarchy, content type (skeleton/animation/physics/ragdoll)",
            "[Feature] Preview all skeleton, animation, and Havok files directly in Explorer",

            # ── File Support ──
            "[Feature] 108 game file extensions recognized with category, description, and preview/edit support",
        ],
    ),
    (
        "1.6.0", "2026-03-30", [
            "[Feature] OBJ export with materials, UVs, normals, and multi-submesh support",
            "[Feature] FBX binary 7.4 export compatible with Blender, Maya, 3ds Max, Unity, Unreal Engine",
            "[Feature] Right-click Export as OBJ / Export as FBX on any mesh file in Explorer",
            "[Feature] DDS texture header info: format name, resolution, mipmap count, alpha channel",
            "[Feature] Mesh preview in Explorer with static 3D render and geometry statistics",
            "[Feature] Split export option: save each submesh as a separate OBJ file",
            "[Feature] Custom scale factor for mesh export",
        ],
    ),
    (
        "1.5.0", "2026-03-30", [
            "[Feature] Ship to App: generate ZIP+BAT packages for end-user mod distribution",
            "[Feature] Ship to App: auto-discovers Steam game, copies pre-patched files, one-click install",
            "[Feature] Ship to App: built-in font donor system — select donor font, auto-adds missing glyphs for target language",
            "[Feature] Ship to App: uninstall via Steam Verify Integrity — clean and reliable restoration",
            "[Feature] Paloc parser now extracts 172K+ entries (both numeric and symbolic keys like questdialog_*, textdialog_*)",
            "[Feature] Dialogue and Documents categories now populated from symbolic keys (was empty before)",
            "[Feature] Auto-lock untranslatable entries (empty, PHM_, placeholder) — marked Approved and protected from editing",
            "[Feature] Locked status filter in translation table — view all auto-locked entries",
            "[Feature] Wildcard search: key:quest*, *dragon*, {*} for brace tokens, locked:yes, empty:yes",
            "[Feature] Game version read from meta/0.paver — shows real version (e.g. v1.01.02) in About tab",
            "[Feature] Status bar auto-reflects on app startup — badges populate immediately after restore",
            "[Feature] Always merge with fresh game data on startup — catches new entries, parser improvements, patches",
            "[Feature] Detailed game update popup with new/changed/removed counts and text samples",
            "[Feature] Arrow key navigation in Explorer tab now triggers preview (was mouse-only)",
            "[Enhancement] Comprehensive tooltips on every widget across all tabs (Translate, Explorer, Font, Settings, About)",
            "[Enhancement] Search supports field:value syntax, quoted phrases, glob wildcards, boolean operators",
            "[Enhancement] LZ4/ChaCha20 checksum computed via native DLL (754x faster than pure Python on large PAZ files)",
            "[Enhancement] Font Builder: GSUB/GPOS merge filters out lookups referencing missing glyphs — no more KeyError crashes",
            "[Enhancement] Font Builder: handles CJK fonts with coordinates > 16-bit by clamping bounding boxes",
            "[Enhancement] Ship to App BAT scripts use delayed expansion for paths with (x86) parentheses",
            "[Fix] Usage filter categories (Dialogue, Documents) were empty on Windows — now auto-discovered from all game groups",
            "[Fix] paloc_parser was discarding 55K+ symbolic key entries (questdialog_*, textdialog_*) — now extracted",
            "[Fix] Patch to Game duplicate popup and 1-3 minute freeze — O(n) duplicate apply, single confirmation dialog",
            "[Fix] QComboBox and QTextBrowser text invisible on Windows — explicit color in QSS for item pseudo-elements",
            "[Fix] checksum_file() was bypassing native DLL, falling back to slow pure Python — now routes through pa_checksum()",
        ],
    ),
    (
        "1.4.0", "2026-03-30", [
            "[Feature] Complete UI overhaul: modern Catppuccin-inspired theme with rounded corners, gradient progress bars, smooth hover states",
            "[Feature] New button variants: primary (blue), danger (red), success (green), warning (yellow) with proper hover/press/disabled states",
            "[Feature] Styled tool buttons with checked state for toggles (loop, mute)",
            "[Enhancement] Buttons now have 6px border-radius, 500 font-weight, proper focus rings",
            "[Enhancement] Tab bar redesigned: no borders, bottom-accent style, cleaner spacing",
            "[Enhancement] Table view: removed grid lines, increased row height (30px), cleaner cell padding",
            "[Enhancement] Context menus: rounded corners (8px), proper padding, separators",
            "[Enhancement] Scrollbars: transparent track, rounded handles, pressed state",
            "[Enhancement] Combobox dropdowns: rounded items, hover highlights, proper padding",
            "[Enhancement] Group boxes: 8px radius, blue title color, more padding",
            "[Enhancement] Search input: clear button enabled, better placeholder text",
            "[Enhancement] Slider controls: styled groove, rounded handle with hover-grow effect, filled sub-page",
            "[Enhancement] Progress bar: gradient fill (teal to blue), rounded shape",
            "[Enhancement] Translate tab: danger-styled Clear/Clear All buttons, success-styled Patch to Game, warning-styled Revert",
            "[Enhancement] Translate tab: proper vertical line separators replacing ugly '|' text labels",
            "[Enhancement] Translate tab: Stop button danger-styled, AI Selected primary-styled, Approve All success-styled",
            "[Enhancement] Filter bar: styled labels, fixed-width status combo, count label highlighted in blue",
            "[Enhancement] Light theme fully redesigned to match dark theme quality",
            "[Enhancement] Tooltips: rounded corners (6px), proper padding",
            "[Enhancement] Text browser (About/Changelog): styled with proper borders and selection colors",
        ],
    ),
    (
        "1.3.0", "2026-03-30", [
            "[Feature] Auto-install vgmstream: one-click download and install for Wwise audio playback (no manual setup needed)",
            "[Feature] Enhanced audio player: volume slider with mute toggle, loop playback, format info display",
            "[Feature] Audio player keyboard shortcuts: Space (play/pause), S (stop), M (mute), L (loop)",
            "[Feature] Video player now has full transport controls: play/pause/stop, seek, volume, mute, loop",
            "[Feature] Video controls properly connected to video player (was using separate audio-only player before)",
            "[Feature] Added Bink2 (.bk2/.bik) video format support - common in Crimson Desert cinematics",
            "[Feature] Added CriWare USM (.usm) video format support - used in game cutscenes",
            "[Feature] Added MKV, FLAC, AAC format detection and preview support",
            "[Feature] Magic byte detection for FLAC, Bink2, and CriWare USM formats",
            "[Enhancement] Audio preview shows centered format icon with file type label",
            "[Enhancement] Time display supports hours for long audio/video (H:MM:SS format)",
            "[Enhancement] vgmstream auto-installer shows download progress and retry on failure",
            "[Enhancement] Explorer file type filters updated with all new audio/video formats",
            "[Fix] Fixed preview_pane.py clear() method was incorrectly nested inside _html_escape function",
            "[Fix] Video player audio was not controllable - now properly wired to volume/mute controls",
        ],
    ),
    (
        "1.2.0", "2026-03-30", [
            "[Feature] Auto-load game on first run - no manual 'Load Game' click needed when Steam install is found",
            "[Feature] Game version display in status bar, translate tab stats row, and paloc info label (CRC fingerprint + modification date)",
            "[Feature] Auto-check for new game files, new package groups, and new language entries on every launch",
            "[Feature] Game update detection with notification banner showing what changed since last session",
            "[Feature] Centralized version system with full changelog in About tab",
            "[Enhancement] Enterprise context menu: shows selection count, Revert to Pending option, Clear Selected, Select All",
            "[Enhancement] Keyboard shortcut: Ctrl+A to select all visible rows in translation table",
            "[Enhancement] Keyboard shortcut: Delete key to clear selected translations and revert to Pending",
            "[Enhancement] Paste feedback: status bar shows 'Pasted to N entries' after paste operation",
            "[Enhancement] Batch status operations now emit proper signals for real-time stats updates",
            "[Fix] Entry editor (double-click) save now correctly persists translation text",
            "[Fix] Auto-transition Pending -> Translated when entering text in entry editor",
            "[Fix] Auto-revert to Pending when clearing translation text (both inline and editor dialog)",
            "[Fix] Real-time status combo auto-update as you type in entry editor",
            "[Fix] Paste to multiple selected rows: single copied line now applies to ALL selected rows",
            "[Fix] Stats bar updates immediately after save/paste/status change operations",
        ],
    ),
    (
        "1.1.0", "2026-03-28", [
            "[Feature] Full 'Patch to Game' pipeline: export, compress, encrypt, write PAZ, update PAMT+PAPGT checksum chain",
            "[Feature] Backup manager creates automatic backups before patching game files",
            "[Feature] Duplicate detection: finds identical original text across entries and offers batch-apply",
            "[Feature] Glossary manager for proper nouns - ensures consistent translation of names, places, factions",
            "[Feature] AI glossary injection: glossary terms are injected into every AI translation prompt",
            "[Feature] Baseline manager: immutable reference of original game text, survives game updates",
            "[Feature] Game update merge: detects new/removed/changed strings and preserves translations",
            "[Feature] Import/Export JSON for external editing with merge-by-key support",
            "[Feature] Export to TXT with tab-separated format for spreadsheet compatibility",
            "[Feature] Autosave manager with configurable interval (default 30s)",
            "[Feature] Session state recovery: restores project, UI selections, and scroll position on restart",
            "[Feature] Translation batch processor with pause/resume/stop controls",
            "[Feature] Localization usage index: tags strings by game context (dialogue, quest, UI, skills, etc.)",
            "[Enhancement] Usage category filter in translation table",
            "[Enhancement] Advanced search: field-specific queries (key:, original:, translation:, usage:, status:)",
            "[Enhancement] Ranked search results with weighted scoring across all fields",
            "[Enhancement] Review All / Approve All bulk operations with progress",
            "[Enhancement] Token and cost tracking per translation batch",
        ],
    ),
    (
        "1.0.0", "2026-03-19", [
            "[Feature] Initial release - Crimson Desert Modding Studio",
            "[Feature] Game auto-discovery: scans Steam libraries for Crimson Desert installation",
            "[Feature] VFS (Virtual File System) for reading game package archives",
            "[Feature] PAPGT root index parser with full checksum chain support",
            "[Feature] PAMT metadata parser for file entries within package groups",
            "[Feature] PAZ archive reader with ChaCha20 decryption and LZ4 decompression",
            "[Feature] Paloc localization file parser and builder",
            "[Feature] Explorer tab: browse, unpack, and inspect game resources",
            "[Feature] Repack tab: rebuild modified resources back into game archives",
            "[Feature] Translate tab: AI-powered and manual translation workspace",
            "[Feature] Font Builder tab: custom font generation for game text rendering",
            "[Feature] Settings tab: configure AI providers, models, and preferences",
            "[Feature] Multi-provider AI translation: OpenAI, Anthropic, Google, DeepSeek, local models",
            "[Feature] Translation table with virtual scrolling (100K+ entries)",
            "[Feature] Column sorting, copy/paste, and status management",
            "[Feature] Dark and Light theme support",
            "[Feature] 17 game languages auto-detected from paloc files",
            "[Feature] 70+ world languages available as translation targets",
        ],
    ),
]


def get_changelog_html() -> str:
    """Render the full changelog as styled HTML for the About tab."""
    tag_colors = {
        "Feature": "#a6e3a1",
        "Enhancement": "#89b4fa",
        "Fix": "#f9e2af",
        "Breaking": "#f38ba8",
        "Security": "#cba6f7",
        "Deprecated": "#fab387",
        "Removed": "#eba0ac",
        "Performance": "#94e2d5",
    }

    html_parts = []
    for version, date, changes in CHANGELOG:
        html_parts.append(
            f'<h3 style="margin-top:18px; margin-bottom:4px;">'
            f'v{version} &mdash; {date}</h3>'
        )
        html_parts.append('<ul style="margin-top:2px;">')
        for change in changes:
            # Parse [Tag] prefix for coloring
            display = change
            for tag, color in tag_colors.items():
                prefix = f"[{tag}]"
                if change.startswith(prefix):
                    rest = change[len(prefix):].strip()
                    display = (
                        f'<span style="color:{color}; font-weight:bold;">[{tag}]</span> '
                        f'{rest}'
                    )
                    break
            html_parts.append(f"<li>{display}</li>")
        html_parts.append("</ul>")

    return "\n".join(html_parts)
