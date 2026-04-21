"""CrimsonForge - Crimson Desert Modding Studio."""

import sys
import os
import tempfile
import glob
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Crash diagnostics: install FIRST, before any import that could
#    fail at module-init time. Previously a native DLL-load crash in
#    QApplication() or in a hidden import exited the process with
#    zero log output. faulthandler now captures C-level faults and
#    the Python excepthook catches uncaught exceptions. Both write
#    to the session log before the process dies. See
#    core.crash_handler for the three-layer defence.
try:
    from core.crash_handler import install_crash_handlers, log_and_show_fatal
    _crash_log_path = os.path.join(tempfile.gettempdir(), "crimsonforge.log")
    install_crash_handlers(_crash_log_path)
except Exception:
    # Best-effort only — the rest of the app must still boot even
    # when the diagnostics module can't load.
    def log_and_show_fatal(title, message):   # type: ignore[no-redef]
        pass

from PySide6.QtWidgets import QApplication

from version import APP_VERSION, APP_NAME
from utils.config import ConfigManager, ConfigLoadError
from utils.logger import setup_logger, get_logger
from ai.provider_registry import ProviderRegistry
from ui.main_window import MainWindow


def _close_splash():
    """Close the PyInstaller splash screen if running from a bundled exe."""
    try:
        import pyi_splash          # only available inside PyInstaller bundle
        pyi_splash.close()
    except ImportError:
        pass


def _cleanup_temp_files():
    """Delete all temporary directories and files created during the session."""
    tmp = tempfile.gettempdir()
    patterns = [
        "crimsonforge_audio_*",
        "crimsonforge_preview_*",
        "cf_wem_out",
        "cf_wwise_project",
        "cf_wwise_*",
        "cf_wem_*.wem"
    ]
    
    count = 0
    for pat in patterns:
        for path in glob.glob(os.path.join(tmp, pat)):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                count += 1
            except Exception:
                pass
                
    if count > 0:
        log = get_logger("cleanup")
        log.info("Cleaned up %d temporary files/folders on exit", count)


def main():
    # QApplication() is the single most-likely silent-crash point in
    # PyInstaller bundles — it loads the Qt platform plugin which can
    # abort natively when the plugin DLL is truncated (e.g. after a
    # force-reboot interrupted the bundle extraction). We surround it
    # with a Python try/except so any Python-level failure surfaces
    # as a log line + native MessageBox; faulthandler (installed at
    # module top) catches the native-abort case.
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        log_and_show_fatal(
            "CrimsonForge — Qt initialisation failed",
            (
                f"Qt / PySide6 could not be initialised:\n\n{type(e).__name__}: {e}\n\n"
                "Likely causes:\n"
                "  • Hard reboot corrupted the PyInstaller extraction. "
                "Delete %TEMP%\\_MEI* folders and retry.\n"
                "  • Missing VC++ 2015-2022 redistributable "
                "(https://aka.ms/vs/17/release/vc_redist.x64.exe).\n"
                "  • Antivirus quarantined a bundled DLL. "
                "Whitelist the exe and %TEMP%\\_MEI* folders."
            ),
        )
        return 1
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("hzeem")

    # Set a multilingual font with fallbacks for Korean, Chinese, Japanese, Arabic, etc.
    from PySide6.QtGui import QFont
    font = QFont("Segoe UI", 10)
    font.setFamilies([
        "Segoe UI",            # Latin, Cyrillic
        "Microsoft YaHei",     # Chinese (Simplified)
        "Malgun Gothic",       # Korean
        "Meiryo",              # Japanese
        "Segoe UI Symbol",     # Symbols, emoji
        "Noto Sans",           # Broad Unicode coverage (if installed)
    ])
    app.setFont(font)

    try:
        config = ConfigManager()
    except ConfigLoadError as e:
        _close_splash()
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Configuration Error", str(e))
        return 1

    logger = setup_logger(
        log_level=config.get("advanced.log_level", "INFO"),
        log_file=config.get("advanced.log_file", ""),
        debug_mode=config.get("advanced.debug_mode", False),
    )
    logger.info("%s v%s starting...", APP_NAME, APP_VERSION)
    logger.info("Config loaded from: %s", config.config_path)

    registry = ProviderRegistry()
    registry.initialize_from_config(config.get_section("ai_providers"))
    logger.info("AI providers initialized: %s", registry.list_enabled_provider_ids())
    window = MainWindow(config, registry)

    _close_splash()
    window.show()

    logger.info("Application ready")
    ret = app.exec()
    
    logger.info("Application closing. Running cleanup...")
    _cleanup_temp_files()
    
    return ret


if __name__ == "__main__":
    sys.exit(main())
