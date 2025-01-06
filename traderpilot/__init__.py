"""Traderpilot bot"""

__version__ = "2025.1-dev"

if "dev" in __version__:
    from pathlib import Path

    try:
        import subprocess  # noqa: S404

        traderpilot_basedir = Path(__file__).parent

        __version__ = (
            __version__
            + "-"
            + subprocess.check_output(
                ["git", "log", '--format="%h"', "-n 1"],
                stderr=subprocess.DEVNULL,
                cwd=traderpilot_basedir,
            )
            .decode("utf-8")
            .rstrip()
            .strip('"')
        )

    except Exception:  # pragma: no cover
        # git not available, ignore
        try:
            # Try Fallback to traderpilot_commit file (created by CI while building docker image)
            versionfile = Path("./traderpilot_commit")
            if versionfile.is_file():
                __version__ = f"docker-{__version__}-{versionfile.read_text()[:8]}"
        except Exception:  # noqa: S110
            pass
