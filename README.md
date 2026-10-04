# Vanta HF Downloader

Vanta is a desktop downloader for Hugging Face model repositories. Its main goal is to keep ComfyUI model folders clean: downloaded files can be placed under an extra named subfolder, so different repositories do not get mixed together in the same model directory.

Repository name: `12343954/vanta-hf-dl`

## Features

- Search Hugging Face repositories or paste a repo URL directly.
- Browse repository file trees and select only the files you need.
- Add a repository-based secondary folder to keep ComfyUI model directories organized.
- Download large files with 32 MB ranged chunks.
- Pause, resume, cancel, and restart downloads.
- Persist download progress in SQLite.
- Continue failed model-level tasks with an auto-retry switch.
- Track total download speed and per-task progress.
- Support Hugging Face tokens, mirrors, and proxies.
- Manage local downloaded models.
- Minimize to system tray while downloads continue.
- Warn on SHA256 mismatch without marking the finished file as failed.

## Tech Stack

- Tauri `2.11.5` / Tauri CLI `2.11.4`
- React `19.2.8`
- TypeScript `6.0.2`
- Vite `8.2.2`
- Rust `2021` edition
- SQLite via `rusqlite 0.32`
- Reqwest `0.12`
- Tokio `1`
- Tailwind CSS `4.3.3`

## Development

Install dependencies:

```bash
npm ci
```

Run the desktop app in development mode:

```bash
npm run tauri:dev
```

On Windows, this helper script ensures Cargo is on `PATH`:

```bash
npm run tauri:dev:win
```

Build the frontend only:

```bash
npm run build
```

Build desktop installers:

```bash
npm run tauri:build
```

## GitHub Actions Build

The workflow at `.github/workflows/build.yml` builds packages for:

- Windows x64
- Linux x64
- macOS Apple Silicon
- macOS Intel

Run it manually from GitHub Actions, or push a tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The generated installers are uploaded as workflow artifacts.

## Update Check

Vanta can use GitHub Releases as a lightweight update source. The app checks the latest release and shows a small update indicator when a newer version exists. Clicking it opens:

```text
https://github.com/12343954/vanta-hf-dl/releases/latest
```

This is only a release-page shortcut, not an automatic installer updater.

## Notes

- Private or gated Hugging Face repositories require a valid access token.
- Existing downloads keep their current URL until paused and resumed.
- Completed files are considered usable when the file size matches. If SHA256 is available and mismatches, Vanta shows a warning instead of failing the task.

## License

MIT
