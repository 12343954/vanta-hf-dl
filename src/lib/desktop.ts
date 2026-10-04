import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { getCurrentWindow, type CloseRequestedEvent } from '@tauri-apps/api/window'
import type { DownloadJob } from './downloads'
import type { RepoResult } from './huggingface'

export type FavoriteRepo = RepoResult & {
  favoritedAt: string
}

export type BackendStatus = {
  ready: boolean
  storage: string
  downloader: string
}

type TargetPreviewRequest = {
  filePath: string
  modelsRoot: string
  subfolder: string
}

type TargetPreviewResponse = {
  target_path: string
}

export type DownloadProgressEvent = {
  downloadedBytes: number
  error?: string
  jobId: string
  status: DownloadJob['status']
  warning?: string
}

export function hasTauriRuntime() {
  return '__TAURI_INTERNALS__' in window
}

export async function getBackendStatus(): Promise<BackendStatus> {
  if (!hasTauriRuntime()) {
    return {
      downloader: 'browser-preview',
      ready: false,
      storage: 'localStorage',
    }
  }

  return invoke<BackendStatus>('backend_status')
}

export async function previewDownloadTarget(request: TargetPreviewRequest) {
  if (!hasTauriRuntime()) {
    return ''
  }

  const response = await invoke<TargetPreviewResponse>('preview_download_target', {
    request: {
      file_path: request.filePath,
      models_root: request.modelsRoot,
      subfolder: request.subfolder,
    },
  })
  return response.target_path
}

export async function pickFolder(title: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<string | undefined>('pick_folder', {
    request: {
      title,
    },
  })
}

export async function listDownloadJobs() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('list_download_jobs')
}

export async function listActiveDownloadJobs() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('list_active_download_jobs')
}

export async function enqueueDownloads(params: {
  accessToken: string
  jobs: DownloadJob[]
  maxConcurrentParts?: number
  proxy: string
}) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('enqueue_downloads', {
    request: {
      access_token: params.accessToken,
      jobs: params.jobs,
      max_concurrent_parts: params.maxConcurrentParts,
      proxy: params.proxy,
    },
  })
}

export async function pauseDownloadJob(jobId: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('pause_download_job', { jobId })
}

export async function resumeDownloadJob(params: { accessToken: string, jobId: string, maxConcurrentParts: number, proxy: string }) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('resume_download_job', {
    accessToken: params.accessToken,
    jobId: params.jobId,
    maxConcurrentParts: params.maxConcurrentParts,
    proxy: params.proxy,
  })
}

export async function cancelDownloadJob(jobId: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('cancel_download_job', { jobId })
}

export async function removeDownloadJob(jobId: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('remove_download_job', { jobId })
}

export async function refreshDownloadSources(hfEndpoint: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<DownloadJob[]>('refresh_download_sources', {
    request: {
      hf_endpoint: hfEndpoint,
    },
  })
}

export async function openPath(path: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<void>('open_path', { path })
}

export async function openUrl(url: string) {
  if (!hasTauriRuntime()) {
    window.open(url, '_blank', 'noopener,noreferrer')
    return undefined
  }

  return invoke<void>('open_url', { url })
}

export async function updateTraySummary(summary: string) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<void>('update_tray_summary', { summary })
}

export async function loadFavorites() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<FavoriteRepo[]>('load_favorites')
}

export async function saveFavorites(favorites: FavoriteRepo[]) {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<FavoriteRepo[]>('save_favorites', { favorites })
}

export async function listenDownloadProgress(callback: (event: DownloadProgressEvent) => void) {
  if (!hasTauriRuntime()) {
    return () => {}
  }

  return listen<DownloadProgressEvent>('download-progress', (event) => callback(event.payload))
}

export async function listenWindowCloseRequested(callback: (event: CloseRequestedEvent) => void | Promise<void>) {
  if (!hasTauriRuntime()) {
    return () => {}
  }

  return getCurrentWindow().onCloseRequested(callback)
}

export async function listenNativeCloseRequested(callback: () => void) {
  if (!hasTauriRuntime()) {
    return () => {}
  }

  return listen('native-close-requested', callback)
}

export async function destroyWindow() {
  if (!hasTauriRuntime()) {
    window.close()
    return
  }

  return invoke<void>('quit_app')
}

export async function minimizeWindow() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<void>('hide_main_to_tray')
}

export async function minimizeNativeWindow() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<void>('minimize_main_window')
}

export async function toggleMaximizeWindow() {
  if (!hasTauriRuntime()) {
    return undefined
  }

  return invoke<void>('toggle_main_window_maximize')
}
