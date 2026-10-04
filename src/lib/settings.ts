import { invoke } from '@tauri-apps/api/core'

export const settingsStorageKey = 'vanta:app-settings'
export const tokenStorageKey = 'vanta:hf-access-token'

export type AppLanguage = 'en' | 'zh-CN' | 'zh-TW'

export type AppSettings = {
  comfyRoot: string
  modelsRoot: string
  hfEndpoint: string
  proxy: string
  maxConcurrentDownloads: number
  defaultSubfolderPattern: string
  readmeDownloadRatio: number
  language: AppLanguage
}

export const defaultSettings: AppSettings = {
  comfyRoot: 'G:\\ComfyUI-Mie\\ComfyUI',
  modelsRoot: 'G:\\ComfyUI-Mie\\ComfyUI\\models',
  hfEndpoint: 'https://huggingface.co',
  proxy: '',
  maxConcurrentDownloads: 3,
  defaultSubfolderPattern: '{repo}',
  readmeDownloadRatio: 58,
  language: 'en',
}

function clampConcurrentDownloads(value: number) {
  return Math.min(12, Math.max(1, Math.round(value)))
}

function clampReadmeDownloadRatio(value: number) {
  return Math.min(70, Math.max(21, value))
}

function normalizeLanguage(value?: string): AppLanguage {
  return value === 'zh-CN' || value === 'zh-TW' ? value : 'en'
}

export type AppStorage = {
  accessToken: string
  settings: AppSettings
}

type AppStorageResponse = {
  access_token: string
  settings: Partial<AppSettings>
}

function hasTauriRuntime() {
  return '__TAURI_INTERNALS__' in window
}

export function normalizeSettings(value: Partial<AppSettings>): AppSettings {
  return {
    ...defaultSettings,
    ...value,
    maxConcurrentDownloads: clampConcurrentDownloads(value.maxConcurrentDownloads ?? defaultSettings.maxConcurrentDownloads),
    readmeDownloadRatio: clampReadmeDownloadRatio(value.readmeDownloadRatio ?? defaultSettings.readmeDownloadRatio),
    language: normalizeLanguage(value.language),
  }
}

export function loadAppSettings() {
  try {
    const stored = localStorage.getItem(settingsStorageKey)
    return normalizeSettings(stored ? JSON.parse(stored) as Partial<AppSettings> : {})
  } catch {
    return defaultSettings
  }
}

export function loadAccessToken() {
  try {
    return localStorage.getItem(tokenStorageKey) ?? ''
  } catch {
    return ''
  }
}

export async function loadAppStorage(): Promise<AppStorage> {
  if (!hasTauriRuntime()) {
    return {
      accessToken: loadAccessToken(),
      settings: loadAppSettings(),
    }
  }

  const stored = await invoke<AppStorageResponse>('load_app_storage')
  return {
    accessToken: stored.access_token ?? '',
    settings: normalizeSettings(stored.settings ?? {}),
  }
}

export function saveAppSettingsLocal(settings: AppSettings) {
  const nextSettings = normalizeSettings(settings)
  localStorage.setItem(settingsStorageKey, JSON.stringify(nextSettings))
  return nextSettings
}

export async function saveAppSettings(settings: AppSettings) {
  const nextSettings = normalizeSettings(settings)
  if (hasTauriRuntime()) {
    await invoke('save_app_settings', { request: { settings: nextSettings } })
  } else {
    localStorage.setItem(settingsStorageKey, JSON.stringify(nextSettings))
  }
  return nextSettings
}

export async function saveAccessToken(accessToken: string) {
  const nextToken = accessToken.trim()
  if (hasTauriRuntime()) {
    await invoke('save_access_token', { request: { access_token: nextToken } })
  } else if (nextToken) {
    localStorage.setItem(tokenStorageKey, nextToken)
  } else {
    localStorage.removeItem(tokenStorageKey)
  }
  return nextToken
}

export function resetAppSettings() {
  localStorage.setItem(settingsStorageKey, JSON.stringify(defaultSettings))
  return defaultSettings
}
