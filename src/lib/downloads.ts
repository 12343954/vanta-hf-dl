import type { AppSettings } from './settings'
import type { FileTreeRow } from './huggingface'

export const downloadQueueStorageKey = 'vanta:download-queue'

export type DownloadStatus = 'queued' | 'downloading' | 'paused' | 'completed' | 'failed' | 'error' | 'canceled'

export type DownloadJob = {
  id: string
  repoId: string
  filePath: string
  fileName: string
  size: number
  downloadedBytes: number
  status: DownloadStatus
  sourceUrl: string
  targetPath: string
  createdAt: string
  updatedAt: string
  sha256?: string
  error?: string
  warning?: string
}

const knownModelRoots = new Set([
  'checkpoints',
  'clip',
  'clip_vision',
  'configs',
  'controlnet',
  'diffusion_models',
  'embeddings',
  'gligen',
  'hypernetworks',
  'latent_upscale_models',
  'loras',
  'model_patches',
  'photomaker',
  'style_models',
  'text_encoders',
  'unet',
  'upscale_models',
  'vae',
  'vae_approx',
])

function normalizeEndpoint(endpoint: string) {
  return endpoint.replace(/\/+$/, '')
}

function joinWindowsPath(...parts: string[]) {
  return parts
    .filter(Boolean)
    .join('\\')
    .replace(/[\\/]+/g, '\\')
}

function makeDownloadId(repoId: string, filePath: string, subfolder: string) {
  return `${repoId}:${subfolder}:${filePath}`
}

export function getDownloadProgress(job: DownloadJob) {
  if (job.size < 1) {
    return job.status === 'completed' ? 100 : 0
  }
  return Math.min(100, Math.round((job.downloadedBytes / job.size) * 100))
}

export function buildTargetPath(settings: AppSettings, filePath: string, subfolder: string) {
  const parts = filePath.split('/')
  const root = knownModelRoots.has(parts[0]) ? parts.shift() ?? '' : ''
  return joinWindowsPath(settings.modelsRoot, root, subfolder, ...parts)
}

export function buildSourceUrl(endpoint: string, repoId: string, filePath: string) {
  return `${normalizeEndpoint(endpoint)}/${repoId}/resolve/main/${filePath.split('/').map(encodeURIComponent).join('/')}`
}

export function createDownloadJobs(params: {
  endpoint: string
  files: FileTreeRow[]
  repoId: string
  settings: AppSettings
  subfolder: string
}) {
  const now = new Date().toISOString()
  return params.files.map<DownloadJob>((file) => ({
    createdAt: now,
    downloadedBytes: 0,
    fileName: file.name,
    filePath: file.path,
    id: makeDownloadId(params.repoId, file.path, params.subfolder),
    repoId: params.repoId,
    sha256: file.sha256,
    size: file.size,
    sourceUrl: buildSourceUrl(params.endpoint, params.repoId, file.path),
    status: 'queued',
    targetPath: buildTargetPath(params.settings, file.path, params.subfolder),
    updatedAt: now,
  }))
}

export function loadDownloadQueue() {
  try {
    const stored = localStorage.getItem(downloadQueueStorageKey)
    const jobs = stored ? JSON.parse(stored) as DownloadJob[] : []
    return jobs.map((job) => (
      job.status === 'downloading' || job.status === 'queued'
        ? { ...job, status: 'paused' as const, updatedAt: new Date().toISOString() }
        : job
    ))
  } catch {
    return []
  }
}

export function saveDownloadQueue(jobs: DownloadJob[]) {
  localStorage.setItem(downloadQueueStorageKey, JSON.stringify(jobs))
}
