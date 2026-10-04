const HF_ENDPOINT = 'https://huggingface.co'

function authHeaders(token?: string) {
  return token ? { Authorization: `Bearer ${token}` } : undefined
}

function normalizeEndpoint(endpoint?: string) {
  return (endpoint || HF_ENDPOINT).replace(/\/+$/, '')
}

export type HfModelSummary = {
  id: string
  author?: string
  downloads?: number
  lastModified?: string
  likes?: number
  pipeline_tag?: string
  tags?: string[]
}

export type HfModelDetails = HfModelSummary & {
  cardData?: {
    license?: string
    tags?: string[]
  }
  gated?: boolean | string
  library_name?: string
  siblings?: Array<{ rfilename: string }>
  usedStorage?: number
}

export type HfTreeEntry = {
  lfs?: {
    oid: string
    size: number
  }
  path: string
  size: number
  type: 'directory' | 'file'
}

export type RepoResult = {
  downloads: string
  id: string
  likes: string
  size: string
  tags: string[]
  title: string
  updated: string
}

export type FileTreeRow = {
  depth: number
  id: string
  name: string
  parent?: string
  path: string
  sha256?: string
  size: number
  type: 'file' | 'folder'
}

const modelFileExtensions = new Set([
  '.safetensors',
  '.ckpt',
  '.pt',
  '.pth',
  '.bin',
  '.gguf',
  '.onnx',
  '.vae',
])

export function extractRepoId(value: string) {
  const trimmed = value.trim()
  const match = trimmed.match(/huggingface\.co\/(?:models\/)?([^/?#]+\/[^/?#]+)/i)
  return match?.[1] ?? trimmed
}

export function formatBytes(value?: number) {
  if (value === undefined) {
    return '-'
  }
  if (value < 1) {
    return '0 B'
  }

  if (value < 1024) {
    return `${Math.round(value)} Bytes`
  }

  const units = ['kB', 'MB', 'GB', 'TB']
  let size = value
  let index = -1
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }

  return `${Number(size.toFixed(2))} ${units[index]}`
}

export function formatCount(value?: number) {
  if (!value) {
    return '0'
  }

  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`
  }

  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`
  }

  return String(value)
}

export function formatRelativeDate(value?: string) {
  if (!value) {
    return 'unknown'
  }

  const timestamp = new Date(value).getTime()
  const days = Math.max(0, Math.floor((Date.now() - timestamp) / 86_400_000))
  if (days < 1) {
    return 'today'
  }
  if (days < 30) {
    return `${days} days ago`
  }
  if (days < 365) {
    return `${Math.floor(days / 30)} months ago`
  }
  return `${Math.floor(days / 365)} years ago`
}

export function isModelFile(path: string) {
  const lower = path.toLowerCase()
  return Array.from(modelFileExtensions).some((extension) => lower.endsWith(extension))
}

export function makeShortSubfolder(repoId: string) {
  const name = repoId.split('/').pop() ?? repoId
  return name
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9.-]+/g, '')
    .replace(/-{2,}/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase()
    .slice(0, 28)
}

export function toRepoResult(model: HfModelSummary | HfModelDetails): RepoResult {
  const tags = [
    model.pipeline_tag,
    ...(model.tags ?? []).filter((tag) => !tag.startsWith('region:') && !tag.includes(':')).slice(0, 4),
  ].filter(Boolean) as string[]

  return {
    downloads: formatCount(model.downloads),
    id: model.id,
    likes: formatCount(model.likes),
    size: formatBytes((model as HfModelDetails).usedStorage),
    tags: Array.from(new Set(tags)).slice(0, 3),
    title: model.pipeline_tag ?? (model as HfModelDetails).library_name ?? 'model repository',
    updated: formatRelativeDate(model.lastModified),
  }
}

export function toFileTree(entries: HfTreeEntry[]) {
  const folders = new Map<string, FileTreeRow>()
  const files = new Map<string, FileTreeRow>()

  for (const entry of entries) {
    const parts = entry.path.split('/')
    parts.slice(0, -1).forEach((_, index) => {
      const path = parts.slice(0, index + 1).join('/')
      if (!folders.has(path)) {
        folders.set(path, {
          depth: index,
          id: path,
          name: parts[index],
          parent: index > 0 ? parts.slice(0, index).join('/') : undefined,
          path,
          size: 0,
          type: 'folder',
        })
      }
    })

    if (entry.type === 'directory' && !folders.has(entry.path)) {
      folders.set(entry.path, {
        depth: parts.length - 1,
        id: entry.path,
        name: parts.at(-1) ?? entry.path,
        parent: parts.length > 1 ? parts.slice(0, -1).join('/') : undefined,
        path: entry.path,
        size: 0,
        type: 'folder',
      })
    }

    if (entry.type === 'file') {
      files.set(entry.path, {
        depth: parts.length - 1,
        id: entry.path,
        name: parts.at(-1) ?? entry.path,
        parent: parts.length > 1 ? parts.slice(0, -1).join('/') : undefined,
        path: entry.path,
        sha256: entry.lfs?.oid,
        size: entry.lfs?.size ?? entry.size,
        type: 'file',
      })
    }
  }

  const byParent = new Map<string, FileTreeRow[]>()
  for (const row of [...folders.values(), ...files.values()]) {
    const parent = row.parent ?? ''
    byParent.set(parent, [...(byParent.get(parent) ?? []), row])
  }

  for (const rows of byParent.values()) {
    rows.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'folder' ? -1 : 1
      }
      return a.name.localeCompare(b.name)
    })
  }

  const ordered: FileTreeRow[] = []
  const visit = (parent = '') => {
    for (const row of byParent.get(parent) ?? []) {
      ordered.push(row)
      if (row.type === 'folder') {
        visit(row.path)
      }
    }
  }

  visit()
  return ordered
}

async function getJson<T>(url: string, signal?: AbortSignal, token?: string) {
  const response = await fetch(url, { headers: authHeaders(token), signal })
  if (!response.ok) {
    throw new Error(`HF request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export async function searchModels(query: string, signal?: AbortSignal, token?: string, endpoint?: string) {
  const search = encodeURIComponent(extractRepoId(query))
  const results = await getJson<HfModelSummary[]>(
    `${normalizeEndpoint(endpoint)}/api/models?search=${search}&limit=12&sort=downloads&direction=-1`,
    signal,
    token,
  )
  return results.map(toRepoResult)
}

export async function getModelDetails(repoId: string, signal?: AbortSignal, token?: string, endpoint?: string) {
  return getJson<HfModelDetails>(`${normalizeEndpoint(endpoint)}/api/models/${repoId}`, signal, token)
}

export async function getModelReadme(repoId: string, signal?: AbortSignal, token?: string, endpoint?: string) {
  const response = await fetch(`${normalizeEndpoint(endpoint)}/${repoId}/raw/main/README.md`, {
    headers: authHeaders(token),
    signal,
  })
  if (response.status === 401 || response.status === 403) {
    return {
      content: '',
      restricted: true,
    }
  }
  if (response.status === 404) {
    return {
      content: '',
      restricted: false,
    }
  }
  if (!response.ok) {
    throw new Error(`README request failed: ${response.status}`)
  }
  return {
    content: await response.text(),
    restricted: false,
  }
}

export async function getModelTree(repoId: string, signal?: AbortSignal, token?: string, endpoint?: string) {
  const entries = await getJson<HfTreeEntry[]>(
    `${normalizeEndpoint(endpoint)}/api/models/${repoId}/tree/main?recursive=1`,
    signal,
    token,
  )
  return toFileTree(entries)
}
