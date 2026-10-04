import {
  useEffect,
  useCallback,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import remarkGfm from 'remark-gfm'
import {
  Bookmark,
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Cloud,
  Copy,
  Database,
  Download,
  File,
  Folder,
  FolderOpen,
  HardDrive,
  Heart,
  History,
  ExternalLink,
  KeyRound,
  Minus,
  Pause,
  Play,
  RotateCcw,
  Search,
  Settings,
  Shield,
  SlidersHorizontal,
  Square,
  Star,
  Tags,
  Terminal,
  Trash2,
  Clock3,
  X,
} from 'lucide-react'
import { motion } from 'framer-motion'
import './App.css'
import {
  extractRepoId,
  formatBytes,
  formatRelativeDate,
  getModelDetails,
  getModelReadme,
  getModelTree,
  makeShortSubfolder,
  isModelFile,
  searchModels,
  toRepoResult,
  type FileTreeRow,
  type RepoResult,
} from './lib/huggingface'
import {
  defaultSettings,
  loadAccessToken,
  loadAppSettings,
  loadAppStorage,
  saveAccessToken,
  saveAppSettings,
  saveAppSettingsLocal,
  type AppSettings,
} from './lib/settings'
import { createT, languageOptions } from './lib/i18n'
import { buildSourceUrl, createDownloadJobs, getDownloadProgress, loadDownloadQueue, saveDownloadQueue, type DownloadJob } from './lib/downloads'
import {
  cancelDownloadJob,
  destroyWindow,
  enqueueDownloads,
  getBackendStatus,
  hasTauriRuntime,
  listActiveDownloadJobs,
  listDownloadJobs,
  listenNativeCloseRequested,
  listenWindowCloseRequested,
  listenDownloadProgress,
  loadFavorites,
  minimizeWindow,
  openPath,
  openUrl,
  pauseDownloadJob,
  pickFolder,
  refreshDownloadSources,
  removeDownloadJob as removeDesktopDownloadJob,
  resumeDownloadJob,
  saveFavorites,
  minimizeNativeWindow,
  toggleMaximizeWindow,
  updateTraySummary,
  type BackendStatus,
  type FavoriteRepo,
} from './lib/desktop'

type View = 'explore' | 'favorites' | 'history' | 'settings'
type TreeFilter = 'all' | 'models' | 'bf16' | 'fp16' | 'fp8' | 'int8' | 'nvfp4' | 'q4km' | 'gguf' | 'safetensors' | 'noDocs'
type MultiTreeFilter = Exclude<TreeFilter, 'all' | 'models' | 'noDocs'>
type LocalSortKey = 'name' | 'time' | 'size'
type SortDirection = 'asc' | 'desc'
type ClosePreference = 'exit' | 'minimize' | 'cancel'
type ClosePreferenceDraft = ClosePreference | ''
type SettingsDropdown = 'close' | 'language' | 'localSort' | ''

const githubRepo = '12343954/vanta-hf-dl'
const githubLatestReleaseUrl = `https://github.com/${githubRepo}/releases/latest`
const githubLatestApiUrl = `https://api.github.com/repos/${githubRepo}/releases/latest`
const currentAppVersion = '1.0.0'

const autoRetryStorageKey = 'vanta:auto-retry-groups'
const closePreferenceStorageKey = 'vanta:close-preference'
const activeViewStorageKey = 'vanta:active-view'

const closeBehaviorOptions: Array<{ labelKey: 'closeAsk' | 'closeExit' | 'closeMinimize', value: ClosePreferenceDraft }> = [
  { labelKey: 'closeAsk', value: '' },
  { labelKey: 'closeExit', value: 'exit' },
  { labelKey: 'closeMinimize', value: 'minimize' },
]

type DownloadGroup = {
  activeCount: number
  completedCount: number
  downloadedBytes: number
  failedCount: number
  id: string
  jobs: DownloadJob[]
  path: string
  progress: number
  repoId: string
  size: number
  sortAt: string
  speedBytesPerSecond: number
  status: DownloadJob['status']
  title: string
}

const navItems: Array<{ icon: typeof Search, id: View }> = [
  { icon: Search, id: 'explore' },
  { icon: Bookmark, id: 'favorites' },
  { icon: History, id: 'history' },
  { icon: Settings, id: 'settings' },
]

const viewIds = new Set<View>(navItems.map((item) => item.id))

const searchHistoryStorageKey = 'vanta:search-history'
const favoritesStorageKey = 'vanta:favorites'
const quickStartRepos = [
  'optimum-internal-testing/sentence-transformers-stsb-bert-tiny',
  'openai/whisper-tiny',
  'stabilityai/sd-vae-ft-mse',
]
const documentFilePattern = /\.(md|markdown|txt|rst|json|yaml|yml|csv|tsv|png|jpe?g|gif|webp|svg)$/i

const localTags = ['Tested', 'Recommended']
const mirrorOptions = [
  { label: 'Official', value: 'https://huggingface.co' },
  { label: 'HF Mirror', value: 'https://hf-mirror.com' },
  { label: 'hf.co', value: 'https://hf.co' },
]

const treeFilterOptions: TreeFilter[] = ['all', 'models', 'bf16', 'fp16', 'fp8', 'int8', 'nvfp4', 'q4km', 'gguf', 'safetensors', 'noDocs']

const multiTreeFilters = new Set<TreeFilter>(['bf16', 'fp16', 'fp8', 'int8', 'nvfp4', 'q4km', 'gguf', 'safetensors'])

function clampReadmeRatio(value: number) {
  return Math.min(70, Math.max(21, value))
}

function stripReadmeFrontmatter(value: string) {
  return value.replace(/^---[\s\S]*?---\s*/, '').trim()
}

function resolveReadmeAsset(repoId: string, value?: string, endpoint = 'https://huggingface.co') {
  if (!value || value.startsWith('http') || value.startsWith('data:')) {
    return value
  }

  const cleanPath = value.replace(/^\.\//, '')
  return `${endpoint.replace(/\/+$/, '')}/${repoId}/resolve/main/${cleanPath}`
}

const downloadStatusRank: Record<DownloadJob['status'], number> = {
  queued: 0,
  downloading: 1,
  paused: 2,
  error: 2,
  failed: 2,
  canceled: 3,
  completed: 4,
}

function mergeDownloadJobs(current: DownloadJob[], incoming: DownloadJob[]) {
  const currentMap = new Map(current.map((job) => [job.id, job]))
  const incomingIds = new Set(incoming.map((job) => job.id))
  const merged = incoming.map((job) => {
    const existing = currentMap.get(job.id)
    if (!existing) {
      return job
    }

    if (
      downloadStatusRank[job.status] < downloadStatusRank[existing.status]
      && job.downloadedBytes <= existing.downloadedBytes
    ) {
      return existing
    }

    return {
      ...job,
      downloadedBytes: Math.max(job.downloadedBytes, existing.downloadedBytes),
    }
  })

  return [...merged, ...current.filter((job) => !incomingIds.has(job.id))]
}

function getDirectoryPath(path: string) {
  const normalized = path.replace(/\//g, '\\')
  const index = normalized.lastIndexOf('\\')
  return index >= 0 ? normalized.slice(0, index) : normalized
}

function getDirectoryName(path: string) {
  const normalized = path.replace(/[\\/]+$/, '')
  const index = normalized.lastIndexOf('\\')
  return index >= 0 ? normalized.slice(index + 1) : normalized
}

function getGroupStatus(jobs: DownloadJob[]): DownloadJob['status'] {
  if (jobs.some((job) => job.status === 'downloading')) {
    return 'downloading'
  }
  if (jobs.some((job) => job.status === 'queued')) {
    return 'queued'
  }
  if (jobs.some((job) => job.status === 'failed' || job.status === 'error')) {
    return 'failed'
  }
  if (jobs.some((job) => job.status === 'paused')) {
    return 'paused'
  }
  if (jobs.every((job) => job.status === 'completed')) {
    return 'completed'
  }
  if (jobs.every((job) => job.status === 'canceled')) {
    return 'canceled'
  }
  return 'queued'
}

function formatSpeed(bytesPerSecond: number) {
  return bytesPerSecond > 0 ? `${formatBytes(bytesPerSecond)}/s` : ''
}

function loadAutoRetryGroupIds() {
  try {
    const stored = localStorage.getItem(autoRetryStorageKey)
    return new Set<string>(stored ? JSON.parse(stored) : [])
  } catch {
    return new Set<string>()
  }
}

function loadClosePreference(): ClosePreferenceDraft {
  const stored = localStorage.getItem(closePreferenceStorageKey)
  return stored === 'exit' || stored === 'minimize' || stored === 'cancel' ? stored : ''
}

function loadActiveView(): View {
  try {
    const stored = localStorage.getItem(activeViewStorageKey)
    return viewIds.has(stored as View) ? stored as View : 'explore'
  } catch {
    return 'explore'
  }
}

function isFinishedStatus(status: DownloadJob['status']) {
  return status === 'completed' || status === 'canceled'
}

function normalizeVersion(value: string) {
  return value.trim().replace(/^v/i, '').split(/[+-]/)[0]
}

function compareVersions(left: string, right: string) {
  const leftParts = normalizeVersion(left).split('.').map((part) => Number.parseInt(part, 10) || 0)
  const rightParts = normalizeVersion(right).split('.').map((part) => Number.parseInt(part, 10) || 0)
  const length = Math.max(leftParts.length, rightParts.length)
  for (let index = 0; index < length; index += 1) {
    const diff = (leftParts[index] ?? 0) - (rightParts[index] ?? 0)
    if (diff !== 0) {
      return diff
    }
  }
  return 0
}

function compareDownloadJobs(a: DownloadJob, b: DownloadJob) {
  const finishedCompare = Number(isFinishedStatus(a.status)) - Number(isFinishedStatus(b.status))
  if (finishedCompare !== 0) {
    return finishedCompare
  }
  const timeCompare = b.createdAt.localeCompare(a.createdAt)
  return timeCompare === 0 ? a.filePath.localeCompare(b.filePath) : timeCompare
}

function compareLocalGroups(a: DownloadGroup, b: DownloadGroup, sortKey: LocalSortKey, direction: SortDirection) {
  const finishedCompare = Number(isFinishedStatus(a.status)) - Number(isFinishedStatus(b.status))
  if (finishedCompare !== 0) {
    return finishedCompare
  }

  const multiplier = direction === 'asc' ? 1 : -1
  if (sortKey === 'name') {
    return a.title.localeCompare(b.title) * multiplier
  }
  if (sortKey === 'size') {
    return (a.size - b.size) * multiplier
  }
  return a.sortAt.localeCompare(b.sortAt) * multiplier
}

function getCompletedCardSpan(fileName: string) {
  const length = fileName.length
  if (length > 56) {
    return 3
  }
  if (length > 24) {
    return 2
  }
  return 1
}

function getRepoCardWidth(repo: RepoResult) {
  const length = Math.max(repo.id.length, repo.title.length)
  return Math.min(760, Math.max(360, length * 8 + 170))
}

function getNavLabel(id: View, t: ReturnType<typeof createT>) {
  if (id === 'favorites') {
    return t('navFavorites')
  }
  if (id === 'history') {
    return t('navLocalModels')
  }
  if (id === 'settings') {
    return t('navSettings')
  }
  return t('navExplore')
}

function getTreeFilterLabel(filter: TreeFilter, t: ReturnType<typeof createT>) {
  if (filter === 'all') {
    return t('filterAll')
  }
  if (filter === 'models') {
    return t('filterModels')
  }
  if (filter === 'noDocs') {
    return t('filterNoDocs')
  }
  if (filter === 'safetensors') {
    return 'SafeTensors'
  }
  if (filter === 'q4km') {
    return 'Q4_K_M'
  }
  return filter
}

function getStatusLabel(status: DownloadJob['status'], t: ReturnType<typeof createT>) {
  return status === 'completed' ? t('completed') : status
}

function renderBreakableFileName(fileName: string) {
  return fileName.split(/([_.-])/g).map((part, index) => (
    <span key={`${part}-${index}`}>
      {part}
      {/[_.-]/.test(part) && <wbr />}
    </span>
  ))
}

function loadSearchHistory() {
  try {
    const stored = localStorage.getItem(searchHistoryStorageKey)
    return stored ? JSON.parse(stored) as string[] : []
  } catch {
    return []
  }
}

function saveSearchHistory(history: string[]) {
  localStorage.setItem(searchHistoryStorageKey, JSON.stringify(history.slice(0, 12)))
}

function loadFavoritesLocal() {
  try {
    const stored = localStorage.getItem(favoritesStorageKey)
    return stored ? JSON.parse(stored) as FavoriteRepo[] : []
  } catch {
    return []
  }
}

function saveFavoritesLocal(favorites: FavoriteRepo[]) {
  localStorage.setItem(favoritesStorageKey, JSON.stringify(favorites))
}

function matchTreeFilter(row: FileTreeRow, filter: TreeFilter) {
  const lowerPath = row.path.toLowerCase()
  if (filter === 'all') {
    return true
  }
  if (filter === 'models') {
    return isModelFile(row.path)
  }
  if (filter === 'noDocs') {
    return !documentFilePattern.test(row.path)
  }
  if (filter === 'safetensors') {
    return lowerPath.endsWith('.safetensors')
  }
  if (filter === 'gguf') {
    return lowerPath.endsWith('.gguf')
  }
  if (filter === 'q4km') {
    return /(^|[-_.])q4[-_.]k[-_.]m($|[-_.])/i.test(row.path)
  }
  return new RegExp(`(^|[-_.])${filter}($|[-_.])`, 'i').test(row.path)
}

function getSelectableFileIds(rows: FileTreeRow[], filters: TreeFilter[]) {
  const activeFilters = filters.length > 0 ? filters : ['all' as const]
  return rows
    .filter((row) => {
      if (row.type !== 'file') {
        return false
      }
      return activeFilters.some((filter) => matchTreeFilter(row, filter))
    })
    .map((row) => row.id)
}

function App() {
  const viewerRef = useRef<HTMLDivElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const speedSamplesRef = useRef<Record<string, { bytes: number, time: number }>>({})
  const totalSpeedSampleRef = useRef<{ bytes: number, time: number } | undefined>(undefined)
  const autoRetryCooldownRef = useRef<Record<string, number>>({})
  const activeDownloadCountRef = useRef(0)
  const closePreferenceRef = useRef<ClosePreference | ''>(loadClosePreference())
  const closingByAppRef = useRef(false)
  const toastTimerRef = useRef<number | undefined>(undefined)
  const [settings, setSettings] = useState(loadAppSettings)
  const [settingsDraft, setSettingsDraft] = useState<AppSettings>(settings)
  const t = useMemo(() => createT(settings.language), [settings.language])
  const [activeView, setActiveView] = useState<View>(loadActiveView)
  const [readmeRatio, setReadmeRatio] = useState(settings.readmeDownloadRatio)
  const [query, setQuery] = useState('')
  const [repoCards, setRepoCards] = useState<RepoResult[]>([])
  const [selectedRepoId, setSelectedRepoId] = useState('')
  const [modelTags, setModelTags] = useState<string[]>([])
  const [repoUpdated, setRepoUpdated] = useState('')
  const [readme, setReadme] = useState('')
  const [isReadmeRestricted, setIsReadmeRestricted] = useState(false)
  const [treeRows, setTreeRows] = useState<FileTreeRow[]>([])
  const [subfolderName, setSubfolderName] = useState('')
  const [searchHistory, setSearchHistory] = useState(loadSearchHistory)
  const [treeFilter, setTreeFilter] = useState<TreeFilter>('all')
  const [multiTreeSelection, setMultiTreeSelection] = useState<Set<MultiTreeFilter>>(() => new Set())
  const [openFolders, setOpenFolders] = useState(() => new Set<string>())
  const [openDownloadGroups, setOpenDownloadGroups] = useState(() => new Set<string>())
  const [checkedRows, setCheckedRows] = useState(() => new Set<string>())
  const [downloadJobs, setDownloadJobs] = useState(loadDownloadQueue)
  const [downloadSpeeds, setDownloadSpeeds] = useState<Record<string, number>>({})
  const [totalDownloadSpeed, setTotalDownloadSpeed] = useState(0)
  const [autoRetryGroupIds, setAutoRetryGroupIds] = useState(loadAutoRetryGroupIds)
  const [downloadActionFeedback, setDownloadActionFeedback] = useState<Record<string, 'copy' | 'open'>>({})
  const [pendingRemoveJobId, setPendingRemoveJobId] = useState('')
  const [pendingCloseRequest, setPendingCloseRequest] = useState(false)
  const [rememberCloseChoice, setRememberCloseChoice] = useState(false)
  const [closePreference, setClosePreference] = useState<ClosePreference | ''>(() => closePreferenceRef.current)
  const [closePreferenceDraft, setClosePreferenceDraft] = useState<ClosePreferenceDraft>(() => closePreferenceRef.current)
  const [settingsDropdown, setSettingsDropdown] = useState<SettingsDropdown>('')
  const [toastMessage, setToastMessage] = useState('')
  const [localModelsQuery, setLocalModelsQuery] = useState('')
  const [localSortKey, setLocalSortKey] = useState<LocalSortKey>('time')
  const [localSortDirection, setLocalSortDirection] = useState<SortDirection>('desc')
  const [favoritesQuery, setFavoritesQuery] = useState('')
  const [favorites, setFavorites] = useState(loadFavoritesLocal)
  const [accessToken, setAccessToken] = useState(loadAccessToken)
  const [draftToken, setDraftToken] = useState('')
  const [isTokenDialogOpen, setIsTokenDialogOpen] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    downloader: 'browser-preview',
    ready: false,
    storage: 'localStorage',
  })
  const [hasUpdate, setHasUpdate] = useState(false)
  const [error, setError] = useState('')

  const folderRows = treeRows.filter((row) => row.type === 'folder')
  const folderIds = new Set(folderRows.map((row) => row.id))
  const visibleTreeRows = treeRows.filter((row) => {
    if (row.depth === 0) {
      return true
    }
    const ancestors = row.path.split('/').slice(0, -1)
    return ancestors.every((_, index) => openFolders.has(ancestors.slice(0, index + 1).join('/')))
  })
  const selectedFiles = treeRows.filter((row) => row.type === 'file' && checkedRows.has(row.id))
  const selectedBytes = selectedFiles.reduce((total, row) => total + row.size, 0)
  const selectedSize = selectedBytes > 0 ? formatBytes(selectedBytes) : '0 B'
  const activeJobs = downloadJobs.filter((job) => job.status === 'downloading' || job.status === 'queued')
  const completedJobs = downloadJobs.filter((job) => job.status === 'completed')
  const hasDownloadingJobs = downloadJobs.some((job) => job.status === 'downloading')
  const hasQueuedJobs = downloadJobs.some((job) => job.status === 'queued')
  const hasResumableJobs = downloadJobs.some((job) => job.status === 'paused' || job.status === 'failed' || job.status === 'error')
  const globalControlMode = hasDownloadingJobs ? 'pause' : hasResumableJobs ? 'start' : hasQueuedJobs ? 'stop' : null
  const sampledSpeed = Object.values(downloadSpeeds).reduce((total, speed) => total + speed, 0)
  const overallSpeed = Math.max(totalDownloadSpeed, sampledSpeed)

  const checkForUpdates = useCallback(async () => {
    try {
      const response = await fetch(githubLatestApiUrl, {
        headers: { Accept: 'application/vnd.github+json' },
      })
      if (!response.ok) {
        return
      }
      const release = await response.json() as { tag_name?: string }
      setHasUpdate(compareVersions(release.tag_name ?? '', currentAppVersion) > 0)
    } catch {
      setHasUpdate(false)
    }
  }, [])

  const openLatestRelease = () => {
    setHasUpdate(false)
    void openUrl(githubLatestReleaseUrl)
  }

  useEffect(() => {
    void checkForUpdates()
  }, [checkForUpdates])

  const applyClosePreference = useCallback((preference: ClosePreference) => {
    if (preference === 'cancel') {
      return
    }

    if (preference === 'minimize') {
      void minimizeWindow()
      return
    }

    closingByAppRef.current = true
    void destroyWindow()
  }, [])

  const handleCloseRequest = useCallback(() => {
    const remembered = closePreferenceRef.current
    const hasActiveDownloads = activeDownloadCountRef.current > 0
    if (!hasActiveDownloads) {
      if (remembered) {
        applyClosePreference(remembered)
      } else {
        closingByAppRef.current = true
        void destroyWindow()
      }
      return
    }

    if (remembered) {
      applyClosePreference(remembered)
      return
    }

    setRememberCloseChoice(false)
    setPendingCloseRequest(true)
  }, [applyClosePreference])

  const downloadGroups = useMemo<DownloadGroup[]>(() => {
    const grouped = new Map<string, DownloadJob[]>()
    for (const job of downloadJobs) {
      const directory = getDirectoryPath(job.targetPath)
      grouped.set(directory, [...(grouped.get(directory) ?? []), job])
    }

    return Array.from(grouped.entries())
      .map(([path, jobs]) => {
        const size = jobs.reduce((total, job) => total + job.size, 0)
        const downloadedBytes = jobs.reduce((total, job) => total + job.downloadedBytes, 0)
        const sortedJobs = [...jobs].sort(compareDownloadJobs)
        return {
          activeCount: jobs.filter((job) => job.status === 'downloading' || job.status === 'queued').length,
          completedCount: jobs.filter((job) => job.status === 'completed').length,
          downloadedBytes,
          failedCount: jobs.filter((job) => job.status === 'failed' || job.status === 'error').length,
          id: path,
          jobs: sortedJobs,
          path,
          progress: size > 0 ? Math.min(100, Math.round((downloadedBytes / size) * 100)) : 0,
          repoId: jobs[0]?.repoId ?? '',
          size,
          sortAt: jobs.reduce((latest, job) => job.createdAt > latest ? job.createdAt : latest, ''),
          speedBytesPerSecond: jobs.reduce((total, job) => total + (downloadSpeeds[job.id] ?? 0), 0),
          status: getGroupStatus(jobs),
          title: getDirectoryName(path),
        }
      })
  }, [downloadJobs, downloadSpeeds])
  const visibleDownloadGroups = useMemo(() => {
    const keyword = localModelsQuery.trim().toLowerCase()
    return downloadGroups
      .filter((group) => {
        if (!keyword) {
          return true
        }
        return [
          group.title,
          group.repoId,
          group.path,
          ...group.jobs.map((job) => job.fileName),
          ...group.jobs.map((job) => job.filePath),
        ].some((value) => value.toLowerCase().includes(keyword))
      })
      .sort((a, b) => compareLocalGroups(a, b, localSortKey, localSortDirection))
  }, [downloadGroups, localModelsQuery, localSortDirection, localSortKey])
  const visibleFavorites = useMemo(() => {
    const keyword = favoritesQuery.trim().toLowerCase()
    return favorites
      .filter((favorite) => !keyword || [favorite.id, favorite.title, ...favorite.tags].some((value) => value.toLowerCase().includes(keyword)))
      .sort((a, b) => b.favoritedAt.localeCompare(a.favoritedAt))
  }, [favorites, favoritesQuery])

  const childMap = useMemo(
    () =>
      treeRows.reduce<Record<string, string[]>>((map, row) => {
        if (row.type === 'file') {
          const parts = row.path.split('/')
          parts.slice(0, -1).forEach((_, index) => {
            const folder = parts.slice(0, index + 1).join('/')
            map[folder] = [...(map[folder] ?? []), row.id]
          })
        }
        return map
      }, {}),
    [treeRows],
  )

  const runSearch = useCallback(async (value: string, signal?: AbortSignal) => {
    const repoId = extractRepoId(value)
    if (!repoId) {
      setRepoCards([])
      setSelectedRepoId('')
      setReadme('')
      setTreeRows([])
      setModelTags([])
      setRepoUpdated('')
      setTreeFilter('all')
      setMultiTreeSelection(new Set())
      return
    }
    setIsSearching(true)
    setError('')
    setIsReadmeRestricted(false)
    try {
      const results = await searchModels(repoId, signal, accessToken, settings.hfEndpoint)
      setRepoCards(results)
      if (repoId.includes('/')) {
        setSelectedRepoId(repoId)
      } else if (results[0]) {
        setSelectedRepoId(results[0].id)
      } else {
        setSelectedRepoId('')
      }
    } catch (searchError) {
      if (!signal?.aborted) {
        setError(searchError instanceof Error ? searchError.message : 'Search failed')
      }
    } finally {
      if (!signal?.aborted) {
        setIsSearching(false)
      }
    }
  }, [accessToken, settings.hfEndpoint])

  const loadRepo = useCallback(async (repoId: string, signal?: AbortSignal) => {
    if (!repoId) {
      return
    }
    setError('')
    setSubfolderName(makeShortSubfolder(repoId))
    setOpenFolders(new Set())
    setCheckedRows(new Set())
    setTreeFilter('all')
    setMultiTreeSelection(new Set())
    try {
      const [details, nextReadme, nextTree] = await Promise.all([
        getModelDetails(repoId, signal, accessToken, settings.hfEndpoint),
        getModelReadme(repoId, signal, accessToken, settings.hfEndpoint),
        getModelTree(repoId, signal, accessToken, settings.hfEndpoint),
      ])
      const result = toRepoResult(details)
      setModelTags(Array.from(new Set([...(details.cardData?.tags ?? []), ...(details.tags ?? [])].filter((tag) => !tag.startsWith('region:')))).slice(0, 10))
      setIsReadmeRestricted(nextReadme.restricted)
      setReadme(stripReadmeFrontmatter(nextReadme.content))
      setRepoUpdated(formatRelativeDate(details.lastModified))
      setTreeRows(nextTree)
      setCheckedRows(new Set(getSelectableFileIds(nextTree, ['all'])))
      setOpenFolders(new Set())
      setRepoCards((current) => {
        const exists = current.some((repo) => repo.id === repoId)
        return exists ? current.map((repo) => (repo.id === repoId ? result : repo)) : [result, ...current]
      })
    } catch (loadError) {
      if (!signal?.aborted) {
        setError(loadError instanceof Error ? loadError.message : 'Repo loading failed')
      }
    }
  }, [accessToken, settings.hfEndpoint])

  useEffect(() => {
    void loadAppStorage().then((stored) => {
      setSettings(stored.settings)
      setSettingsDraft(stored.settings)
      setReadmeRatio(stored.settings.readmeDownloadRatio)
      setAccessToken(stored.accessToken)
    })
  }, [])

  useEffect(() => {
    if (!selectedRepoId) {
      return
    }
    const controller = new AbortController()
    // oxlint-disable-next-line react/set-state-in-effect
    void loadRepo(selectedRepoId, controller.signal)
    return () => controller.abort()
  }, [selectedRepoId, loadRepo])

  useEffect(() => {
    void getBackendStatus().then(setBackendStatus)
    void listDownloadJobs().then((jobs) => {
      if (jobs) {
        setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
      }
    })
    void loadFavorites().then((nextFavorites) => {
      if (nextFavorites) {
        setFavorites(nextFavorites)
        saveFavoritesLocal(nextFavorites)
      }
    })
  }, [])

  useEffect(() => {
    if (!hasTauriRuntime()) {
      saveDownloadQueue(downloadJobs)
    }
  }, [downloadJobs])

  useEffect(() => {
    localStorage.setItem(autoRetryStorageKey, JSON.stringify([...autoRetryGroupIds]))
  }, [autoRetryGroupIds])

  useEffect(() => {
    localStorage.setItem(activeViewStorageKey, activeView)
  }, [activeView])

  useEffect(() => {
    if (activeView !== 'settings') {
      return
    }
    setSettingsDraft(settings)
    setClosePreferenceDraft(closePreferenceRef.current)
  }, [activeView, settings])

  useEffect(() => {
    activeDownloadCountRef.current = downloadJobs.filter((job) => job.status === 'downloading' || job.status === 'queued').length
  }, [downloadJobs])

  useEffect(() => {
    closePreferenceRef.current = closePreference
    if (closePreference) {
      localStorage.setItem(closePreferenceStorageKey, closePreference)
    } else {
      localStorage.removeItem(closePreferenceStorageKey)
    }
  }, [closePreference])

  useEffect(() => {
    const summary = [
      'Vanta',
      `${t('active')} ${activeJobs.length}`,
      `${t('completed')} ${completedJobs.length}`,
      `${t('total')} ${downloadJobs.length}`,
      `${t('speed')} ${formatSpeed(overallSpeed) || '0 B/s'}`,
    ].join('\n')
    void updateTraySummary(summary)
  }, [activeJobs.length, completedJobs.length, downloadJobs.length, overallSpeed, t])

  useEffect(() => {
    let unlisten: (() => void) | undefined
    void listenWindowCloseRequested((event) => {
      if (closingByAppRef.current) {
        return
      }
      event.preventDefault()
      handleCloseRequest()
    }).then((nextUnlisten) => {
      unlisten = nextUnlisten
    })

    return () => unlisten?.()
  }, [handleCloseRequest])

  useEffect(() => {
    let unlisten: (() => void) | undefined
    void listenNativeCloseRequested(() => {
      if (!closingByAppRef.current) {
        handleCloseRequest()
      }
    }).then((nextUnlisten) => {
      unlisten = nextUnlisten
    })

    return () => unlisten?.()
  }, [handleCloseRequest])

  useEffect(() => {
    const handleNativeCloseRequested = () => {
      if (!closingByAppRef.current) {
        handleCloseRequest()
      }
    }

    window.addEventListener('vanta-native-close-requested', handleNativeCloseRequested)
    return () => window.removeEventListener('vanta-native-close-requested', handleNativeCloseRequested)
  }, [handleCloseRequest])

  useEffect(() => {
    if (hasTauriRuntime()) {
      return
    }

    const timer = window.setInterval(() => {
      setDownloadJobs((current) => {
        const now = new Date().toISOString()
        let running = current.filter((job) => job.status === 'downloading').length
        let changed = false
        const next = current.map((job) => {
          if (job.status === 'queued' && running < settings.maxConcurrentDownloads) {
            running += 1
            changed = true
            return { ...job, status: 'downloading' as const, updatedAt: now }
          }

          if (job.status !== 'downloading') {
            return job
          }

          const chunk = Math.max(256 * 1024, Math.min(48 * 1024 * 1024, job.size * 0.035))
          const downloadedBytes = Math.min(job.size, job.downloadedBytes + chunk)
          changed = true
          return {
            ...job,
            downloadedBytes,
            status: downloadedBytes >= job.size ? 'completed' as const : job.status,
            updatedAt: now,
          }
        })
        return changed ? next : current
      })
    }, 900)

    return () => window.clearInterval(timer)
  }, [settings.maxConcurrentDownloads])

  useEffect(() => {
    if (!settingsDropdown) {
      return
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSettingsDropdown('')
      }
    }

    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [settingsDropdown])

  useEffect(() => {
    let unlisten: (() => void) | undefined
    void listenDownloadProgress((event) => {
      const now = Date.now()
      const previous = speedSamplesRef.current[event.jobId]
      speedSamplesRef.current[event.jobId] = { bytes: event.downloadedBytes, time: now }
      if (previous) {
        const elapsedSeconds = (now - previous.time) / 1000
        const deltaBytes = Math.max(0, event.downloadedBytes - previous.bytes)
        if (elapsedSeconds > 0.2 && (deltaBytes > 0 || event.status !== 'downloading')) {
          setDownloadSpeeds((current) => ({
            ...current,
            [event.jobId]: event.status === 'downloading' ? Math.round(deltaBytes / elapsedSeconds) : 0,
          }))
        }
      }
      setDownloadJobs((current) =>
        current.map((job) =>
          job.id === event.jobId
            ? {
              ...job,
              downloadedBytes: event.downloadedBytes,
              error: event.error,
              status: event.status,
              updatedAt: new Date().toISOString(),
              warning: event.warning ?? (event.status === 'completed' ? job.warning : undefined),
            }
            : job,
        ),
      )
      if (event.status === 'completed' || event.status === 'failed') {
        setDownloadSpeeds((current) => ({ ...current, [event.jobId]: 0 }))
        void listDownloadJobs().then((jobs) => {
          if (jobs) {
            setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
          }
        })
      }
    }).then((nextUnlisten) => {
      unlisten = nextUnlisten
    })
    return () => unlisten?.()
  }, [])

  useEffect(() => {
    if (!hasTauriRuntime() || activeView !== 'history') {
      return
    }

    const activeTimer = window.setInterval(() => {
      if (activeDownloadCountRef.current < 1) {
        return
      }
      void listActiveDownloadJobs().then((jobs) => {
        if (jobs) {
          setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
        }
      })
    }, 5000)

    const idleTimer = window.setInterval(() => {
      if (activeDownloadCountRef.current > 0) {
        return
      }
      void listDownloadJobs().then((jobs) => {
        if (jobs) {
          setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
        }
      })
    }, 30000)

    return () => {
      window.clearInterval(activeTimer)
      window.clearInterval(idleTimer)
    }
  }, [activeView])

  useEffect(() => {
    const now = Date.now()
    const downloadedBytes = downloadJobs.reduce((total, job) => total + job.downloadedBytes, 0)
    const hasActiveDownload = downloadJobs.some((job) => job.status === 'downloading')
    const previous = totalSpeedSampleRef.current
    totalSpeedSampleRef.current = { bytes: downloadedBytes, time: now }

    if (!previous || !hasActiveDownload) {
      if (!hasActiveDownload) {
        setTotalDownloadSpeed(0)
      }
      return
    }

    const elapsedSeconds = (now - previous.time) / 1000
    const deltaBytes = Math.max(0, downloadedBytes - previous.bytes)
    if (elapsedSeconds > 0.2 && deltaBytes > 0) {
      setTotalDownloadSpeed(Math.round(deltaBytes / elapsedSeconds))
    }
  }, [downloadJobs])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const previous = totalSpeedSampleRef.current
      if (!previous || Date.now() - previous.time > 8000) {
        setTotalDownloadSpeed(0)
      }
    }, 1200)

    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const now = Date.now()
    for (const job of downloadJobs) {
      const groupId = getDirectoryPath(job.targetPath)
      if (!autoRetryGroupIds.has(groupId) || (job.status !== 'failed' && job.status !== 'error')) {
        continue
      }
      if (now - (autoRetryCooldownRef.current[groupId] ?? 0) < 8000) {
        continue
      }

      autoRetryCooldownRef.current[groupId] = now
      if (hasTauriRuntime()) {
        void resumeDownloadJob({
          accessToken,
          jobId: job.id,
          maxConcurrentParts: 2,
          proxy: settings.proxy,
        }).then((jobs) => {
          if (jobs) {
            setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
          }
        })
      } else {
        updateDownloadJob(job.id, { error: undefined, status: 'queued' })
      }
    }
  }, [accessToken, autoRetryGroupIds, downloadJobs, settings.proxy])

  const toggleFolder = (id: string) => {
    setOpenFolders((current) => {
      const next = new Set(current)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleToggleFolder = (id: string) => (event: ReactMouseEvent<HTMLButtonElement>) => {
    event.stopPropagation()
    toggleFolder(id)
  }

  const handleToggleCheck = (id: string) => (event: ReactMouseEvent<HTMLButtonElement>) => {
    event.stopPropagation()
    toggleCheck(id)
  }

  const switchTreeFilter = (filter: TreeFilter) => {
    if (!multiTreeFilters.has(filter)) {
      setTreeFilter(filter)
      setMultiTreeSelection(new Set())
      setCheckedRows(new Set(getSelectableFileIds(treeRows, [filter])))
      return
    }

    setTreeFilter(filter)
    setMultiTreeSelection((current) => {
      const next = new Set(current)
      const typedFilter = filter as MultiTreeFilter
      if (next.has(typedFilter)) {
        next.delete(typedFilter)
      } else {
        next.add(typedFilter)
      }
      const filters = Array.from(next)
      setCheckedRows(new Set(getSelectableFileIds(treeRows, filters)))
      return next
    })
  }

  const scrollTreeFilters = (event: ReactWheelEvent<HTMLDivElement>) => {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
      return
    }
    event.currentTarget.scrollLeft += event.deltaY
  }

  const toggleCheck = (id: string) => {
    const childIds = childMap[id] ?? []
    const targetIds = folderIds.has(id) ? childIds : [id]
    if (targetIds.length === 0) {
      return
    }
    setCheckedRows((current) => {
      const next = new Set(current)
      const shouldCheck = targetIds.some((targetId) => !next.has(targetId))
      targetIds.forEach((targetId) => {
        if (shouldCheck) {
          next.add(targetId)
        } else {
          next.delete(targetId)
        }
      })
      return next
    })
  }

  const scrollResults = (direction: -1 | 1) => {
    resultsRef.current?.scrollBy({ left: direction * 430, behavior: 'smooth' })
  }

  const handleResultsWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    const element = resultsRef.current
    if (!element) {
      return
    }

    const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY
    if (delta === 0) {
      return
    }
    event.preventDefault()
    element.scrollBy({ left: delta * 2, behavior: 'auto' })
  }

  const copyRepoId = () => {
    if (!selectedRepoId) {
      return
    }
    void navigator.clipboard?.writeText(selectedRepoId)
  }

  const rememberSearch = (value: string) => {
    const repoId = extractRepoId(value)
    if (!repoId) {
      return
    }
    setSearchHistory((current) => {
      const next = [repoId, ...current.filter((item) => item !== repoId)].slice(0, 12)
      saveSearchHistory(next)
      return next
    })
  }

  const clearSearchHistory = () => {
    setSearchHistory([])
    saveSearchHistory([])
  }

  const persistFavorites = (nextFavorites: FavoriteRepo[]) => {
    setFavorites(nextFavorites)
    saveFavoritesLocal(nextFavorites)
    void saveFavorites(nextFavorites).catch((favoriteError) => {
      setError(favoriteError instanceof Error ? favoriteError.message : 'Favorites save failed')
    })
  }

  const toggleFavorite = (repoId: string) => {
    const existing = favorites.some((favorite) => favorite.id === repoId)
    if (existing) {
      persistFavorites(favorites.filter((favorite) => favorite.id !== repoId))
      return
    }

    const repo = repoCards.find((card) => card.id === repoId) ?? {
      downloads: '0',
      id: repoId,
      likes: '0',
      size: '-',
      tags: modelTags.slice(0, 3),
      title: modelTags[0] ?? t('modelRepository'),
      updated: repoUpdated || t('unknown'),
    }
    persistFavorites([{ ...repo, favoritedAt: new Date().toISOString() }, ...favorites])
  }

  const selectRepo = (repoId: string) => {
    setSelectedRepoId(repoId)
    setQuery(repoId)
    rememberSearch(repoId)
  }

  const submitSearch = (event: FormEvent) => {
    event.preventDefault()
    rememberSearch(query)
    void runSearch(query)
  }

  const enqueueSelectedDownloads = () => {
    if (!selectedRepoId || selectedFiles.length === 0) {
      return
    }

    const nextJobs = createDownloadJobs({
      endpoint: settings.hfEndpoint,
      files: selectedFiles,
      repoId: selectedRepoId,
      settings,
      subfolder: subfolderName,
    })
    const nextGroups = new Set(nextJobs.map((job) => getDirectoryPath(job.targetPath)))
    setOpenDownloadGroups((current) => new Set([...current, ...nextGroups]))
    setToastMessage(`${t('queuedFiles')} · ${nextJobs.length}`)
    window.setTimeout(() => setToastMessage(''), 2200)
    if (hasTauriRuntime()) {
      void enqueueDownloads({
        accessToken,
        jobs: nextJobs,
        maxConcurrentParts: 2,
        proxy: settings.proxy,
      }).then((jobs) => {
        if (jobs) {
          setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
        }
      }).catch((downloadError) => {
        setError(downloadError instanceof Error ? downloadError.message : 'Download enqueue failed')
      })
    } else {
      setDownloadJobs((current) => {
        const currentIds = new Set(current.map((job) => job.id))
        return [...nextJobs.filter((job) => !currentIds.has(job.id)), ...current]
      })
    }
    setActiveView('history')
  }

  const updateDownloadJob = (id: string, patch: Partial<DownloadJob>) => {
    setDownloadJobs((current) => current.map((job) => (job.id === id ? { ...job, ...patch, updatedAt: new Date().toISOString() } : job)))
  }

  const pauseJob = (id: string) => {
    if (hasTauriRuntime()) {
      void pauseDownloadJob(id).then((jobs) => jobs && setDownloadJobs(jobs))
    } else {
      updateDownloadJob(id, { status: 'paused' })
    }
  }

  const resumeJob = (id: string) => {
    updateDownloadJob(id, { error: undefined, status: 'queued' })
    if (hasTauriRuntime()) {
      void resumeDownloadJob({ accessToken, jobId: id, maxConcurrentParts: 2, proxy: settings.proxy }).then((jobs) => {
        if (jobs) {
          setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
        }
      })
    }
  }

  const toggleAutoRetryGroup = (id: string) => {
    setAutoRetryGroupIds((current) => {
      const next = new Set(current)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const startAllJobs = () => {
    for (const job of downloadJobs) {
      if (job.status === 'paused' || job.status === 'failed' || job.status === 'error') {
        resumeJob(job.id)
      }
    }
  }

  const pauseAllJobs = () => {
    for (const job of downloadJobs) {
      if (job.status === 'downloading' || job.status === 'queued') {
        pauseJob(job.id)
      }
    }
  }

  const stopAllJobs = () => {
    for (const job of downloadJobs) {
      if (job.status !== 'completed' && job.status !== 'canceled') {
        cancelJob(job.id)
      }
    }
  }

  const startDownloadGroup = (group: DownloadGroup) => {
    for (const job of group.jobs) {
      if (job.status === 'paused' || job.status === 'failed' || job.status === 'error') {
        resumeJob(job.id)
      }
    }
  }

  const pauseDownloadGroup = (group: DownloadGroup) => {
    for (const job of group.jobs) {
      if (job.status === 'downloading' || job.status === 'queued') {
        pauseJob(job.id)
      }
    }
  }

  const handleCloseChoice = (preference: ClosePreference) => {
    if (rememberCloseChoice) {
      setClosePreference(preference)
    }
    setPendingCloseRequest(false)
    applyClosePreference(preference)
  }

  const cancelJob = (id: string) => {
    if (hasTauriRuntime()) {
      void cancelDownloadJob(id).then((jobs) => jobs && setDownloadJobs(jobs))
    } else {
      updateDownloadJob(id, { status: 'canceled' })
    }
  }

  const requestRemoveDownloadJob = (id: string) => {
    setPendingRemoveJobId(id)
  }

  const confirmRemoveDownloadJob = () => {
    const id = pendingRemoveJobId
    if (!id) {
      return
    }
    if (hasTauriRuntime()) {
      void removeDesktopDownloadJob(id).then((jobs) => jobs && setDownloadJobs(jobs))
    } else {
      setDownloadJobs((current) => current.filter((job) => job.id !== id))
    }
    setDownloadSpeeds((current) => {
      const next = { ...current }
      delete next[id]
      return next
    })
    setAutoRetryGroupIds((current) => {
      const next = new Set(current)
      const removedJob = downloadJobs.find((job) => job.id === id)
      if (removedJob) {
        next.delete(getDirectoryPath(removedJob.targetPath))
      }
      return next
    })
    delete speedSamplesRef.current[id]
    setPendingRemoveJobId('')
  }

  const toggleDownloadGroup = (id: string) => {
    setOpenDownloadGroups((current) => {
      const next = new Set(current)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const openDownloadDirectory = (path: string) => {
    setDownloadActionFeedback((current) => ({ ...current, [`open:${path}`]: 'open' }))
    window.setTimeout(() => {
      setDownloadActionFeedback((current) => {
        const next = { ...current }
        delete next[`open:${path}`]
        return next
      })
    }, 1000)
    void openPath(path).catch((openError) => {
      setError(openError instanceof Error ? openError.message : 'Open directory failed')
    })
  }

  const copyDownloadUrl = (id: string, url: string) => {
    setDownloadActionFeedback((current) => ({ ...current, [`copy:${id}`]: 'copy' }))
    window.setTimeout(() => {
      setDownloadActionFeedback((current) => {
        const next = { ...current }
        delete next[`copy:${id}`]
        return next
      })
    }, 1000)
    void navigator.clipboard?.writeText(url)
  }

  const openSettings = () => {
    setSettingsDraft(settings)
    setClosePreferenceDraft(closePreferenceRef.current)
    setActiveView('settings')
  }

  const showSettingsSavedToast = () => {
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current)
    }
    setToastMessage(t('settingsSaved'))
    toastTimerRef.current = window.setTimeout(() => setToastMessage(''), 1800)
  }

  const applySettings = (nextDraft: AppSettings, nextClosePreference = closePreferenceDraft) => {
    const endpointChanged = settings.hfEndpoint.replace(/\/+$/, '') !== nextDraft.hfEndpoint.replace(/\/+$/, '')
    const nextSettings = saveAppSettingsLocal(nextDraft)
    setSettingsDraft(nextSettings)
    setSettings(nextSettings)
    setReadmeRatio(nextSettings.readmeDownloadRatio)
    setClosePreference(nextClosePreference)
    showSettingsSavedToast()
    void saveAppSettings(nextSettings).catch((saveError) => {
      setError(saveError instanceof Error ? saveError.message : 'Settings save failed')
    })
    if (endpointChanged) {
      setDownloadJobs((current) =>
        current.map((job) =>
          job.status === 'completed' || job.status === 'canceled'
            ? job
            : { ...job, sourceUrl: buildSourceUrl(nextSettings.hfEndpoint, job.repoId, job.filePath) },
        ),
      )
      void refreshDownloadSources(nextSettings.hfEndpoint).then((jobs) => {
        if (jobs) {
          setDownloadJobs((current) => mergeDownloadJobs(current, jobs))
        }
      }).catch((saveError) => {
        setError(saveError instanceof Error ? saveError.message : 'Download source refresh failed')
      })
    }
  }

  const updateSettingsDraft = <Key extends keyof AppSettings>(key: Key, value: AppSettings[Key]) => {
    const nextDraft = { ...settingsDraft, [key]: value }
    applySettings(nextDraft)
  }

  const updateClosePreferenceDraft = (value: ClosePreferenceDraft) => {
    setClosePreferenceDraft(value)
    applySettings(settingsDraft, value)
  }

  const resetSettings = () => {
    setClosePreferenceDraft('')
    applySettings(defaultSettings, '')
  }

  const browseFolder = async (target: 'comfyRoot' | 'modelsRoot') => {
    const selected = await pickFolder(target === 'comfyRoot' ? 'Select ComfyUI root' : 'Select models root')
    if (!selected) {
      return
    }

    const nextDraft = (() => {
      if (target === 'modelsRoot') {
        return { ...settingsDraft, modelsRoot: selected }
      }

      const shouldUpdateModelsRoot =
        !settingsDraft.modelsRoot || settingsDraft.modelsRoot === `${settingsDraft.comfyRoot}\\models` || settingsDraft.modelsRoot.startsWith(`${settingsDraft.comfyRoot}\\`)
      return {
        ...settingsDraft,
        comfyRoot: selected,
        modelsRoot: shouldUpdateModelsRoot ? `${selected}\\models` : settingsDraft.modelsRoot,
      }
    })()
    applySettings(nextDraft)
  }

  const openTokenDialog = () => {
    setDraftToken(accessToken)
    setIsTokenDialogOpen(true)
  }

  const saveToken = (event: FormEvent) => {
    event.preventDefault()
    const nextToken = draftToken.trim()
    setAccessToken(nextToken)
    setIsTokenDialogOpen(false)
    void saveAccessToken(nextToken).catch((saveError) => {
      setError(saveError instanceof Error ? saveError.message : 'Token save failed')
    })
  }

  const clearToken = () => {
    setAccessToken('')
    setDraftToken('')
    setIsTokenDialogOpen(false)
    void saveAccessToken('').catch((saveError) => {
      setError(saveError instanceof Error ? saveError.message : 'Token clear failed')
    })
  }

  const updateSplit = (clientX: number) => {
    if (!viewerRef.current) {
      return
    }

    const rect = viewerRef.current.getBoundingClientRect()
    const nextRatio = ((clientX - rect.left) / rect.width) * 100
    const clampedRatio = clampReadmeRatio(nextRatio)
    setReadmeRatio(clampedRatio)
    setSettings((current) => {
      const nextSettings = saveAppSettingsLocal({ ...current, readmeDownloadRatio: clampedRatio })
      void saveAppSettings(nextSettings)
      return nextSettings
    })
  }

  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    updateSplit(event.clientX)

    const move = (moveEvent: PointerEvent) => updateSplit(moveEvent.clientX)
    const stop = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', stop)
    }

    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', stop, { once: true })
  }

  const startMouseResize = (event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault()
    updateSplit(event.clientX)

    const move = (moveEvent: MouseEvent) => updateSplit(moveEvent.clientX)
    const stop = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', stop)
    }

    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', stop, { once: true })
  }

  const renderSettingsDropdown = <Value extends string>(
    id: SettingsDropdown,
    value: Value,
    options: Array<{ label: string, value: Value }>,
    onChange: (value: Value) => void,
  ) => {
    const selected = options.find((option) => option.value === value) ?? options[0]
    const isOpen = settingsDropdown === id

    return (
      <div className={`custom-select ${isOpen ? 'open' : ''}`}>
        <button
          type="button"
          className="custom-select-trigger"
          onClick={() => setSettingsDropdown(isOpen ? '' : id)}
        >
          <span>{selected.label}</span>
          <ChevronDown size={18} />
        </button>
        {isOpen && (
          <div className="custom-select-menu">
            {options.map((option) => (
              <button
                type="button"
                className={option.value === value ? 'selected' : ''}
                key={option.value || 'ask'}
                onClick={() => {
                  onChange(option.value)
                  setSettingsDropdown('')
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  const closeSettingsDropdownOutside = (event: ReactPointerEvent<HTMLElement>) => {
    if (!settingsDropdown) {
      return
    }

    const target = event.target
    if (target instanceof Element && !target.closest('.custom-select')) {
      setSettingsDropdown('')
    }
  }

  return (
    <main className="app-shell" onPointerDown={closeSettingsDropdownOutside}>
      <div className="aurora aurora-a" />
      <div className="aurora aurora-b" />
      <div className="grain" />

      <div className="app-titlebar" data-tauri-drag-region>
        <div className="titlebar-brand" data-tauri-drag-region>
          <div className="titlebar-mark">V</div>
          <span>Vanta</span>
        </div>
        <div className="titlebar-actions">
          <button type="button" onClick={() => void minimizeNativeWindow()} aria-label="Minimize window">
            <Minus size={14} />
          </button>
          <button type="button" onClick={() => void toggleMaximizeWindow()} aria-label="Maximize window">
            <Square size={12} />
          </button>
          <button type="button" className="close" onClick={handleCloseRequest} aria-label="Close window">
            <X size={15} />
          </button>
        </div>
      </div>

      <aside className="sidebar glass-panel">
        <div className="brand">
          <div className="brand-mark">V</div>
          <div>
            <button type="button" className={`brand-name ${hasUpdate ? 'has-update' : ''}`} onClick={openLatestRelease}>
              Vanta
              {hasUpdate && <span className="update-dot brand-update-dot" />}
            </button>
            <div className="brand-caption">HF Model Download Hub</div>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              className={`nav-item ${activeView === item.id ? 'active' : ''}`}
              key={item.id}
              onClick={item.id === 'settings' ? openSettings : () => setActiveView(item.id)}
            >
              <item.icon size={18} />
              <span>{getNavLabel(item.id, t)}</span>
            </button>
          ))}
        </nav>

        <div className="side-card">
          <div className="mini-label">{t('storage')}</div>
          <div className="storage-line">
            <HardDrive size={16} />
            <span>{settings.modelsRoot ? t('modelsRootSet') : t('storageNotSet')}</span>
          </div>
          <div className="meter">
            <span />
          </div>
          <div className="muted-row">
            <span>{settings.maxConcurrentDownloads} {t('workers')}</span>
            <span>{settings.hfEndpoint.replace('https://', '')}</span>
          </div>
        </div>
      </aside>

      <section className={`workspace ${activeView === 'explore' ? 'explore-workspace' : ''}`}>
        {activeView === 'settings' ? (
          <header className="settings-topbar">
            <div>
              <div className="mini-label">{t('settingsMini')}</div>
              <h1>{t('settingsTitle')}</h1>
              <p>{t('settingsDesc')}</p>
            </div>
            <div className="settings-actions">
              <button type="button" className="pill ghost" onClick={resetSettings}>
                <RotateCcw size={15} />
                {t('reset')}
              </button>
            </div>
          </header>
        ) : activeView === 'history' ? (
          <header className="settings-topbar">
            <div className="local-title">
              <div className="mini-label">{t('localMini')}</div>
              <h1>{t('localTitle')}</h1>
              <p>{t('localDesc')}</p>
            </div>
            <div className="download-summary-grid compact">
              <section className="queue-stat-card glass-panel">
                <span>{t('active')}/{t('completed')}/{t('total')}</span>
                <strong>{activeJobs.length}/{completedJobs.length}/{downloadJobs.length}</strong>
              </section>
              <section className="download-metric speed-control-card glass-panel">
                <span>{t('speed')}</span>
                <strong>{formatSpeed(overallSpeed) || '0 B/s'}</strong>
                <div className="download-speed-actions">
                  {globalControlMode === 'start' && (
                    <button type="button" className="icon-pill" onClick={startAllJobs} aria-label="Start all downloads">
                      <Play size={15} />
                    </button>
                  )}
                  {globalControlMode === 'pause' && (
                    <button type="button" className="icon-pill" onClick={pauseAllJobs} aria-label="Pause all downloads">
                      <Pause size={15} />
                    </button>
                  )}
                  {globalControlMode === 'stop' && (
                    <button type="button" className="icon-pill" onClick={stopAllJobs} aria-label="Stop all downloads">
                      <X size={15} />
                    </button>
                  )}
                </div>
              </section>
            </div>
          </header>
        ) : activeView === 'favorites' ? (
          <header className="settings-topbar">
            <div>
              <div className="mini-label">{t('favoritesMini')}</div>
              <h1>{t('favoritesTitle')}</h1>
              <p>{t('favoritesDesc')}</p>
            </div>
            <div className="settings-actions">
              <button type="button" className="pill primary" onClick={() => setActiveView('explore')}>
                <Search size={15} />
                {t('browseModels')}
              </button>
            </div>
          </header>
        ) : (
          <header className="topbar">
            <motion.form
              onSubmit={submitSearch}
              className="search-glass"
              whileHover={{ scale: 1.006 }}
              transition={{ type: 'spring', stiffness: 380, damping: 30 }}
            >
              <button className="search-submit" type="submit" aria-label={t('searchAria')}>
                <Search size={20} />
              </button>
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t('searchPlaceholder')} />
              <kbd>HF</kbd>
            </motion.form>

            <div className="top-actions">
              <button className={`pill ghost ${accessToken ? 'active-token' : ''}`} onClick={openTokenDialog}>
                <Shield size={16} />
                {accessToken ? t('tokenSet') : t('token')}
              </button>
              <button className="pill ghost">
                <Cloud size={16} />
                {settings.hfEndpoint.replace('https://', '')}
              </button>
              <button className="pill ghost update-pill" onClick={openLatestRelease} title={t('githubRelease')}>
                <ExternalLink size={16} />
                {t('github')}
              </button>
              <button className="pill primary" onClick={enqueueSelectedDownloads} disabled={!selectedRepoId || selectedFiles.length === 0}>
                <Download size={17} />
                {t('downloadSelected')}
              </button>
            </div>
          </header>
        )}

        {activeView === 'settings' ? (
          <section className="settings-view">
            <form className="settings-form" id="settings-form" onSubmit={(event) => event.preventDefault()}>
              <div className="settings-grid">
                <section className="settings-card">
                  <div className="settings-card-head">
                    <FolderOpen size={18} />
                    <h2>{t('storageRoot')}</h2>
                  </div>
                  <label>
                    {t('comfyRoot')}
                    <div className="path-picker">
                      <input
                        value={settingsDraft.comfyRoot}
                        onChange={(event) => updateSettingsDraft('comfyRoot', event.target.value)}
                        placeholder="G:\ComfyUI-Mie\ComfyUI"
                      />
                      <button type="button" onClick={() => void browseFolder('comfyRoot')} disabled={!hasTauriRuntime()}>
                        {t('browse')}
                      </button>
                    </div>
                  </label>
                  <label>
                    {t('modelsRoot')}
                    <div className="path-picker">
                      <input
                        value={settingsDraft.modelsRoot}
                        onChange={(event) => updateSettingsDraft('modelsRoot', event.target.value)}
                        placeholder="G:\ComfyUI-Mie\ComfyUI\models"
                      />
                      <button type="button" onClick={() => void browseFolder('modelsRoot')} disabled={!hasTauriRuntime()}>
                        {t('browse')}
                      </button>
                    </div>
                  </label>
                  <p>{hasTauriRuntime() ? t('folderHintTauri') : t('folderHintBrowser')}</p>
                </section>

                <section className="settings-card">
                  <div className="settings-card-head">
                    <Cloud size={18} />
                    <h2>Hugging Face</h2>
                  </div>
                  <label>
                    {t('endpointMirror')}
                    <input
                      value={settingsDraft.hfEndpoint}
                      onChange={(event) => updateSettingsDraft('hfEndpoint', event.target.value)}
                      placeholder="https://huggingface.co"
                    />
                  </label>
                  <div className="mirror-options">
                    {mirrorOptions.map((mirror) => (
                      <button
                        type="button"
                        className={settingsDraft.hfEndpoint.replace(/\/+$/, '') === mirror.value ? 'active' : ''}
                        key={mirror.value}
                        onClick={() => updateSettingsDraft('hfEndpoint', mirror.value)}
                      >
                        {mirror.label}
                      </button>
                    ))}
                  </div>
                  <label>
                    {t('proxy')}
                    <input
                      value={settingsDraft.proxy}
                      onChange={(event) => updateSettingsDraft('proxy', event.target.value)}
                      placeholder="http://127.0.0.1:7890"
                    />
                  </label>
                  <button type="button" className="pill ghost settings-token-link" onClick={openTokenDialog}>
                    <Shield size={15} />
                    {accessToken ? t('manageToken') : t('addToken')}
                  </button>
                </section>

                <section className="settings-card">
                  <div className="settings-card-head">
                    <SlidersHorizontal size={18} />
                    <h2>{t('downloadBehavior')}</h2>
                  </div>
                  <label>
                    {t('maxConcurrentDownloads')}
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={settingsDraft.maxConcurrentDownloads}
                      onChange={(event) => updateSettingsDraft('maxConcurrentDownloads', Number(event.target.value))}
                    />
                  </label>
                  <label>
                    {t('defaultSecondaryFolder')}
                    <input
                      value={settingsDraft.defaultSubfolderPattern}
                      onChange={(event) => updateSettingsDraft('defaultSubfolderPattern', event.target.value)}
                      placeholder="{repo}"
                    />
                  </label>
                </section>

                <section className="settings-card">
                  <div className="settings-card-head">
                    <SlidersHorizontal size={18} />
                    <h2>{t('appBehavior')}</h2>
                  </div>
                  <div className="settings-field">
                    <span>{t('closeButtonBehavior')}</span>
                    {renderSettingsDropdown(
                      'close',
                      closePreferenceDraft,
                      closeBehaviorOptions.map((option) => ({ label: t(option.labelKey), value: option.value })),
                      updateClosePreferenceDraft,
                    )}
                  </div>
                  <div className="settings-field">
                    <span>{t('language')}</span>
                    {renderSettingsDropdown(
                      'language',
                      settingsDraft.language,
                      languageOptions,
                      (value) => updateSettingsDraft('language', value as AppSettings['language']),
                    )}
                  </div>
                  <div className="settings-kv">
                    <span>{t('splitRatio')}</span>
                    <strong>{Math.round(readmeRatio)}%</strong>
                  </div>
                </section>

              </div>
            </form>
          </section>
        ) : activeView === 'history' ? (
          <section className="download-view">
            <div className="local-toolbar glass-strip">
              <div className="local-search">
                <Search size={16} />
                <input
                  value={localModelsQuery}
                  onChange={(event) => setLocalModelsQuery(event.target.value)}
                  placeholder={t('searchLocal')}
                />
              </div>
              <div className="local-sort-select">
                {renderSettingsDropdown(
                  'localSort',
                  localSortKey,
                  [
                    { label: t('sortName'), value: 'name' as const },
                    { label: t('sortTime'), value: 'time' as const },
                    { label: t('sortSize'), value: 'size' as const },
                  ],
                  setLocalSortKey,
                )}
              </div>
              <button className="sort-toggle" onClick={() => setLocalSortDirection((current) => current === 'asc' ? 'desc' : 'asc')}>
                {localSortDirection === 'asc' ? t('asc') : t('desc')}
              </button>
            </div>

            <section className="download-queue glass-panel">
              {visibleDownloadGroups.length > 0 ? (
                visibleDownloadGroups.map((group) => {
                  const isOpen = openDownloadGroups.has(group.id)
                  const groupSpeed = formatSpeed(group.speedBytesPerSecond)
                  const canStartGroup = group.jobs.some((job) => job.status === 'paused' || job.status === 'failed' || job.status === 'error')
                  const canPauseGroup = group.jobs.some((job) => job.status === 'downloading' || job.status === 'queued')
                  return (
                    <article className={`download-group ${group.status}`} key={group.id}>
                      <button className="download-group-head" onClick={() => toggleDownloadGroup(group.id)}>
                        <ChevronRight size={16} className={isOpen ? 'tree-chevron open' : 'tree-chevron'} />
                        <div className="download-group-main">
                          <div className="download-job-title">
                            <strong>{group.title}</strong>
                            <span>{group.repoId}</span>
                          </div>
                          <div className="download-path">{group.path}</div>
                          <div className="download-meta">
                            <span>{group.status === 'completed' ? formatBytes(group.size) : `${formatBytes(group.downloadedBytes)} / ${formatBytes(group.size)}${groupSpeed ? ` · ${groupSpeed}` : ''}`}</span>
                            <span>{group.completedCount}/{group.jobs.length} files{group.status !== 'completed' ? ` · ${getStatusLabel(group.status, t)}` : ''}</span>
                          </div>
                        </div>
                        {group.status !== 'completed' && (
                          <div className="download-group-side">
                            <strong>{group.progress}%</strong>
                            <span>{group.activeCount > 0 ? `${group.activeCount} ${t('active')}` : group.failedCount > 0 ? `${group.failedCount} failed` : `${group.completedCount} ${t('completed')}`}</span>
                          </div>
                        )}
                      </button>
                      <div className="download-group-actions">
                        {canPauseGroup ? (
                          <button className="icon-pill text" onClick={() => pauseDownloadGroup(group)} aria-label={`Pause ${group.title}`}>
                            <Pause size={15} />
                          </button>
                        ) : canStartGroup ? (
                          <button className="icon-pill text" onClick={() => startDownloadGroup(group)} aria-label={`Start ${group.title}`}>
                            <Play size={15} />
                          </button>
                        ) : null}
                        {(group.failedCount > 0 || group.activeCount > 0 || group.status === 'paused') && (
                          <label className="auto-retry-switch group-auto-retry" title={t('autoRetryFailed')}>
                            <input
                              type="checkbox"
                              checked={autoRetryGroupIds.has(group.id)}
                              onChange={() => toggleAutoRetryGroup(group.id)}
                            />
                            <span />
                            <em>{t('autoRetry')}</em>
                          </label>
                        )}
                        <button className="icon-pill text" onClick={() => openDownloadDirectory(group.path)}>
                          <FolderOpen size={15} />
                          {t('open')}
                        </button>
                      </div>
                      {isOpen && (
                        <div className="download-file-tree">
                          {group.jobs.map((job) => {
                            const progress = getDownloadProgress(job)
                            const canPause = job.status === 'downloading' || job.status === 'queued'
                            const canResume = job.status === 'paused' || job.status === 'error' || job.status === 'failed'
                            const jobSpeed = formatSpeed(downloadSpeeds[job.id] ?? 0)
                            const isCompleted = job.status === 'completed'
                            return (
                              <div
                                className={`download-job ${job.status}`}
                                key={job.id}
                                style={isCompleted ? { gridColumn: `span ${getCompletedCardSpan(job.fileName)}` } : undefined}
                              >
                                <div className="download-job-main">
                                  <div className="download-job-title">
                                    <strong>{isCompleted ? renderBreakableFileName(job.fileName) : job.fileName}</strong>
                                    <span className="download-title-actions">
                                      <button className="inline-action" onClick={() => copyDownloadUrl(job.id, job.sourceUrl)} aria-label={`Copy download link for ${job.fileName}`}>
                                        {downloadActionFeedback[`copy:${job.id}`] ? <Check size={14} /> : <Copy size={14} />}
                                      </button>
                                      <button className="inline-action" onClick={() => openDownloadDirectory(job.targetPath)} aria-label={`Open directory for ${job.fileName}`}>
                                        {downloadActionFeedback[`open:${job.targetPath}`] ? <Check size={14} /> : <FolderOpen size={14} />}
                                      </button>
                                      <button className="inline-action danger" onClick={() => requestRemoveDownloadJob(job.id)} aria-label={`Remove ${job.fileName}`}>
                                        <Trash2 size={14} />
                                      </button>
                                    </span>
                                    {!isCompleted && <span>{getStatusLabel(job.status, t)}</span>}
                                  </div>
                                  {!isCompleted && <div className="download-path">{job.filePath}</div>}
                                  {!isCompleted && (
                                    <div className="download-progress">
                                      <span style={{ width: `${progress}%` }} />
                                    </div>
                                  )}
                                  <div className="download-meta">
                                    <span>{isCompleted ? formatBytes(job.size) : `${formatBytes(job.downloadedBytes)} / ${formatBytes(job.size)}${jobSpeed ? ` · ${jobSpeed}` : ''}`}</span>
                                    {job.warning && (
                                      <span className="download-warning" title={job.warning}>
                                        <AlertTriangle size={14} />
                                        SHA256
                                      </span>
                                    )}
                                    {!isCompleted && <span>{progress}%</span>}
                                  </div>
                                </div>
                                <div className="download-job-actions">
                                  {canPause && (
                                    <button className="icon-pill" onClick={() => pauseJob(job.id)} aria-label={`Pause ${job.fileName}`}>
                                      <Pause size={16} />
                                    </button>
                                  )}
                                  {canResume && (
                                    <button className="icon-pill" onClick={() => resumeJob(job.id)} aria-label={`Resume ${job.fileName}`}>
                                      <Play size={16} />
                                    </button>
                                  )}
                                  {job.status !== 'completed' && job.status !== 'canceled' && (
                                    <button className="icon-pill" onClick={() => cancelJob(job.id)} aria-label={`Cancel ${job.fileName}`}>
                                      <X size={16} />
                                    </button>
                                  )}
                                  {!isCompleted && (
                                    <button className="icon-pill" onClick={() => requestRemoveDownloadJob(job.id)} aria-label={`Remove ${job.fileName}`}>
                                      <Trash2 size={16} />
                                    </button>
                                  )}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </article>
                  )
                })
              ) : (
                <div className="download-empty">
                  <Download size={20} />
                  <p>{t('noDownloads')}</p>
                </div>
              )}
            </section>
          </section>
        ) : activeView === 'favorites' ? (
          <section className="favorites-view">
            <div className="local-toolbar glass-strip">
              <div className="local-search">
                <Search size={16} />
                <input
                  value={favoritesQuery}
                  onChange={(event) => setFavoritesQuery(event.target.value)}
                  placeholder={t('searchFavorites')}
                />
              </div>
              <span className="favorites-count">{visibleFavorites.length} {t('savedCount')}</span>
            </div>

            {visibleFavorites.length > 0 ? (
              <div className="favorites-grid">
                {visibleFavorites.map((repo) => (
                  <motion.article
                    className="repo-card favorite-card"
                    key={repo.id}
                    onClick={() => {
                      setActiveView('explore')
                      selectRepo(repo.id)
                    }}
                    style={{ flexBasis: getRepoCardWidth(repo) }}
                    whileHover={{ y: -3, scale: 1.006 }}
                    transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                  >
                    <div>
                      <div className="repo-name">{repo.id}</div>
                      <p>{repo.title}</p>
                      <div className="tag-row">
                        {repo.tags.map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                    </div>
                    <div className="repo-meta">
                      <span>{repo.size}<HardDrive size={13} /></span>
                      <span>{repo.downloads}<Download size={13} /></span>
                      <span>{repo.likes}<Heart size={13} /></span>
                      <span>{repo.updated}<Clock3 size={13} /></span>
                    </div>
                    <button
                      className="favorite-remove"
                      onClick={(event) => {
                        event.stopPropagation()
                        toggleFavorite(repo.id)
                      }}
                      aria-label={`Remove ${repo.id} from favorites`}
                    >
                      <Star size={16} fill="currentColor" />
                    </button>
                  </motion.article>
                ))}
              </div>
            ) : (
              <div className="empty-panel glass-panel">
                <Star size={22} />
                <h2>{t('noFavorites')}</h2>
                <p>{t('noFavoritesDesc')}</p>
              </div>
            )}
          </section>
        ) : (
          <>
        <section className="search-results">
          <div className="section-title">
            <div className="section-title-main">
              <span>{repoCards.length > 0 ? t('searchResults') : t('startHere')}</span>
              <small>{isSearching ? t('searching') : repoCards.length > 0 ? `${repoCards.length} ${t('repos')}` : t('ready')}</small>
            </div>
            <div className="result-actions">
              {repoCards.length > 0 && <button>{t('sortByRelevance')}</button>}
              {repoCards.length > 0 && (
                <>
                  <button className="nav-arrow" onClick={() => scrollResults(-1)} aria-label="Previous results">
                    <ChevronLeft size={16} />
                  </button>
                  <button className="nav-arrow" onClick={() => scrollResults(1)} aria-label="Next results">
                    <ChevronRight size={16} />
                  </button>
                </>
              )}
            </div>
          </div>
          {repoCards.length > 0 ? (
            <div className="repo-list" ref={resultsRef} onWheel={handleResultsWheel}>
              {repoCards.map((repo) => (
                <motion.article
                  className={`repo-card ${repo.id === selectedRepoId ? 'selected' : ''}`}
                  key={repo.id}
                  onClick={() => selectRepo(repo.id)}
                  style={{ flexBasis: getRepoCardWidth(repo) }}
                  whileHover={{ y: -3, scale: 1.006 }}
                  transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                >
                  <div>
                    <div className="repo-name">{repo.id}</div>
                    <p>{repo.title}</p>
                    <div className="tag-row">
                      {repo.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  </div>
                  <div className="repo-meta">
                    <span>{repo.size}<HardDrive size={13} /></span>
                    <span>{repo.downloads}<Download size={13} /></span>
                    <span>{repo.likes}<Heart size={13} /></span>
                    <span>{repo.updated}<Clock3 size={13} /></span>
                  </div>
                </motion.article>
              ))}
            </div>
          ) : (
            <div className="welcome-grid">
              <section className="welcome-card">
                <div className="welcome-icon">
                  <Search size={18} />
                </div>
                <h2>{t('searchHFTitle')}</h2>
                <p>{t('searchHFDesc')}</p>
              </section>
              <section className="welcome-card">
                <div className="welcome-icon">
                  <Folder size={18} />
                </div>
                <h2>{t('pickFilesTitle')}</h2>
                <p>{t('pickFilesDesc')}</p>
              </section>
              <section className="welcome-card">
                <div className="welcome-icon">
                  <Download size={18} />
                </div>
                <h2>{t('downloadResumeTitle')}</h2>
                <p>{t('downloadResumeDesc')}</p>
              </section>
            </div>
          )}
          <div className="search-history-row">
            <span>{t('recent')}</span>
            {[...searchHistory, ...quickStartRepos].filter((item, index, list) => list.indexOf(item) === index).slice(0, 8).map((item) => (
              <button key={item} onClick={() => selectRepo(item)}>
                {item}
              </button>
            ))}
            {searchHistory.length > 0 && (
              <button className="history-clear" onClick={clearSearchHistory}>
                {t('clear')}
              </button>
            )}
          </div>
        </section>

        <div
          className="viewer-grid"
          ref={viewerRef}
          style={{ gridTemplateColumns: `minmax(0, ${readmeRatio}fr) 8px minmax(420px, ${100 - readmeRatio}fr)` }}
        >
          <section className="glass-panel readme-panel">
            <div className="panel-head">
              {selectedRepoId ? (
                <div className="repo-heading">
                  <a className="repo-title-link" href={`https://huggingface.co/${selectedRepoId}`} target="_blank" rel="noreferrer">
                    {selectedRepoId}
                  </a>
                  <button className="copy-button" onClick={copyRepoId} aria-label="Copy repo id">
                    <Copy size={16} />
                  </button>
                  {repoUpdated && <span className="repo-updated">{t('updated')} {repoUpdated}</span>}
                  <div className="repo-actions">
                    <div className="tag-cloud inline">
                      {localTags.map((tag) => (
                        <button key={tag}>
                          <Tags size={13} />
                          {tag}
                        </button>
                      ))}
                    </div>
                    <button
                      className={`icon-pill ${favorites.some((favorite) => favorite.id === selectedRepoId) ? 'favorite-active' : ''}`}
                      onClick={() => toggleFavorite(selectedRepoId)}
                      aria-label={`${favorites.some((favorite) => favorite.id === selectedRepoId) ? 'Remove' : 'Add'} ${selectedRepoId} favorite`}
                    >
                      <Star size={17} fill={favorites.some((favorite) => favorite.id === selectedRepoId) ? 'currentColor' : 'none'} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="empty-panel-head">
                  <div className="mini-label">{t('modelReadme')}</div>
                  <h2>{t('readmeEmptyTitle')}</h2>
                </div>
              )}
            </div>

            {selectedRepoId && (
              <div className="model-tag-row">
                {modelTags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
            )}

            <div className="readme-body">
              <div className="readme-section">
                {!selectedRepoId ? (
                  <div className="readme-placeholder">
                    <h2>{t('readmePlaceholderTitle')}</h2>
                    <p>{t('readmePlaceholderDesc')}</p>
                    <div className="placeholder-steps">
                      <span>{t('stepSearchRepo')}</span>
                      <span>{t('stepViewReadme')}</span>
                      <span>{t('stepPickDownload')}</span>
                    </div>
                  </div>
                ) : readme ? (
                  <ReactMarkdown
                    rehypePlugins={[rehypeRaw]}
                    remarkPlugins={[remarkGfm]}
                    components={{
                      img: ({ src, alt }) => <img src={resolveReadmeAsset(selectedRepoId, src, settings.hfEndpoint)} alt={alt ?? ''} loading="lazy" />,
                      source: ({ src, ...props }) => <source src={resolveReadmeAsset(selectedRepoId, src, settings.hfEndpoint)} {...props} />,
                      video: ({ children, ...props }) => (
                        <video controls muted playsInline {...props}>
                          {children}
                        </video>
                      ),
                    }}
                  >
                    {readme}
                  </ReactMarkdown>
                ) : (
                  <div className="readme-empty">
                    <p>
                      {isReadmeRestricted
                        ? t('readmeRestricted')
                        : error || t('noReadme')}
                    </p>
                    {isReadmeRestricted && (
                      <button className="pill ghost readme-token-button" onClick={openTokenDialog}>
                        <Shield size={15} />
                        {t('addToken')}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </section>

          <div
            className="splitter"
            onPointerDown={startResize}
            onMouseDown={startMouseResize}
            role="separator"
            aria-label="Resize README and download tree"
          />

          <section className="glass-panel file-viewer">
            <div className="panel-head">
              <div>
                <div className="mini-label">{t('downloadTree')}</div>
                <h2>{t('selectFiles')}</h2>
              </div>
              <div className="selection-summary">
                <span>{selectedFiles.length} {t('selected')}</span>
                <strong>{selectedSize}</strong>
              </div>
            </div>

            <div className="tree-toolbar">
              <div className="tree-filter-strip" onWheel={scrollTreeFilters}>
                {treeFilterOptions.map((option) => (
                  <button
                    className={(multiTreeFilters.has(option) ? multiTreeSelection.has(option as MultiTreeFilter) : treeFilter === option && multiTreeSelection.size === 0) ? 'active' : ''}
                    key={option}
                    onClick={() => switchTreeFilter(option)}
                  >
                    {getTreeFilterLabel(option, t)}
                  </button>
                ))}
              </div>
              <label className="subfolder-field">
                <span>{t('saveTo')}</span>
                <em>/models/*/</em>
                <input value={subfolderName} onChange={(event) => setSubfolderName(event.target.value)} />
                <em>/</em>
              </label>
            </div>

            <div className="file-table">
              {!selectedRepoId ? (
                <div className="file-placeholder">
                  <FolderOpen size={22} />
                  <strong>{t('fileTreeTitle')}</strong>
                  <p>{t('fileTreeHint')}</p>
                </div>
              ) : visibleTreeRows.length === 0 ? (
                <div className="file-placeholder">
                  <FolderOpen size={22} />
                  <strong>{t('noFilesLoaded')}</strong>
                  <p>{error || t('noFilesHint')}</p>
                </div>
              ) : visibleTreeRows.map((row) => {
                const Icon = row.type === 'folder' ? Folder : File
                const isOpen = openFolders.has(row.id)
                const childIds = childMap[row.id] ?? []
                const isChecked =
                  row.type === 'folder'
                    ? childIds.length > 0 && childIds.every((childId) => checkedRows.has(childId))
                    : checkedRows.has(row.id)
                const isPartial =
                  row.type === 'folder'
                    && !isChecked
                    && childIds.some((childId) => checkedRows.has(childId))
                return (
                  <div className={`file-row ${isChecked ? 'checked' : ''} ${isPartial ? 'partial' : ''}`} key={row.id}>
                    <div className="tree-indent" style={{ width: row.depth * 22 }} />
                    {row.type === 'folder' ? (
                      <button className="tree-toggle" onClick={handleToggleFolder(row.id)} aria-label={`${isOpen ? 'Collapse' : 'Expand'} ${row.name}`}>
                        <ChevronRight size={15} className={isOpen ? 'tree-chevron open' : 'tree-chevron'} />
                      </button>
                    ) : (
                      <span className="tree-spacer" />
                    )}
                    <button className="check-box" onClick={handleToggleCheck(row.id)} aria-label={`${isChecked ? 'Unselect' : 'Select'} ${row.name}`}>
                      {isChecked && <Check size={13} />}
                      {isPartial && <Minus size={13} />}
                    </button>
                    <Icon size={17} className={row.type === 'folder' ? 'folder-icon' : 'file-icon'} />
                    <div className="file-name">{row.name}</div>
                    <div className="file-size">{row.type === 'folder' ? '' : formatBytes(row.size)}</div>
                  </div>
                )
              })}
            </div>
          </section>
        </div>
          </>
        )}

        <footer className="statusbar glass-strip">
          <span>
            <Database size={15} /> {downloadJobs.length} {t('persistedJobs')}
          </span>
          <span>
            <KeyRound size={15} /> {completedJobs.length} {t('completed')}
          </span>
          <span>
            <Terminal size={15} /> {backendStatus.ready ? t('tauriReady') : backendStatus.downloader}
          </span>
        </footer>
      </section>

      {isTokenDialogOpen && (
        <div className="modal-backdrop" onMouseDown={() => setIsTokenDialogOpen(false)}>
          <motion.form
            className="token-dialog glass-panel"
            onMouseDown={(event) => event.stopPropagation()}
            onSubmit={saveToken}
            initial={{ opacity: 0, scale: 0.96, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 420, damping: 34 }}
          >
            <div className="dialog-head">
              <div>
                <h2>{t('hfTokenTitle')}</h2>
                <p>{t('hfTokenDesc')}</p>
              </div>
              <button type="button" className="icon-pill" onClick={() => setIsTokenDialogOpen(false)}>
                <X size={17} />
              </button>
            </div>
            <label className="token-field">
              {t('accessToken')}
              <input
                autoFocus
                type="password"
                value={draftToken}
                onChange={(event) => setDraftToken(event.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
              />
            </label>
            <div className="dialog-actions">
              {accessToken && (
                <button type="button" className="pill ghost" onClick={clearToken}>
                  {t('clear')}
                </button>
              )}
              <button type="button" className="pill ghost" onClick={() => setIsTokenDialogOpen(false)}>
                {t('cancel')}
              </button>
              <button type="submit" className="pill primary">
                {t('saveToken')}
              </button>
            </div>
          </motion.form>
        </div>
      )}

      {pendingRemoveJobId && (
        <div className="modal-backdrop" onMouseDown={() => setPendingRemoveJobId('')}>
          <motion.div
            className="token-dialog glass-panel confirm-dialog"
            onMouseDown={(event) => event.stopPropagation()}
            initial={{ opacity: 0, scale: 0.96, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 420, damping: 34 }}
          >
            <div className="dialog-head">
              <div>
                <h2>{t('removeRecordTitle')}</h2>
                <p>{t('removeRecordDesc')}</p>
              </div>
              <button type="button" className="icon-pill" onClick={() => setPendingRemoveJobId('')}>
                <X size={17} />
              </button>
            </div>
            <div className="dialog-actions">
              <button type="button" className="pill ghost" onClick={() => setPendingRemoveJobId('')}>{t('cancel')}</button>
              <button type="button" className="pill danger" onClick={confirmRemoveDownloadJob}>{t('remove')}</button>
            </div>
          </motion.div>
        </div>
      )}

      {pendingCloseRequest && (
        <div className="modal-backdrop" onMouseDown={() => setPendingCloseRequest(false)}>
          <motion.div
            className="token-dialog glass-panel confirm-dialog"
            onMouseDown={(event) => event.stopPropagation()}
            initial={{ opacity: 0, scale: 0.96, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ type: 'spring', stiffness: 420, damping: 34 }}
          >
            <div className="dialog-head">
              <div>
                <h2>仍有下载任务进行中</h2>
                <p>关闭窗口会影响当前下载。请选择这次要怎么处理。</p>
              </div>
              <button type="button" className="icon-pill" onClick={() => setPendingCloseRequest(false)}>
                <X size={17} />
              </button>
            </div>
            <label className="remember-choice">
              <input
                type="checkbox"
                checked={rememberCloseChoice}
                onChange={(event) => setRememberCloseChoice(event.target.checked)}
              />
              记住选择，以后都按此执行
            </label>
            <div className="dialog-actions close-actions">
              <button type="button" className="pill danger" onClick={() => handleCloseChoice('exit')}>直接退出</button>
              <button type="button" className="pill primary" onClick={() => handleCloseChoice('minimize')}>最小化继续下载</button>
              <button type="button" className="pill ghost" onClick={() => handleCloseChoice('cancel')}>取消</button>
            </div>
          </motion.div>
        </div>
      )}

      {toastMessage && (
        <motion.div
          className="toast glass-strip"
          initial={{ opacity: 0, x: 18, y: -8 }}
          animate={{ opacity: 1, x: 0, y: 0 }}
          exit={{ opacity: 0, x: 18 }}
        >
          <Check size={15} />
          {toastMessage}
        </motion.div>
      )}
    </main>
  )
}

export default App
