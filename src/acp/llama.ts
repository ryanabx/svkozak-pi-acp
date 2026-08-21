import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { getAgentDir } from './pi-settings.js'

// Mirrors the llama.cpp router client of pi's built-in `llama` extension
// (packages/coding-agent/src/extensions/llama), which is TUI-only. RPC mode
// has no llama commands, so the adapter talks to the router directly.

export const LLAMA_PROVIDER_ID = 'llama.cpp'
export const DEFAULT_LLAMA_SERVER_URL = 'http://127.0.0.1:8080'

export type LlamaModelStatus = 'unloaded' | 'loading' | 'loaded' | 'downloading' | 'sleeping'

export interface LlamaModelInfo {
  id: string
  aliases?: string[]
  status: {
    value: LlamaModelStatus
    failed?: boolean
    exit_code?: number
    progress?: Record<string, { done: number; total: number }>
  }
  meta?: { n_ctx?: number; n_ctx_train?: number }
}

export interface LlamaServerConfig {
  serverUrl: string
  apiKey?: string
}

export interface LlamaLoadProgress {
  message: string
  ratio: number | null
  detail: string | null
}

export interface LlamaCatalog {
  serverUrl: string
  models: LlamaModelInfo[]
}

export function normalizeLlamaServerUrl(value: string): string {
  const url = new URL(value.trim())
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('llama.cpp server URL must use http or https')
  }
  url.hash = ''
  url.search = ''
  url.pathname = url.pathname.replace(/\/+$/u, '').replace(/\/v1$/u, '') || '/'
  return url.toString().replace(/\/$/u, '')
}

interface LlamaCredential {
  key?: string
  env?: Record<string, string>
}

function isLlamaCredential(value: unknown): value is LlamaCredential {
  if (!value || typeof value !== 'object') return false
  const cred = value as LlamaCredential
  return (
    (cred.key === undefined || typeof cred.key === 'string') &&
    (cred.env === undefined ||
      (typeof cred.env === 'object' &&
        !Array.isArray(cred.env) &&
        Object.values(cred.env as Record<string, unknown>).every(v => typeof v === 'string')))
  )
}

/**
 * Read the `llama.cpp` credential pi stores after `/login llama.cpp`
 * (auth.json in the agent dir). Returns undefined when not configured.
 */
export function readLlamaCredential(): LlamaCredential | undefined {
  const authPath = join(getAgentDir(), 'auth.json')
  if (!existsSync(authPath)) return undefined

  try {
    const raw: unknown = JSON.parse(readFileSync(authPath, 'utf-8'))
    if (!raw || typeof raw !== 'object') return undefined
    const cred = (raw as Record<string, unknown>)[LLAMA_PROVIDER_ID]
    return isLlamaCredential(cred) ? cred : undefined
  } catch {
    return undefined
  }
}

/**
 * Resolve the llama.cpp server to talk to:
 * 1. pi's stored credential (auth.json `llama.cpp` entry)
 * 2. LLAMA_BASE_URL / LLAMA_API_KEY environment variables
 * 3. pi's default local router
 */
export function resolveLlamaServer(): LlamaServerConfig {
  const credential = readLlamaCredential()
  const baseUrl = credential?.env?.LLAMA_BASE_URL || process.env.LLAMA_BASE_URL || DEFAULT_LLAMA_SERVER_URL
  const apiKey = credential?.key || process.env.LLAMA_API_KEY
  return { serverUrl: normalizeLlamaServerUrl(baseUrl), apiKey: apiKey || undefined }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KiB', 'MiB', 'GiB', 'TiB']
  let value = bytes / 1024
  let unit = units[0]!
  for (let i = 1; i < units.length && value >= 1024; i++) {
    value /= 1024
    unit = units[i]!
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${unit}`
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) return fallback
  const error = (payload as { error?: unknown }).error
  if (typeof error === 'string') return error
  if (error && typeof error === 'object') {
    const message = (error as { message?: unknown }).message
    if (typeof message === 'string' && message) return message
  }
  return fallback
}

function isModelInfo(value: unknown): value is LlamaModelInfo {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as { id?: unknown; status?: { value?: unknown } }
  return typeof candidate.id === 'string' && typeof candidate.status?.value === 'string'
}

export class LlamaClient {
  readonly serverUrl: string
  private readonly apiKey: string | undefined
  private readonly timeoutMs: number

  constructor(serverUrl: string, apiKey?: string, timeoutMs = 15_000) {
    this.serverUrl = normalizeLlamaServerUrl(serverUrl)
    this.apiKey = apiKey
    this.timeoutMs = timeoutMs
  }

  async list(): Promise<LlamaModelInfo[]> {
    const payload = await this.request('/models')
    const data = (payload as { data?: unknown } | null)?.data
    if (!Array.isArray(data)) throw new Error('llama.cpp returned an invalid model catalog')
    if (!data.every(isModelInfo)) throw new Error('Server is not running in llama.cpp router mode')
    return data
  }

  async load(model: string): Promise<void> {
    await this.request('/models/load', { method: 'POST', body: JSON.stringify({ model }) })
  }

  async unload(model: string): Promise<void> {
    await this.request('/models/unload', { method: 'POST', body: JSON.stringify({ model }) })
  }

  /**
   * Post /models/load and poll the catalog until the model reports "loaded".
   * llama.cpp loads can take a long time, so there is no overall timeout;
   * only the "load never started" window is bounded.
   */
  async loadAndWait(
    model: string,
    onProgress?: (progress: LlamaLoadProgress) => void,
    pollMs = 250,
    startTimeoutMs = 10_000
  ): Promise<LlamaModelInfo> {
    await this.load(model)
    const startedAt = Date.now()
    let sawLoading = false

    while (true) {
      const models = await this.list()
      const entry = models.find(m => m.id === model)
      const status = entry?.status.value

      if (status === 'loaded') return entry!
      if (status === 'loading') sawLoading = true
      if (entry?.status.failed) {
        const code = typeof entry.status.exit_code === 'number' ? ` (exit code ${entry.status.exit_code})` : ''
        throw new Error(`Model ${model} failed to load${code}`)
      }
      if (!sawLoading && Date.now() - startedAt > startTimeoutMs) {
        throw new Error(
          entry ? `Model ${model} did not start loading` : `Model ${model} not found on the llama.cpp server`
        )
      }

      onProgress?.(parseLlamaLoadProgress(entry, model))
      await sleep(pollMs)
    }
  }

  private async request(path: string, init: RequestInit = {}): Promise<unknown> {
    const headers = new Headers(init.headers)
    if (init.body !== undefined) headers.set('Content-Type', 'application/json')
    if (this.apiKey) headers.set('Authorization', `Bearer ${this.apiKey}`)

    const timeout = AbortSignal.timeout(this.timeoutMs)
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout

    let response: Response
    try {
      response = await fetch(`${this.serverUrl}${path}`, { ...init, headers, signal })
    } catch {
      throw new Error(`Could not connect to the llama.cpp server at ${this.serverUrl}`)
    }

    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      payload = undefined
    }

    if (!response.ok) throw new Error(errorMessage(payload, `llama.cpp returned HTTP ${response.status}`))
    return payload
  }
}

export function parseLlamaLoadProgress(entry: LlamaModelInfo | undefined, modelId: string): LlamaLoadProgress {
  let done = 0
  let total = 0
  const progress = entry?.status?.progress
  if (progress) {
    for (const value of Object.values(progress)) {
      if (typeof value?.done === 'number') done += value.done
      if (typeof value?.total === 'number') total += value.total
    }
  }

  const ratio = total > 0 ? Math.min(1, done / total) : null
  const detail = total > 0 ? `${formatBytes(done)} / ${formatBytes(total)}` : null
  return { message: `Loading ${modelId}`, ratio, detail }
}

/**
 * Best-effort probe of the llama.cpp router used to decide whether ACP can be
 * used without provider auth. Returns null when the server is unreachable or
 * has no models to offer.
 */
export async function getLlamaCatalog(timeoutMs = 5_000): Promise<LlamaCatalog | null> {
  try {
    const server = resolveLlamaServer()
    const client = new LlamaClient(server.serverUrl, server.apiKey, timeoutMs)
    const models = await client.list()
    if (!models.length) return null
    return { serverUrl: server.serverUrl, models }
  } catch {
    return null
  }
}

export function llamaOptionTitle(model: LlamaModelInfo): string {
  const parts = [model.id, model.status.value]
  const ctx = formatLlamaModelContext(model)
  if (ctx) parts.push(ctx)
  return parts.join(' · ')
}

export function llamaStartupSection(catalog: LlamaCatalog): string[] {
  const lines: string[] = [`## llama.cpp models (${catalog.serverUrl})`]
  for (const model of catalog.models) lines.push(`- ${llamaOptionTitle(model)}`)
  lines.push('- Use /llama to select a model to load')
  lines.push('')
  return lines
}

export function formatLlamaModelContext(model: LlamaModelInfo): string | null {
  const ctx = model.meta?.n_ctx ?? model.meta?.n_ctx_train
  if (!ctx || ctx <= 0) return null
  // Same "k" convention as pi's llama UI.
  return ctx >= 1000 ? `${Math.round(ctx / 1000)}k context` : `${ctx} context`
}

export function formatLlamaModels(serverUrl: string, models: LlamaModelInfo[], currentModelId?: string): string {
  const current = currentModelId ?? models.find(m => m.status.value === 'loaded' || m.status.value === 'sleeping')?.id

  const lines: string[] = ['llama.cpp models', serverUrl, '']
  for (const model of models) {
    const marker = current && model.id === current ? '→' : ' '
    lines.push(`${marker} ${llamaOptionTitle(model)}`)
  }
  lines.push('', 'Use /llama to select a model to load, or /llama <model> to load one directly.')

  return lines.join('\n')
}
