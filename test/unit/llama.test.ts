import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  LlamaClient,
  formatBytes,
  formatLlamaModelContext,
  formatLlamaModels,
  normalizeLlamaServerUrl,
  parseLlamaLoadProgress,
  resolveLlamaServer,
  type LlamaModelInfo
} from '../../src/acp/llama.js'

async function withFetch<T>(handler: (url: string, init?: RequestInit) => unknown, fn: () => Promise<T>): Promise<T> {
  const original = globalThis.fetch
  globalThis.fetch = (async (input: unknown, init?: RequestInit): Promise<Response> => {
    const url = String(typeof input === 'string' ? input : ((input as { url?: string })?.url ?? ''))
    const body = handler(url, init)
    return new Response(JSON.stringify(body ?? {}), { status: 200 })
  }) as typeof fetch
  try {
    return await fn()
  } finally {
    globalThis.fetch = original
  }
}

test('normalizeLlamaServerUrl: strips trailing slashes, /v1, query and hash', () => {
  assert.equal(normalizeLlamaServerUrl('http://10.18.0.1:8080'), 'http://10.18.0.1:8080')
  assert.equal(normalizeLlamaServerUrl('http://10.18.0.1:8080/'), 'http://10.18.0.1:8080')
  assert.equal(normalizeLlamaServerUrl('http://10.18.0.1:8080/v1'), 'http://10.18.0.1:8080')
  assert.equal(normalizeLlamaServerUrl('http://10.18.0.1:8080/v1/?a=1#x'), 'http://10.18.0.1:8080')
  assert.throws(() => normalizeLlamaServerUrl('ftp://10.18.0.1:8080'))
})

test('resolveLlamaServer: prefers pi stored credential, then env, then default', () => {
  const agentDir = mkdtempSync(join(tmpdir(), 'pi-acp-llama-'))
  const originalDir = process.env.PI_CODING_AGENT_DIR
  const originalUrl = process.env.LLAMA_BASE_URL
  const originalKey = process.env.LLAMA_API_KEY

  try {
    process.env.PI_CODING_AGENT_DIR = agentDir
    delete process.env.LLAMA_BASE_URL
    delete process.env.LLAMA_API_KEY

    // Default when nothing is configured.
    const def = resolveLlamaServer()
    assert.equal(def.serverUrl, 'http://127.0.0.1:8080')
    assert.equal(def.apiKey, undefined)

    // Env fallback.
    process.env.LLAMA_BASE_URL = 'http://env-host:9090/'
    process.env.LLAMA_API_KEY = 'env-key'
    assert.equal(resolveLlamaServer().serverUrl, 'http://env-host:9090')
    assert.equal(resolveLlamaServer().apiKey, 'env-key')

    // pi stored credential wins over env.
    writeFileSync(
      join(agentDir, 'auth.json'),
      JSON.stringify({
        'llama.cpp': {
          type: 'api_key',
          key: 'cred-key',
          env: { LLAMA_BASE_URL: 'http://cred-host:8080/v1' }
        }
      })
    )
    const server = resolveLlamaServer()
    assert.equal(server.serverUrl, 'http://cred-host:8080')
    assert.equal(server.apiKey, 'cred-key')
  } finally {
    if (originalDir === undefined) delete process.env.PI_CODING_AGENT_DIR
    else process.env.PI_CODING_AGENT_DIR = originalDir
    if (originalUrl === undefined) delete process.env.LLAMA_BASE_URL
    else process.env.LLAMA_BASE_URL = originalUrl
    if (originalKey === undefined) delete process.env.LLAMA_API_KEY
    else process.env.LLAMA_API_KEY = originalKey
  }
})

test('LlamaClient.list: parses catalog, rejects non-router payloads', async () => {
  const catalog: unknown = {
    data: [
      { id: 'a', status: { value: 'loaded' } },
      { id: 'b', status: { value: 'unloaded' } }
    ]
  }

  await withFetch(
    () => catalog,
    async () => {
      const client = new LlamaClient('http://127.0.0.1:8080')
      const models = await client.list()
      assert.equal(models.length, 2)
      assert.equal(models[0].id, 'a')
    }
  )

  await withFetch(
    () => ({ data: 'nope' }),
    async () => {
      const client = new LlamaClient('http://127.0.0.1:8080')
      await assert.rejects(client.list(), /invalid model catalog/)
    }
  )
})

test('LlamaClient.loadAndWait: polls to loaded and reports progress', async () => {
  let listCount = 0
  const ratios: number[] = []

  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      listCount += 1
      if (listCount === 1) {
        return {
          data: [{ id: 'm', status: { value: 'loading', progress: { file: { done: 512, total: 1024 } } } }]
        }
      }
      return { data: [{ id: 'm', status: { value: 'loaded' } }] }
    },
    async () => {
      const client = new LlamaClient('http://127.0.0.1:8080')
      const result = await client.loadAndWait(
        'm',
        p => {
          if (p.ratio !== null) ratios.push(p.ratio)
        },
        1,
        1000
      )
      assert.equal(result.status.value, 'loaded')
      assert.ok(ratios.includes(0.5), `expected 0.5 in ratios: ${JSON.stringify(ratios)}`)
    }
  )
})

test('LlamaClient.loadAndWait: reports failed load', async () => {
  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      return {
        data: [{ id: 'm', status: { value: 'unloaded', failed: true, exit_code: 2 } }]
      }
    },
    async () => {
      const client = new LlamaClient('http://127.0.0.1:8080')
      await assert.rejects(client.loadAndWait('m', undefined, 1, 50), /failed to load/)
    }
  )
})

test('LlamaClient.loadAndWait: unknown model fails after start window', async () => {
  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      return { data: [] }
    },
    async () => {
      const client = new LlamaClient('http://127.0.0.1:8080')
      await assert.rejects(client.loadAndWait('m', undefined, 1, 50), /not found on the llama\.cpp server/)
    }
  )
})

test('parseLlamaLoadProgress: sums per-file progress', () => {
  const entry: LlamaModelInfo = {
    id: 'm',
    status: { value: 'loading', progress: { a: { done: 1, total: 4 }, b: { done: 1, total: 4 } } }
  }
  const progress = parseLlamaLoadProgress(entry, 'm')
  assert.equal(progress.ratio, 0.25)
  assert.match(progress.detail ?? '', /\/ 8 B/)

  const empty = parseLlamaLoadProgress(undefined, 'm')
  assert.equal(empty.ratio, null)
  assert.equal(empty.detail, null)
})

test('formatLlamaModels: lists models with status, context and usage hint', () => {
  const models: LlamaModelInfo[] = [
    {
      id: 'Fast',
      status: { value: 'loaded' },
      meta: { n_ctx: 268435456 }
    },
    { id: 'Slow', status: { value: 'unloaded' }, meta: { n_ctx_train: 131072 } }
  ]

  const text = formatLlamaModels('http://10.18.0.1:8080', models, 'Fast')

  assert.match(text, /llama\.cpp models/)
  assert.match(text, /http:\/\/10\.18\.0\.1:8080/)
  assert.match(text, /→ Fast · loaded · 268435k context/)
  assert.match(text, / Slow · unloaded · 131k context/)
  assert.match(text, /Load a model with: \/llama <model>/)
})

test('formatLlamaModelContext: prefers n_ctx over n_ctx_train', () => {
  assert.equal(formatLlamaModelContext({ id: 'm', status: { value: 'unloaded' }, meta: { n_ctx: 8192 } }), '8k context')
  assert.equal(formatLlamaModelContext({ id: 'm', status: { value: 'unloaded' } }), null)
})

test('formatBytes: human readable sizes', () => {
  assert.equal(formatBytes(512), '512 B')
  assert.equal(formatBytes(1024), '1.00 KiB')
  assert.equal(formatBytes(3 * 1024 * 1024), '3.00 MiB')
})
