import test from 'node:test'
import assert from 'node:assert/strict'
import { PiAcpAgent } from '../../src/acp/agent.js'
import { FakeAgentSideConnection, asAgentConn } from '../helpers/fakes.js'

class FakeSessions {
  closeCalls: string[] = []

  constructor(private readonly session: any) {}

  async create(_params: any) {
    return this.session
  }

  close(sessionId: string) {
    this.closeCalls.push(sessionId)
  }
}

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

function noModelsSession() {
  return {
    sessionId: 's1',
    cwd: process.cwd(),
    setStartupInfo() {},
    sendStartupInfoIfPending() {},
    proc: {
      async getAvailableModels() {
        return { models: [] }
      },
      async getState() {
        return { thinkingLevel: 'medium', model: null }
      }
    }
  }
}

test('PiAcpAgent: newSession throws AUTH_REQUIRED when pi reports zero models and no llama router', async () => {
  await withFetch(
    () => {
      throw new Error('fetch failed')
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      const sessions = new FakeSessions(noModelsSession())
      const agent = new PiAcpAgent(asAgentConn(conn), {} as any)
      ;(agent as any).sessions = sessions as any

      let threw = false
      try {
        await agent.newSession({ cwd: process.cwd(), mcpServers: [] } as any)
      } catch (e: any) {
        threw = true
        assert.equal(e?.code, -32000)
        assert.match(String(e?.message), /Configure an API key or log in with an OAuth provider/i)
      }

      assert.equal(threw, true)
      assert.deepEqual(sessions.closeCalls, ['s1'])
    }
  )
})

test('PiAcpAgent: newSession proceeds with zero models when a llama router is reachable', async () => {
  const catalog = {
    data: [
      { id: 'Fast', status: { value: 'unloaded' }, meta: { n_ctx: 268435456 } },
      { id: 'Slow', status: { value: 'unloaded' } }
    ]
  }

  await withFetch(
    () => catalog,
    async () => {
      const conn = new FakeAgentSideConnection()
      const sessions = new FakeSessions(noModelsSession())
      const agent = new PiAcpAgent(asAgentConn(conn), {} as any)
      ;(agent as any).sessions = sessions as any

      const res = await agent.newSession({ cwd: process.cwd(), mcpServers: [] } as any)

      assert.equal(sessions.closeCalls.length, 0)
      assert.ok(res.sessionId)

      // No previously selected model should leak into the model state.
      assert.equal((res as any).models, null)
      assert.ok(!((res as any).configOptions ?? []).some((o: any) => o?.id === 'model'))

      const startupInfo = String((res as any)._meta?.piAcp?.startupInfo ?? '')
      assert.match(startupInfo, /## llama\.cpp models/)
      assert.match(startupInfo, /Fast · unloaded/)
      assert.match(startupInfo, /Load a model with: \/llama <model>/)
    }
  )
})
