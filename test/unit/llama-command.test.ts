import test from 'node:test'
import assert from 'node:assert/strict'
import { PiAcpAgent } from '../../src/acp/agent.js'
import { FakeAgentSideConnection, FakePiRpcProcess, asAgentConn } from '../helpers/fakes.js'

class FakeSessions {
  constructor(private readonly session: any) {}
  maybeGet(_id: string) {
    return this.session
  }
  get(_id: string) {
    return this.session
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

function agentText(conn: FakeAgentSideConnection): string {
  return conn.updates
    .map(u => (u.update as { content?: { text?: string } }).content?.text ?? '')
    .filter(Boolean)
    .join('\n')
}

test('PiAcpAgent: /llama lists llama.cpp models adapter-side', async () => {
  const catalog = {
    data: [
      { id: 'Fast', status: { value: 'loaded' }, meta: { n_ctx: 131072 } },
      { id: 'Slow', status: { value: 'unloaded' } }
    ]
  }

  await withFetch(
    () => catalog,
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any
      proc.getState = async () => ({ model: { provider: 'llama.cpp', id: 'Fast' } })

      const agent = new PiAcpAgent(asAgentConn(conn))
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(proc.prompts.length, 0)

      const text = agentText(conn)
      assert.match(text, /llama\.cpp models/)
      assert.match(text, /→ Fast · loaded · 131k context/)
      assert.match(text, / Slow · unloaded/)
      assert.match(text, /Load a model with: \/llama <model>/)
    }
  )
})

test('PiAcpAgent: /llama <model> loads and sets the current model', async () => {
  let listCount = 0
  let setModel: string | null = null

  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      listCount += 1
      if (listCount === 1) {
        return { data: [{ id: 'Fast', status: { value: 'unloaded' } }] }
      }
      return { data: [{ id: 'Fast', status: { value: 'loaded' } }] }
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any
      proc.setModel = async (provider: string, modelId: string) => {
        setModel = `${provider}/${modelId}`
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama Fast' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(proc.prompts.length, 0)
      assert.equal(setModel, 'llama.cpp/Fast')

      const text = agentText(conn)
      assert.match(text, /Loading Fast/)
      assert.match(text, /Loaded Fast and set it as the current model\./)
    }
  )
})

test('PiAcpAgent: /llama <model> already-loaded model skips loading', async () => {
  let posts = 0
  let setModel: string | null = null

  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') {
        posts += 1
        return {}
      }
      return { data: [{ id: 'Fast', status: { value: 'loaded' } }] }
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any
      proc.setModel = async (provider: string, modelId: string) => {
        setModel = `${provider}/${modelId}`
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama Fast' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(posts, 0)
      assert.equal(setModel, 'llama.cpp/Fast')

      const text = agentText(conn)
      assert.match(text, /already loaded; set it as the current model\./)
    }
  )
})

test('PiAcpAgent: /llama <model> reports when pi cannot select the model yet', async () => {
  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      return { data: [{ id: 'Fast', status: { value: 'loaded' } }] }
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any
      proc.setModel = async () => {
        throw new Error('Model not found: llama.cpp/Fast')
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama Fast' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.match(agentText(conn), /Loaded Fast in llama\.cpp/)
    }
  )
})
