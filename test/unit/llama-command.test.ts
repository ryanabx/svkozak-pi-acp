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
      assert.match(text, /Use \/llama to select a model to load, or \/llama <model> to load one directly\./)
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

test('PiAcpAgent: /llama uses form elicitation to pick the model when supported', async () => {
  let listCount = 0
  let setModel: string | null = null

  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      listCount += 1
      // First two lists (selection + catalog check) still unloaded; load finishes after.
      if (listCount <= 2) {
        return { data: [{ id: 'Fast', status: { value: 'unloaded' } }] }
      }
      return { data: [{ id: 'Fast', status: { value: 'loaded' } }] }
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      conn.nextElicitationResponse = { action: 'accept', content: { model: 'Fast' } }
      const proc = new FakePiRpcProcess() as any
      proc.setModel = async (provider: string, modelId: string) => {
        setModel = `${provider}/${modelId}`
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: { elicitation: { form: {} } }
      } as any)
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(proc.prompts.length, 0)

      const [request] = conn.elicitationRequests as Array<Record<string, any>>
      assert.equal(request.mode, 'form')
      assert.equal(request.sessionId, 's1')
      assert.equal(request.requestedSchema.required[0], 'model')
      const options = request.requestedSchema.properties.model.oneOf as Array<Record<string, any>>
      assert.deepEqual(
        options.map(o => o.const),
        ['Fast']
      )

      assert.equal(setModel, 'llama.cpp/Fast')
      assert.match(agentText(conn), /Loaded Fast and set it as the current model\./)
    }
  )
})

test('PiAcpAgent: /llama honours a cancelled elicitation', async () => {
  let setModel = 0

  await withFetch(
    () => ({ data: [{ id: 'Fast', status: { value: 'unloaded' } }] }),
    async () => {
      const conn = new FakeAgentSideConnection()
      conn.nextElicitationResponse = { action: 'cancel' }
      const proc = new FakePiRpcProcess() as any
      proc.setModel = async () => {
        setModel += 1
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: { elicitation: { form: {} } }
      } as any)
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(setModel, 0)
      assert.match(agentText(conn), /Model selection cancelled\./)
    }
  )
})

test('PiAcpAgent: /llama falls back to a text listing without elicitation support', async () => {
  const catalog = { data: [{ id: 'Fast', status: { value: 'unloaded' } }] }

  await withFetch(
    () => catalog,
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any

      const agent = new PiAcpAgent(asAgentConn(conn))
      await agent.initialize({ protocolVersion: 1 } as any)
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(conn.elicitationRequests.length, 0)
      assert.match(agentText(conn), /llama\.cpp models/)
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

test('PiAcpAgent: /llama <model> reports load progress via tool call updates', async () => {
  let listCount = 0
  const snapshots = [
    { id: 'Fast', status: { value: 'unloaded' } },
    { id: 'Fast', status: { value: 'loading', progress: { a: { done: 1, total: 2 } } } },
    { id: 'Fast', status: { value: 'loading', progress: { a: { done: 1, total: 2 } } } },
    { id: 'Fast', status: { value: 'loading', progress: { a: { done: 2, total: 4 } } } },
    { id: 'Fast', status: { value: 'loaded' } }
  ]

  await withFetch(
    (url, init) => {
      if (init?.method === 'POST') return {}
      const snapshot = snapshots[Math.min(listCount, snapshots.length - 1)]
      listCount += 1
      return { data: [snapshot] }
    },
    async () => {
      const conn = new FakeAgentSideConnection()
      const proc = new FakePiRpcProcess() as any
      let setModel = 0
      proc.setModel = async () => {
        setModel += 1
      }

      const agent = new PiAcpAgent(asAgentConn(conn))
      ;(agent as any).sessions = new FakeSessions({ sessionId: 's1', proc, fileCommands: [] })

      const res = await agent.prompt({
        sessionId: 's1',
        prompt: [{ type: 'text', text: '/llama Fast' }]
      } as any)

      assert.equal(res.stopReason, 'end_turn')
      assert.equal(setModel, 1)

      const toolUpdates = conn.updates
        .map(u => (u as any).update)
        .filter(u => u?.sessionUpdate === 'tool_call' || u?.sessionUpdate === 'tool_call_update')

      assert.equal(toolUpdates[0].sessionUpdate, 'tool_call')
      assert.equal(toolUpdates[0].title, 'Loading Fast')
      assert.equal(toolUpdates[0].status, 'in_progress')

      // Repeated identical progress must not be re-emitted.
      const progressTexts = toolUpdates
        .slice(1)
        .map(u => u.content?.[0]?.content?.text)
        .filter(Boolean)

      assert.deepEqual(progressTexts, ['Loading Fast: 50% · 1 B / 2 B', 'Loading Fast: 50% · 2 B / 4 B'])
      assert.equal(toolUpdates.at(-1).status, 'completed')

      // The chat itself gets exactly one "Loading" message.
      const chatLoading = agentText(conn).match(/Loading Fast/g)
      assert.equal(chatLoading?.length, 1)
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
