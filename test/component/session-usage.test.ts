import test from 'node:test'
import assert from 'node:assert/strict'
import { PiAcpSession } from '../../src/acp/session.js'
import { FakeAgentSideConnection, FakePiRpcProcess, asAgentConn } from '../helpers/fakes.js'

function newSession(proc: FakePiRpcProcess, conn: FakeAgentSideConnection): PiAcpSession {
  return new PiAcpSession({
    sessionId: 's1',
    cwd: process.cwd(),
    mcpServers: [],
    proc: proc as any,
    conn: asAgentConn(conn),
    fileCommands: []
  })
}

function usageUpdates(conn: FakeAgentSideConnection): unknown[] {
  return conn.updates.filter(u => u.update.sessionUpdate === 'usage_update').map(u => u.update)
}

test('PiAcpSession: publishes usage when a turn settles, before resolving the prompt', async () => {
  const conn = new FakeAgentSideConnection()
  const proc = new FakePiRpcProcess()
  proc.sessionStats = { cost: 0.25, contextUsage: { tokens: 4184, contextWindow: 272_000 } }

  const session = newSession(proc, conn)
  const prompt = session.prompt('hello')
  proc.emit({ type: 'agent_start' })
  proc.emit({ type: 'agent_end' })
  proc.emit({ type: 'agent_settled' })

  assert.equal(await prompt, 'end_turn')
  assert.deepEqual(usageUpdates(conn), [
    { sessionUpdate: 'usage_update', used: 4184, size: 272_000, cost: { amount: 0.25, currency: 'USD' } }
  ])
})

test('PiAcpSession: publishes the compaction estimate while pi reports no tokens', async () => {
  const conn = new FakeAgentSideConnection()
  const proc = new FakePiRpcProcess()
  // Pi withholds occupancy from a compaction boundary until an assistant replies past it.
  proc.sessionStats = { contextUsage: { tokens: null, contextWindow: 272_000, percent: null } }

  newSession(proc, conn)
  proc.emit({
    type: 'compaction_end',
    reason: 'manual',
    result: { summary: 'summary', tokensBefore: 180_000, estimatedTokensAfter: 12_000 },
    aborted: false,
    willRetry: false
  })
  await new Promise(r => setTimeout(r, 0))

  assert.deepEqual(usageUpdates(conn), [{ sessionUpdate: 'usage_update', used: 12_000, size: 272_000 }])
})

test('PiAcpSession: skips readings that have not changed', async () => {
  const conn = new FakeAgentSideConnection()
  const proc = new FakePiRpcProcess()
  proc.sessionStats = { contextUsage: { tokens: 100, contextWindow: 1_000_000 } }

  const session = newSession(proc, conn)
  await session.publishUsage()
  await session.publishUsage()
  proc.sessionStats = { contextUsage: { tokens: 200, contextWindow: 1_000_000 } }
  await session.publishUsage()

  assert.deepEqual(usageUpdates(conn), [
    { sessionUpdate: 'usage_update', used: 100, size: 1_000_000 },
    { sessionUpdate: 'usage_update', used: 200, size: 1_000_000 }
  ])
})

test('PiAcpSession: a failing stats request neither emits usage nor blocks the turn', async () => {
  const conn = new FakeAgentSideConnection()
  const proc = new FakePiRpcProcess()

  const session = newSession(proc, conn)
  const prompt = session.prompt('hello')
  proc.emit({ type: 'agent_start' })
  proc.emit({ type: 'agent_settled' })

  assert.equal(await prompt, 'end_turn')
  assert.equal(proc.getSessionStatsCount, 1)
  assert.deepEqual(usageUpdates(conn), [])
})

test('PiAcpSession: a cancelled turn keeps its stop reason and still reports usage', async () => {
  const conn = new FakeAgentSideConnection()
  const proc = new FakePiRpcProcess()
  proc.sessionStats = { contextUsage: { tokens: 900, contextWindow: 200_000 } }

  const session = newSession(proc, conn)
  const prompt = session.prompt('hello')
  proc.emit({ type: 'agent_start' })
  await session.cancel()
  proc.emit({ type: 'agent_settled' })

  assert.equal(await prompt, 'cancelled')
  assert.deepEqual(usageUpdates(conn), [{ sessionUpdate: 'usage_update', used: 900, size: 200_000 }])
})
