import test from 'node:test'
import assert from 'node:assert/strict'
import { toUsageUpdate } from '../../src/acp/translate/usage.js'

test('toUsageUpdate: maps pi contextUsage onto used/size and reports cost in USD', () => {
  assert.deepEqual(
    toUsageUpdate({
      cost: 0.42,
      tokens: { total: 999_999 },
      contextUsage: { tokens: 4184, contextWindow: 272_000, percent: 1.54 }
    }),
    { used: 4184, size: 272_000, cost: { amount: 0.42, currency: 'USD' } }
  )
})

test('toUsageUpdate: omits cost when pi reports none', () => {
  assert.deepEqual(toUsageUpdate({ contextUsage: { tokens: 0, contextWindow: 200_000 } }), {
    used: 0,
    size: 200_000
  })
})

test('toUsageUpdate: reports nothing when pi has no context reading', () => {
  // Pre-compaction shapes pi can return, plus older builds without contextUsage.
  assert.equal(toUsageUpdate({ contextUsage: { tokens: null, contextWindow: 200_000, percent: null } }), null)
  assert.equal(toUsageUpdate({ cost: 1, tokens: { total: 5000 } }), null)
  assert.equal(toUsageUpdate(undefined), null)
})

test('toUsageUpdate: rejects readings that are not usable token counts', () => {
  assert.equal(toUsageUpdate({ contextUsage: { tokens: -1, contextWindow: 200_000 } }), null)
  assert.equal(toUsageUpdate({ contextUsage: { tokens: 1.5, contextWindow: 200_000 } }), null)
  assert.equal(toUsageUpdate({ contextUsage: { tokens: Number.NaN, contextWindow: 200_000 } }), null)
  assert.equal(toUsageUpdate({ contextUsage: { tokens: 10, contextWindow: 0 } }), null)
  assert.equal(toUsageUpdate({ contextUsage: { tokens: 10, contextWindow: '200000' } }), null)
})

test('toUsageUpdate: falls back to the compaction estimate while pi reports null tokens', () => {
  assert.deepEqual(toUsageUpdate({ contextUsage: { tokens: null, contextWindow: 200_000 } }, 12_000), {
    used: 12_000,
    size: 200_000
  })
})

test('toUsageUpdate: prefers pi over the compaction estimate once pi has a reading', () => {
  assert.deepEqual(toUsageUpdate({ contextUsage: { tokens: 13_500, contextWindow: 200_000 } }, 12_000), {
    used: 13_500,
    size: 200_000
  })
})

test('toUsageUpdate: ignores a bad cost without dropping the context reading', () => {
  assert.deepEqual(toUsageUpdate({ cost: -1, contextUsage: { tokens: 10, contextWindow: 200_000 } }), {
    used: 10,
    size: 200_000
  })
})
