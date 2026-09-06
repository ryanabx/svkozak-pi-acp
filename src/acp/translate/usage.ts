import type { UsageUpdate } from '@agentclientprotocol/sdk'

/** The slice of pi's `get_session_stats` payload that describes context occupancy. */
type PiSessionStats = {
  cost?: unknown
  contextUsage?: { tokens?: unknown; contextWindow?: unknown } | null
}

function isTokenCount(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0
}

/**
 * Map pi's session stats onto ACP's `usage_update`.
 *
 * `contextUsage` is pi's own reading of the active branch, which is what ACP's
 * `used`/`size` describe; the session-wide `tokens.total` is a cumulative bill
 * and is deliberately not substituted for it. Pi reports `tokens: null` between
 * a compaction and the next assistant reply, so `compactedTokens` lets callers
 * supply the compaction's own estimate to cover that gap.
 */
export function toUsageUpdate(stats: unknown, compactedTokens?: number): UsageUpdate | null {
  const { contextUsage, cost } = (stats ?? {}) as PiSessionStats
  const size = contextUsage?.contextWindow
  const used = contextUsage?.tokens ?? compactedTokens
  if (!isTokenCount(used) || !isTokenCount(size) || size === 0) return null

  // Pi prices every model in USD per million tokens.
  const billed = typeof cost === 'number' && cost >= 0 ? { cost: { amount: cost, currency: 'USD' } } : {}
  return { used, size, ...billed }
}
