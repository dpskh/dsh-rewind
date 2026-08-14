import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SessionStore, { type Session, type SessionEvent } from '@deepseek-ai/dsh-session'
import * as RewindInvariant from '../src/invariant.ts'
import InvariantService from '@deepseek-ai/dsh-invariants'

async function setup(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SessionStore)
  await ctx.plugin(InvariantService, { enabled: true })
  await ctx.plugin(RewindInvariant)
  return ctx
}

function event<T extends SessionEvent['type']>(
  type: T,
  data: SessionEvent<T>['data'],
  seq = 0,
): SessionEvent {
  return { type, seq, time: 0, data } as SessionEvent
}

describe('rewind fold invariants', () => {
  it('accepts a balanced fold bracket and a mark-referencing record', async () => {
    const ctx = await setup()
    // seq === array index (the append contract): the mark lives at index 1.
    const markEvents: (SessionEvent | undefined)[] = []
    markEvents[1] = event('checkpoint/mark', { turn: null }, 1)
    const session = { events: markEvents as SessionEvent[] } as unknown as Session
    expect(() => {
      ctx.emit('session/event', session, event('checkpoint/fold-start', { turn: null }, 2))
      ctx.emit('session/event', session, event('checkpoint/rewind', {
        turn: null,
        report: 'r',
        checkpointSeq: 1,
        shadowedRange: { start: 0, end: 0 },
        shadowedSeqs: [0],
        provider: 'p',
        model: 'm',
      }, 3))
      ctx.emit('session/event', session, event('checkpoint/fold-end', { turn: null }, 4))
    }).not.toThrow()
  })

  it('rejects a fold-end that closes no open fold', async () => {
    const ctx = await setup()
    expect(() => {
      ctx.emit('session/event', {} as Session, event('checkpoint/fold-end', { turn: null }))
    }).toThrow(/closes a fold that is not open/)
  })

  it('rejects a rewind record whose checkpointSeq is not a checkpoint/mark', async () => {
    const ctx = await setup()
    const session = { events: [] } as unknown as Session
    expect(() => {
      ctx.emit('session/event', session, event('checkpoint/rewind', {
        turn: null,
        report: 'r',
        checkpointSeq: 7,
        shadowedRange: { start: 0, end: 0 },
        shadowedSeqs: [0],
        provider: 'p',
        model: 'm',
      }))
    }).toThrow(/not a checkpoint\/mark/)
  })

  it('ignores unrelated session events', async () => {
    const ctx = await setup()
    expect(() => {
      ctx.emit('session/event', {} as Session, event('turn/start', { turn: 1 }))
      ctx.emit('session/event', {} as Session, event('checkpoint/mark', { turn: 1 }))
    }).not.toThrow()
  })
})
