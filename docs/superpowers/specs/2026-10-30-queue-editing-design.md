# Queue editing tools: design

Date: 2026-10-30. Status: approved 2026-10-30; implemented.

## Why

nil 0.3.0 lets you remove, reorder and clear upcoming tracks in the Now Playing queue, and gives
plugins the same through `player.remove`, `player.move` and `player.clear_upcoming`. The owner
wants an AI agent talking to the DJ over MCP to be able to do it too: "drop the next two", "play
this one next", "clear what's coming up".

## Depends on nil 0.3.0

| Method | Params | Result |
|---|---|---|
| `player.remove` | `{ "tracks": [ {"index": n, "path": p}, … ] }` | `{}` |
| `player.move` | `{ "tracks": [ … ], "to": n }`: in front of the track now at `to` | `{}` |
| `player.clear_upcoming` | none | `{ "removed": n }` |

Only tracks after the current one can be edited. A stale index or path, a track that has
started (including the next track in the last ~2 s of the current one, which the decoder has
already taken), or nothing playing is `-32004` with `data.reason` `"changed"`, `"started"` or
`"stopped"`. Older nil answers `-32601`. A remove or clear fires `player.changed` (the queue
length changes), so the radio tops up as usual; a move does not.

## MCP tools

- **`now_playing`** gains `upcoming`: the next 20 tracks as
  `{ "position": 1, "path", "title", "artist" }` (position 1 = the next track; title and artist
  from the index, `null` for a path it lacks). `queue_len` stays.
- **`remove_from_queue(positions: list[int], paths: list[str] | None = None)`**: removes those
  upcoming tracks.
- **`move_in_queue(positions: list[int], to: int, paths: list[str] | None = None)`**: the
  tracks, in the order given, start at position `to` (`to = 1`: play them next; past the last
  position: the end).
- **`clear_upcoming()`**: returns `{ "removed": n }`.

Positions are 1-based within the upcoming tracks. Each tool reads `player.state`, turns
positions into `{index: current + position, path}` and `to` into the index of the `to`-th
upcoming track not being moved (the queue length past the last), then calls nil. With upcoming
`[A, B, C, D]`, moving `[1]` to `3` gives `[B, C, A, D]`. Positions must be distinct and within the upcoming tracks, else
the tool fails with a message naming the bad ones before calling nil.

`paths` holds the `upcoming[].path` values the agent saw at `positions`, in the same order. If it
is given with a different length, or the queue no longer holds those paths at those positions
(compared as nil does), the edit fails with the changed message, before any call to nil.

The tools return `{}` / `{ "removed": n }` on success. Errors, as tool errors with plain text:

| From nil | Message |
|---|---|
| `-32004` `"changed"` or `"started"` | "The queue changed (or the next track is already starting). Call now_playing and try again." |
| `-32004` `"stopped"` | "Nothing is playing." |
| `-32601` | "Queue editing needs nil 0.3.0 or newer." |

No retry inside the tool: the agent decides what to do with the new queue.

## Panel

Unchanged. Now Playing in nil is where the owner edits the queue by hand.

## Testing

- FakeNil gets the three methods with nil's rules (upcoming only, index + path check, the
  reasons) and a "next track starting" switch.
- Tests: position → index/path mapping; `to` mapping and capping; bad positions refused before
  any call; each `-32004` reason and `-32601` give the messages above; `upcoming` in
  `now_playing` (fewer than 20, more than 20, a path missing from the index, stopped → `[]`).
- Windows check by the owner, recorded in `docs/validation-log.md`: ask the agent to remove,
  move and clear while playing, with radio on and off.
