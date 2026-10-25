# nil-dj

An AI DJ for the nil bit-perfect music player. Ask any MCP-capable agent for music in plain
language ("45 minutes of mellow instrumental for coding at night") and nil-dj finds matching tracks
in your own library by how they sound (CLAP embeddings, run locally with ONNX Runtime), orders them
so they flow, and plays them in nil.

## Install

nil-dj runs as a plugin inside nil (see `PLUGIN-API.md` for the protocol). You need
[uv](https://docs.astral.sh/uv/) and ffmpeg on `PATH`.

1. Build nil.
2. From this repo: `uv run nil-dj install --nil E:\Projects\nil`
3. In nil, open Settings → Plugins and switch nil-dj on. Its tab indexes your library; the sound
   model downloads on first use.

## Connect an agent

nil-dj serves MCP on `http://127.0.0.1:47263/mcp` while nil is running (`NIL_DJ_MCP_PORT` changes
the port).

Claude Code:

```
claude mcp add --transport http nil-dj http://127.0.0.1:47263/mcp
```

Claude Desktop and Cursor (stdio; add to the MCP servers config):

```json
{
  "mcpServers": {
    "nil-dj": {
      "command": "uv",
      "args": ["run", "--directory", "<path to nil-dj>", "nil-dj", "serve", "--stdio"]
    }
  }
}
```

Without nil-dj running in nil, `serve --stdio` is search-only: play, queue and playlist tools
answer "nil-dj isn't running inside nil".
