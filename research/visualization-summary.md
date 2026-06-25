## IRC Conversation Disentanglement — Visualization System

### Overview

The project contains two visually identical web apps — `app/` and `app2/` — that display IRC chat conversations as a three-panel browser-based visualization. Both serve the same purpose (visually comparing thread structures), but display different data sources:

| App | Data Source | What it Shows |
|-----|------------|---------------|
| `app/` | `app/data/tiny.dev.json` — exported from gold annotation files via `scripts/export_chat_json.py` | Ground-truth conversation threads (human-labelled reply links) |
| `app2/` | `app2/data/predicted.json` — exported from model predictions via `src/evaluate_pred.py --export-json` | The model's predicted thread structure for the same conversations |

Both apps are structurally identical (HTML, CSS, JS) with the **only difference** being the `DATA_URL` on line 13 of `visualize.js`:
- `app/visualize.js` → `"data/tiny.dev.json"`
- `app2/visualize.js` → `"data/predicted.json"`

This enables side-by-side qualitative comparison: gold vs. predicted thread structure in a browser.

---

### Data Pipeline (JSON Format)

Both visualizers consume the same JSON schema, produced by two different exporters:

**Schema:**
```
{
  "name": "conversation_filename",
  "messages": [
    {
      "index": integer,         // message position in conversation
      "timestamp": "HH:MM",     // from IRC line [HH:MM]
      "speaker": "username",    // <username>
      "text": "message body",
      "is_system": boolean      // system join/quit messages
    },
    ...
  ],
  "links": [
    { "child": msg_index, "parent": msg_index },  // child → parent edge
    ...
  ],
  "threads": [
    { "id": root_index, "messages": [sorted indices], "size": count },
    ...
  ]
}
```

**Two export paths:**

1. **Gold data** (`scripts/export_chat_json.py`):
   - Input: Raw `.ascii.txt` files (IRC logs) + `.annotation.txt` files (gold parent-child labels)
   - Parses IRC lines via regex: `\[(\d{2}):(\d{2})\] <(\S+)> (.*)`
   - Reads annotations where column 0 = parent index, column 1 = child index
   - Computes thread clusters via Union-Find over the gold link set
   - Output: `app/data/<conv_name>.json`

2. **Predicted data** (`src/evaluate_pred.py`):
   - Input: A trained model checkpoint + the evaluation dataset
   - Runs the model in eval mode to get per-sample predicted parent indices (argmax over C candidates)
   - Uses the dataset's `conversation_map` to resolve predicted candidate indices back to actual message indices
   - Builds predicted links per conversation: `(child_idx, predicted_parent_idx)`
   - Computes thread clusters via Union-Find over predicted links
   - Output: `app/predicted_data/predicted_<split>_<conv_name>.json`

---

### Three-Panel Layout

The page is structured as a horizontal flex container (`#app-container`) split into:

| Panel | Width | Content |
|-------|-------|---------|
| **Left — Thread Legend** (`#sidebar`) | 20% | Ordered list of all threads with coloured dots and message counts |
| **Center — Chat Messages** (`#chat-container`) | 50% | Chronological message list with thread-coloured left borders |
| **Right — Link Graph** (`#graph-panel`) | 30% | SVG node-link diagram with Bezier curves |

---

### Chat Panel (`buildChat`)

- Messages are rendered as `<div class="message">` elements in chronological order
- Each message displays: index, timestamp `[HH:MM]`, speaker `<nick>`, and text content
- System messages (join/quit) are italicised, greyed out, and excluded from the thread graph
- The **left border** of each message is coloured according to its thread's assigned HSL colour
- The header shows: conversation name, message count, link count, and thread count
- `syncYPositions()` records the vertical pixel position of each non-system message, which is critical for aligning the graph panel

---

### Graph Panel (`buildGraph`)

The SVG graph is a **node-link diagram** with a single vertical axis (`NODE_X = 50px`) where:
- **Nodes**: Small circles (`r=5px`) positioned at the same Y-coordinate as their corresponding chat message, creating a vertical alignment between the two panels
- **Edges**: Quadratic Bezier curves (`M x y1 C x+cpOff y1, x+cpOff y2, x y2`) connecting child (top) → parent (bottom) nodes, with control points offset horizontally based on the vertical distance between messages
- **Labels**: Index numbers shown every 10 messages at the far left
- **Colours**: Both nodes and edges are coloured by thread (same HSL assignment as chat panel)

The graph panel is not a force-directed layout — it is a **static vertical alignment** that mirrors the chronological order of the chat panel. This design choice makes it trivial to cross-reference graph edges with chat messages by scanning horizontally.

---

### Thread Colouring

`hashToHue(threadId)` generates a deterministic hue using golden-angle hashing (`seed * 137.508 % 360`), ensuring:
- Different thread IDs get visually distinct colours
- The same thread ID always gets the same colour across sessions
- Colours are rendered as `hsl(H, 70%, 55%)` with consistent saturation/lightness

---

### Interactive Highlighting (`highlightThread`)

Hovering over **any message in the chat panel** or **any node in the graph panel** triggers `highlightThread(msgIndex)`:

1. **Union-Find on-the-fly**: `computeThreadLookup(n, links)` runs a fresh Union-Find every time to determine which thread root `msgIndex` belongs to. (This is O(n α(n)) per hover — acceptable for conversations with ~300 messages.)
2. **Chat dimming**: Messages not in the same thread get `opacity: 0.25` via the `message-dimming` CSS class. The hovered thread gets `opacity: 1.0` and a light grey background via `message-highlight`.
3. **Graph dimming**: Edges, nodes, and labels not in the same thread are reduced to `opacity: 0.08` (edges) or `0.1` (nodes/labels). The hovered thread's elements remain at full opacity.
4. **Scroll sync**: The graph panel scrolls to bring the hovered message's node into view using `scrollIntoView({ block: "center", behavior: "smooth" })`.

`resetAll()` restores all elements to full opacity on mouseleave.

---

### Union-Find (Core Algorithm)

The `computeThreadLookup(n, links)` function runs Union-Find with path compression to partition M message indices into thread clusters:

```
UF array: [0, 1, 2, ..., n-1]  (each message its own root initially)
For each link (parent, child):  union(parent, child)
Result:  lookup[i] = find(i)    (root index for each message)
```

Path compression (`uf[x] = uf[uf[x]]`) keeps the tree flat. This is applied:
1. At page load to colour all messages/nodes by thread
2. On every hover interaction (re-computed fresh each time — the data is small enough that caching isn't necessary)

---

### CSS Styling

- Monospace font stack, clean white/grey colour scheme
- Messages have `border-left: 4px solid <threadColor>` for thread identification
- Transitions on opacity (150ms) smooth the hover dimming effect
- Custom scrollbars in all three panels (5px width, grey)

---

### How This Serves the Thesis

The dual-app setup provides a **qualitative evaluation mechanism** alongside quantitative metrics (F1, ARI, VI). The researcher can:
1. View gold thread structure in `app/` (served on port 8080)
2. View predicted thread structure in `app2/` (served on port 8081)
3. Scan visually for: thread fragmentation (model splits one gold thread into multiple predicted threads), thread merging (model combines distinct gold threads), hallucinated links, or recency bias

The demo setup (documented in `context/PROGRESS.md`) uses a file with the highest link count (`predicted_test_2016-02-22_17.json`, 442 links) for maximum visual density.