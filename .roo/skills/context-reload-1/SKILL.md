---
name: context-reload-1
description: Context will be reloaded when necessary
---

# Context Reload 1

## Instructions


This skill ensures that relevant project context is loaded at the start of each task or when manually refreshed, while avoiding unnecessary reads.

## When to Load Context

### Automatic (Task Start)
- **New task**: Always load context at the start of a new task
- **Heavy task**: Load context when task complexity increases (detected via tool usage patterns)
- **Context switch**: Load context when switching between different project areas

### Manual (Refresh)
- User can trigger refresh with: `/refresh_context`
- Use when: Task has become complex, or you need to re-establish project understanding

## Context Files to Load

1. **`context/CONTEXT.md`** - Project knowledge, research background, technical invariants
2. **`context/PROGRESS.md`** - Current status, recent completions, next steps
3. **`context/INSTRUCTIONS.md`** - Agent behavioral rules and constraints

## Loading Logic (Avoids Unnecessary Reads)

### Skip Loading If:
- ✅ Context was loaded within the last 5 interactions
- ✅ Task is simple (e.g., single file read, simple question)
- ✅ No significant time has passed since last load
- ✅ Task is a continuation of the previous task

### Load Context If:
- ❌ First interaction in a new session
- ❌ Task involves multiple files or complex operations
- ❌ More than 10 minutes have passed since last load
- ❌ User explicitly requests refresh with `/refresh_context`
- ❌ Task switches to a different project area

## Implementation

### Automatic Loading (Task Start)
At the start of each new task, the agent should:
1. Check if context was recently loaded
2. If not, load all three context files
3. Summarize key points for quick reference
4. Store timestamp of last load

### Manual Refresh Command
User can type: `/refresh_context`

This triggers:
1. Force reload of all context files
2. Update timestamp
3. Provide summary of current project state

## Usage Examples

### Example 1: New Task
```
User: "I need to fix the data loader"
Agent: [Loads context files automatically]
Agent: "I've loaded the project context. Based on context/PROGRESS.md, the data loader was recently updated for lazy tokenization. What specific issue are you experiencing?"
```

### Example 2: Manual Refresh
```
User: "/refresh_context"
Agent: [Reloads all context files]
Agent: "Context refreshed. Current status: Training stable on RTX 5070, Bunya OOM issue resolved with lazy tokenization. Next: Supervisor meeting preparation."
```

### Example 3: Skip Unnecessary Load
```
User: "What's in src/data.py?"
Agent: [Skips context load - simple file read task]
Agent: [Reads file and answers directly]
```

## Configuration

- **Cache duration**: 5 interactions or 10 minutes (whichever comes first)
- **Files to load**: `context/CONTEXT.md`, `context/PROGRESS.md`, `context/INSTRUCTIONS.md`
- **Manual trigger**: `/refresh_context`
- **Auto-trigger**: Task start, heavy task detection, context switch
