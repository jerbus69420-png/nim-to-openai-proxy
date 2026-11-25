
 // reasoning.js
// Backend-specific "thinking" control: which chat_template_kwargs/top-level
// fields each model needs to toggle reasoning, and how to extract reasoning
// text from responses that embed it inline vs. as a structured field.

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

// Everything returned by getReasoningPayload() is spread into the top-level
// JSON body sent to NIM. Do not wrap it in `extra_body` — that's an
// openai-SDK convention this proxy (raw axios) doesn't use.
//
// By default, reasoning is kept out of `content` and returned in a
// structured `reasoning`/`reasoning_content` field. Clients that want
// legacy inline <thinking> tags can opt in via `x-reasoning-format: inline`.

// Models that embed reasoning inline in `content` via delimiter tags instead
// of a structured field.
const CONTENT_DELIMITER_TAGS = {
  'minimaxai/minimax-m3': ['<mm:think>', '</mm:think>']
};

// Stateful parser for extracting reasoning blocks across streamed chunks.
class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }

    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) {
        reasoning += this.buffer;
      } else {
        content += this.buffer;
      }
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      this.parser = new DelimiterParser(tags[0], tags[1]);
    }
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      parser = new DelimiterParser(tags[0], tags[1]);
    }
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

// Valid reasoning_effort values per model, where NIM enforces an enum.
// Values outside the set are dropped with a warning rather than forwarded.
const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],

  'deepseek-ai/deepseek-v4-flash-0731': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-pro-0813': ['low', 'high', 'max'],

  'nvidia/nemotron-3-super-120b-a12b': ['none', 'low', 'high'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['none', 'medium', 'high'],
  'nvidia/nemotron-3.5-super-vl-120b-a12b': ['none', 'low', 'high'],

  'minimaxai/minimax-m3': ['adaptive'],
  'minimaxai/minimax-m3.1-flash': ['low', 'medium', 'high', 'xhigh', 'max'],

  'moonshotai/kimi-k3': ['low', 'high', 'max'],
  'moonshotai/kimi-k3.1-pro': ['low', 'high', 'max'],

  'meta/muse-glimmer-30b': ['none', 'minimal', 'low', 'medium', 'high', 'max'],

  'z-ai/glm-5.3-flash': ['low', 'high', 'max']
};

function validReasoningEffort(model, effort) {
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (!allowed) return effort;
  if (allowed.includes(effort)) return effort;
  if (effort) {
    console.warn(`[REASONING] Dropping invalid reasoning_effort "${effort}" for ${model} (allowed: ${allowed.join(', ')})`);
  }
  return undefined;
}

// Resolves the client "off"/"on" override into an effective boolean.
// Reasoning is now always enabled where the backend supports it.
// Client overrides and ENABLE_THINKING_MODE no longer disable it.
function resolveEffectiveThinking(enableThinking, clientReasoningEffort) {
  return true;
}

// Nemotron 3.5 Lightning has no boolean flag — only a top-level integer
// reasoning_budget (max reasoning tokens, -1 to 32768, default 16384). This
// tier mapping is this proxy's own approximation, not an NVIDIA-defined enum.
const NEMOTRON_LIGHTNING_BUDGET_MAP = { low: 2048, medium: 8192, high: 16384, max: -1 };

// Nemotron Nano Omni also uses reasoning_budget rather than reasoning_effort.
// 2048 is this proxy's LOW-budget approximation.
const NEMOTRON_NANO_OMNI_LOW_BUDGET = 2048;

// Returns model-specific reasoning request payloads, spread into the
// top-level request body. The proxy enforces LOW for supported models,
// except Nemotron 3 Ultra, which is explicitly set to HIGH.
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  enableThinking = resolveEffectiveThinking(enableThinking, clientReasoningEffort);

  const effort = validReasoningEffort(
    model,
    model === 'nvidia/nemotron-3-ultra-550b-a55b' ? 'high' : 'low'
  );

  switch (model) {

    // ── Nemotron models ──────────────────────────────────────────

    case 'nvidia/nemotron-3-super-120b-a12b': {
      return { reasoning_effort: effort };
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      const payload = {
        reasoning_effort: effort
      };

      // Required for correct reasoning + tool-call handling.
      if (hasTools) {
        payload.chat_template_kwargs = {
          force_nonempty_content: true
        };
      }

      return payload;
    }

    case 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning': {
      return {
        reasoning_budget: NEMOTRON_NANO_OMNI_LOW_BUDGET
      };
    }

    case 'nvidia/nemotron-3.5-super-vl-120b-a12b': {
      return { reasoning_effort: effort };
    }

    case 'nvidia/nemotron-3.5-lightning-30b-a3b': {
      return {
        reasoning_budget: NEMOTRON_LIGHTNING_BUDGET_MAP.low
      };
    }

    // ── DeepSeek ─────────────────────────────────────────────────

    case 'deepseek-ai/deepseek-v4-flash-0731':
    case 'deepseek-ai/deepseek-v4-pro-0813': {
      return {
        chat_template_kwargs: {
          thinking: true,
          reasoning_effort: effort
        }
      };
    }

    // DeepSeek V4.1 Flash's hosted NIM API reference does not currently
    // document a reasoning_effort request parameter.
    // Its upstream encoding supports LOW, but we avoid sending an
    // undocumented parameter to the hosted endpoint.

    // ── OpenAI GPT-OSS ───────────────────────────────────────────

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      return { reasoning_effort: effort };
    }

    // ── Google Gemma ─────────────────────────────────────────────

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      return {
        chat_template_kwargs: { enable_thinking: true },
        include_reasoning: SHOW_REASONING
      };
    }

    // ── Meta Muse Glimmer ────────────────────────────────────────

    case 'meta/muse-glimmer-30b': {
      return { reasoning_effort: effort };
    }

    // ── GLM ──────────────────────────────────────────────────────

    case 'z-ai/glm-5.3-flash': {
      return {
        reasoning_effort: effort,
        chat_template_kwargs: {
          clear_thinking: true
        }
      };
    }

    // ── MiniMax ──────────────────────────────────────────────────

    case 'minimaxai/minimax-m3': {
      const thinkingMode = clientReasoningEffort === 'adaptive'
        ? 'adaptive'
        : 'enabled';

      return {
        chat_template_kwargs: {
          thinking_mode: thinkingMode
        }
      };
    }

    case 'minimaxai/minimax-m3.1-flash': {
      return {
        chat_template_kwargs: {
          reasoning_effort: effort
        }
      };
    }

    // ── Kimi ─────────────────────────────────────────────────────

    case 'moonshotai/kimi-k3': {
      return { reasoning_effort: 'low' };
    }

    case 'moonshotai/kimi-k3.1-pro': {
      return { reasoning_effort: effort };
    }

    // ── Models without a documented LOW setting ──────────────────
    //
    // poolside/laguna-xs-2.1:
    // Supports reasoning, but the hosted NIM endpoint does not document
    // a configurable LOW reasoning level.
    //
    // google/diffusiongemma-26b-a4b-it:
    // Reasoning is supported, but no documented LOW control is exposed.
    // Its NVIDIA free endpoint is also marked deprecated.
    //
    // stepfun-ai/step-5-preview:
    // Reasoning is enabled on the checkpoint, but no LOW/HIGH intensity
    // parameter is exposed in its hosted API.
    //
    // mistralai/mistral-large-4-1t-a52b:
    // Only NONE and HIGH are documented; LOW is not supported.
    //
    // The remaining Llama and Mistral instruct models in server.js
    // do not expose a documented reasoning-effort control.
    //
    // These models intentionally fall through to the backend default.

    default:
      return {};
  }
}

module.exports = {
  SHOW_REASONING,
  getReasoningPayload,
  resolveEffectiveThinking,
  StreamNormalizer,
  normalizeNonStreamChoice
};
