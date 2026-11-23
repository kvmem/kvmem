<script lang="ts">
  import { onMount, tick } from "svelte";
  import MarkdownContent from "$lib/components/app/content/MarkdownContent/MarkdownContent.svelte";
  import { DatabaseService as DB } from "$lib/services/database.service";
  import { settingsStore } from "$lib/stores/settings/index.svelte";
  import { context, streamChat, requestRecord, cancelRequest } from "./api";
  import { activeBranch, normalizeHistory } from "./history";
  import { english } from "./labels";
  import { readAttachment, fileAccept, supportedAttachment, attachmentContent } from "./attachments";
  import { setMode } from "mode-watcher";
  import {
    Plus,
    ArrowUp,
    Square,
    PanelLeft,
    MoreHorizontal,
    ChevronDown,
    MessageSquare,
    Download,
    Copy,
    RotateCcw,
    Pencil,
    Paperclip,
    SlidersHorizontal,
    Brain,
    X,
    Trash2,
  } from "@lucide/svelte";
  type Row = any;
  let ctx = $state<Row>(null),
    conversations = $state<Row[]>([]),
    active = $state<Row>(null),
    messages = $state<Row[]>([]);
  let input = $state(""),
    images = $state<Row[]>([]),
    busy = $state(false),
    loading = $state(false),
    error = $state(""),
    notice = $state("");
  let generatingConversation = $state<Row>(null);
  const viewingGeneration = $derived(busy && active?.id === generatingConversation?.id);
  let generationMessages: Row[] = [];
  const generationErrors = new Map<string, string>();
  const draftAttachments = new Map<string, Row[]>();
  let dragDepth = $state(0);
  let editingId = $state<string | null>(null),
    editedInput = $state(""),
    editedImages = $state<Row[]>([]);
  let editField: HTMLTextAreaElement | undefined = $state();
  const lastUserIndex = $derived(messages.findLastIndex((m) => m.role === "user"));
  const lastUser = $derived(messages[lastUserIndex]);
  let settingsOpen = $state(false),
    historyOpen = $state(false),
    migrationOpen = $state(false),
    mobile = $state(false);
  let historyButton: HTMLButtonElement | null = $state(null);
  let theme = $state("dark"),
    density = $state("comfortable"),
    language = $state("zh-CN");
  let overrides = $state<Row>({}),
    requestId = $state(""),
    live = $state<Row>(null),
    now = $state(Date.now());
  let controller: AbortController | null = null,
    bottom: HTMLDivElement,
    scroll: HTMLDivElement,
    reply: Row = null;
  let saveTimer: ReturnType<typeof setInterval>,
    drawTimer: ReturnType<typeof setTimeout> | null = null,
    writes = Promise.resolve(),
    contextBusy = false,
    historyInitialized = false;
  const t = (s: string) => (language === "en" ? english[s] || s : s);
  function mtpStats(message: Row) {
    const spec = message.speculative;
    const generic = spec?.backend === "radiance";
    const totals = generic || spec?.backend === "draft-mtp";
    const offered = spec?.mtp_drafted_tokens ?? (totals ? spec.drafted_tokens : null);
    const accepted = spec?.mtp_accepted_tokens ?? (totals ? spec.accepted_tokens : null);
    return Number.isFinite(offered) && Number.isFinite(accepted) ? {offered,accepted,generic} : null;
  }
  const scope = () => ctx?.backend + ":" + ctx?.model;
  const draftKey = () => "kvmem-console-draft:" + (active?.id || "new");
  function get(key: string, fallback: any) {
    try {
      return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
    } catch {
      return fallback;
    }
  }
  function set(key: string, value: any) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      error = t("浏览器存储已满，请导出历史备份");
    }
  }
  const plain = (v: any) => JSON.parse(JSON.stringify(v));
  const modelLabel = (name: unknown) =>
    typeof name === "string"
      ? name
          .split(/[\\/]/)
          .pop()!
          .replace(/\.gguf$/i, "")
      : "";
  const backendLabel = (name: string) =>
    ({ llamacpp: "llama.cpp", ninfer: "NInfer", polystrata: "PolyStrata" })[
      name
    ] ||
    name ||
    "KVMem";
  const historyGroups = $derived.by(() => {
    const today = new Date(now);
    today.setHours(0, 0, 0, 0);
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    const week = new Date(today);
    week.setDate(today.getDate() - 7);
    const groups = ["今天", "昨天", "近七天", "更早"].map((label) => ({
      label,
      items: [] as Row[],
    }));
    for (const c of [...conversations].sort(
      (a, b) => (b.lastModified || 0) - (a.lastModified || 0),
    )) {
      const time = Number(c.lastModified) || 0;
      groups[
        time >= +today ? 0 : time >= +yesterday ? 1 : time >= +week ? 2 : 3
      ].items.push(c);
    }
    return groups.filter((group) => group.items.length);
  });
  function autosize(node: HTMLTextAreaElement, _value: string) {
    const main = node.closest<HTMLElement>(".chat-main")!;
    const composer = node.closest<HTMLElement>(".composer-wrap")!;
    const header = main.querySelector<HTMLElement>(".chat-header")!;
    let pending = 0;
    const resize = () => {
      pending = 0;
      if (!node.isConnected) return;
      node.style.height = "auto";
      // Keep the action row visible even when the viewport gets shorter or
      // notices/controls wrap. Long drafts scroll within the textarea.
      const available =
        main.clientHeight -
        header.offsetHeight -
        (composer.offsetHeight - node.offsetHeight) -
        48;
      const minimum = parseFloat(getComputedStyle(node).minHeight) || 56;
      const maximum = Math.max(minimum, Math.min(200, available));
      node.style.height = Math.min(node.scrollHeight, maximum) + "px";
    };
    const schedule = () => {
      if (!pending) pending = requestAnimationFrame(resize);
    };
    const observer = new ResizeObserver(schedule);
    for (const element of [main, header, composer]) observer.observe(element);
    resize();
    return {
      update: () => {
        void tick().then(schedule);
      },
      destroy: () => {
        observer.disconnect();
        cancelAnimationFrame(pending);
      },
    };
  }
  function disclosure(node: HTMLDetailsElement) {
    const close = (event: PointerEvent) => {
      if (!node.contains(event.target as Node)) node.open = false;
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape" && node.open) {
        node.open = false;
        node.querySelector("summary")?.focus();
        event.stopPropagation();
      }
    };
    document.addEventListener("pointerdown", close);
    node.addEventListener("keydown", key);
    return {
      destroy() {
        document.removeEventListener("pointerdown", close);
        node.removeEventListener("keydown", key);
      },
    };
  }
  const shown = (key: string) =>
    typeof value(key) === "number" ? Number(value(key).toFixed(6)) : value(key);
  function drawer(
    node: HTMLElement,
    options: {
      enabled: boolean;
      close: () => void;
      trigger?: HTMLElement | null;
    } = {
      enabled: true,
      close: () => {
        settingsOpen = false;
      },
    },
  ) {
    let before: HTMLElement | null = null,
      enabled = false;
    const restore = () => {
      const target = before;
      before = null;
      queueMicrotask(() => target?.isConnected && target.focus());
    };
    const controls = () =>
      Array.from(
        node.querySelectorAll<HTMLElement>(
          "button:not(:disabled),input:not(:disabled),select:not(:disabled),summary",
        ),
      ).filter((el) => el.getClientRects().length && !el.closest("[inert]"));
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        options.close();
        e.preventDefault();
        e.stopPropagation();
      }
      if (e.key === "Tab") {
        const a = controls();
        const first = a[0],
          last = a.at(-1);
        if (
          e.shiftKey &&
          (document.activeElement === first || document.activeElement === node)
        ) {
          last?.focus();
          e.preventDefault();
        } else if (!e.shiftKey && document.activeElement === last) {
          first?.focus();
          e.preventDefault();
        }
      }
    };
    const update = (next: typeof options) => {
      options = next;
      if (options.enabled === enabled) return;
      enabled = options.enabled;
      if (enabled) {
        before = options.trigger ?? (document.activeElement as HTMLElement);
        queueMicrotask(() => {
          if (enabled && node.isConnected) (controls()[0] ?? node).focus();
        });
        node.addEventListener("keydown", key);
      } else {
        node.removeEventListener("keydown", key);
        restore();
      }
    };
    update(options);
    return {
      update,
      destroy() {
        node.removeEventListener("keydown", key);
        if (enabled) restore();
      },
    };
  }
  const source = (key: string) =>
    t(Object.hasOwn(overrides, key) ? "本对话已修改" : "服务设置");
  function value(key: string): any {
    if (key === "enable_thinking" && ctx?.backend === "radiance" && overrides.reasoning_effort) {
      if (["minimal", "none"].includes(overrides.reasoning_effort)) return false;
      return overrides.enable_thinking ?? true;
    }
    if (Object.hasOwn(overrides, key)) return overrides[key];
    if (["temperature", "top_p", "top_k", "min_p"].includes(key)) {
      const effort =
          overrides.reasoning_effort ?? ctx?.effective?.reasoning_effort,
        on = overrides.enable_thinking ?? ctx?.effective?.enable_thinking;
      const preset =
        ctx?.sampling_by_effort?.[on === false ? "none" : effort] ??
        ctx?.sampling_by_mode?.[on ? "thinking" : "non_thinking"];
      if (preset && key in preset) return preset[key];
    }
    return ctx?.effective?.[key];
  }
  function appearance(data: Row) {
    if (data.theme) theme = data.theme;
    if (data.density) density = data.density;
    if (data.language) language = data.language;
    const dark =
      theme === "dark" ||
      (theme === "system" &&
        matchMedia("(prefers-color-scheme: dark)").matches);
    setMode(dark ? "dark" : "light");
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    document.documentElement.lang = language;
  }
  async function refreshContext() {
    if (contextBusy) return;
    contextBusy = true;
    try {
      const old = scope();
      ctx = await context();
      if (ctx.state === "ready" && scope() !== old) {
        overrides = active?.consoleSettings?.[scope()] ?? {};
        if (old !== "undefined:undefined")
          notice = t("模型已切换，已读取该模型的对话设置");
      }
      if (ctx.state === "ready") validateOverrides();
    } catch {
      ctx = { ...ctx, state: "unavailable" };
      if (!ctx?.model) notice = t("请从 KVMem 控制台打开聊天");
    } finally {
      contextBusy = false;
    }
  }
  async function saveSettings() {
    if (active) {
      active = {
        ...active,
        consoleSettings: {
          ...active.consoleSettings,
          [scope()]: { ...overrides },
        },
      };
      await DB.updateConversation(active.id, {
        consoleSettings: plain(active.consoleSettings),
      } as any);
    }
  }
  function validateOverrides() {
    const invalid = Object.keys(overrides).filter(
      (k) =>
        (k === "preserve_thinking" && !ctx.capabilities?.preserve) ||
        (k === "reasoning_effort" &&
          overrides[k] !== "none" &&
          !ctx.capabilities?.efforts?.includes(overrides[k])) ||
        (k === "max_tokens" && overrides[k] > ctx.limit),
    );
    if (invalid.length) {
      const n = { ...overrides };
      invalid.forEach((k) => delete n[k]);
      overrides = n;
      notice =
        t("部分覆盖参数不再适用，已恢复服务设置") + ": " + invalid.join(", ");
      void saveSettings();
    }
  }
  function edit(key: string, raw: any) {
    error = "";
    let v = raw;
    if (
      !["enable_thinking", "preserve_thinking", "reasoning_effort"].includes(
        key,
      )
    ) {
      if (
        raw === "" ||
        (key === "reasoning_budget_tokens" &&
          Number(raw) === -1 &&
          ctx.backend === "llamacpp")
      ) {
        const n = { ...overrides };
        delete n[key];
        overrides = n;
        void saveSettings();
        return;
      }
      v = Number(raw);
      if (
        !Number.isFinite(v) ||
        (key !== "seed" && key !== "reasoning_budget_tokens" && v < 0) ||
        (["max_tokens", "reasoning_budget_tokens", "top_k", "seed"].includes(
          key,
        ) &&
          !Number.isInteger(v)) ||
        (key === "max_tokens" && (v < 1 || v > ctx.limit)) ||
        (key === "reasoning_budget_tokens" && v < (ctx.budget_min ?? 0)) ||
        (["top_p", "min_p"].includes(key) && v > 1) ||
        (key === "temperature" && v > 10)
      ) {
        error = t("参数超出有效范围，已保留原值") + ": " + key;
        return;
      }
    }
    overrides = { ...overrides, [key]: v };
    if (key === "enable_thinking") {
      const n = { ...overrides };
      if (ctx.backend === "polystrata")
        n.reasoning_effort = v
          ? ctx.effective.reasoning_effort === "none"
            ? "xhigh"
            : ctx.effective.reasoning_effort || "xhigh"
          : "none";
      else if (!v || (ctx.backend === "radiance" && ["none", "minimal"].includes(n.reasoning_effort))) delete n.reasoning_effort;
      overrides = n;
    }
    void saveSettings();
  }
  function editNumber(key: string, field: HTMLInputElement) {
    edit(key, field.value);
    field.value = String(shown(key) ?? "");
  }
  function reset() {
    overrides = {};
    void saveSettings();
    notice = t("已恢复当前服务设置");
  }
  function reportGenerationError(message: string, conversationId: string | undefined = generatingConversation?.id) {
    if (conversationId) generationErrors.set(conversationId, message);
    if (!conversationId || active?.id === conversationId) error = message;
  }
  async function refreshHistory(errorConversationId?: string) {
    try {
      conversations = await DB.getAllConversations();
    } catch (e) {
      if (errorConversationId) reportGenerationError(t("读取历史失败") + ": " + e, errorConversationId);
      else error = t("读取历史失败") + ": " + e;
    }
  }
  async function selectConversation(conv: Row) {
    if (loading) return;
    loading = true;
    const runningId = generatingConversation?.id;
    try {
      if (historyInitialized) {
        set(draftKey(), input);
        draftAttachments.set(draftKey(), plain(images));
      }
      historyInitialized = true;
      // Refresh currNode: sidebar rows may predate a newly created reply branch.
      const current = conv ? await DB.getConversation(conv.id) : null;
      const saved = current ? await DB.getConversationMessages(current.id) : [];
      messages = current?.id === runningId && current
        ? generationMessages.map((m) => ({ ...m }))
        : current ? activeBranch(saved, current.currNode).map((m: Row) => ({
            ...m,
            uiStatus: m.uiStatus === "pending" ? "interrupted" : m.uiStatus,
          })) : [];
      active = current;
      overrides = current?.consoleSettings?.[scope()] ?? {};
      input = get(draftKey(), "");
      images = plain(draftAttachments.get(draftKey()) || []);
      cancelEdit();
      error = generationErrors.get(current?.id) || "";
      historyOpen = false;
      set("kvmem-console-active", current?.id ?? null);
      await tick();
      bottom?.scrollIntoView();
    } catch (e) {
      error = t("读取历史失败") + ": " + e;
    } finally {
      loading = false;
    }
  }
  async function remove(conversation: Row) {
    if (
      !conversation ||
      busy ||
      loading ||
      !confirm(
        conversation.name + "\n\n" + t("删除此对话？请先导出需要保留的记录。"),
      )
    )
      return;
    await DB.deleteConversation(conversation.id);
    if (active?.id === conversation.id) await selectConversation(null);
    await refreshHistory();
  }
  function navigate(page: string) {
    parent.postMessage({ type: "kvmem-navigate", page }, location.origin);
  }
  async function addFiles(files: File[]) {
    if (busy || loading || !files.length) return;
    const editing = !!editingId;
    const existing = editing ? editedImages : images;
    if (existing.length + files.length > 4) {
      error = t("每条消息最多添加 4 个文件");
      return;
    }
    loading = true;
    error = "";
    try {
      const added = [];
      for (const file of files) {
        try { added.push(await readAttachment(file, !!ctx?.capabilities?.vision)); }
        catch (e) { throw Error(file.name + ": " + t(e instanceof Error ? e.message : String(e))); }
      }
      if (editing) editedImages = [...existing, ...added];
      else images = [...existing, ...added];
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }
  function fileDrag(e: DragEvent) {
    return Array.from(e.dataTransfer?.types || []).includes("Files");
  }
  function dragOver(e: DragEvent) {
    if (!fileDrag(e)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = busy || loading ? "none" : "copy";
  }
  function dropFiles(e: DragEvent) {
    dragDepth = 0;
    if (!fileDrag(e)) return;
    e.preventDefault();
    if (busy || loading) return;
    if (settingsOpen || migrationOpen) {
      error = t("请先关闭面板，再将文件拖入聊天");
      return;
    }
    const items = Array.from(e.dataTransfer?.items || []);
    if (items.some((item) => item.webkitGetAsEntry?.()?.isDirectory)) {
      error = t("请拖入文件，不支持文件夹");
      return;
    }
    const files = Array.from(e.dataTransfer?.files || []);
    if (!files.length) { error = t("无法读取拖入的文件，请使用附件按钮选择"); return; }
    void addFiles(files);
  }
  function wire(m: Row) {
    if (m.tool_calls || m.toolCalls || m.tools?.length || m.role === "tool")
      throw Error(t("此历史包含工具记录，请使用支持工具的客户端继续"));
    const extras = m.extra || [];
    if (
      m.legacyContent ||
      m.legacySource?.images?.some((x: Row) => !x.url) ||
      extras.some((x: Row) => !supportedAttachment(x))
    )
      throw Error(t("此历史包含当前聊天页尚不支持的附件，原记录已保留"));
    return {
      role: m.role,
      content: attachmentContent(m.content, extras),
      ...(m.reasoningContent ? { reasoning_content: m.reasoningContent } : {}),
    };
  }
  function options() {
    const body: Row = { model: ctx.model },
      template: Row = {};
    for (const [k, v] of Object.entries(overrides)) {
      if (["enable_thinking", "preserve_thinking"].includes(k)) template[k] = v;
      else body[k] = v;
    }
    if (Object.keys(template).length) body.chat_template_kwargs = template;
    return body;
  }
  function draw() {
    if (drawTimer) return;
    drawTimer = setTimeout(() => {
      drawTimer = null;
      if (active?.id !== generatingConversation?.id) return;
      const follow =
        scroll &&
        scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 140;
      messages = messages.map((m) => (m.id === reply?.id ? { ...reply } : m));
      if (follow) void tick().then(() => bottom?.scrollIntoView());
    }, 60);
  }
  function persistReply() {
    if (!reply) return;
    const { id, ...saved } = JSON.parse(JSON.stringify(reply));
    writes = writes
      .then(() => DB.updateMessage(id, saved))
      .catch((e) => {
        reportGenerationError(t("保存历史失败") + ": " + e);
      });
  }
  function cancelEdit() {
    editingId = null;
    editedInput = "";
    editedImages = [];
  }
  async function editLastMessage() {
    if (busy || loading || !lastUser) return;
    try {
      wire(lastUser);
      editingId = lastUser.id;
      editedInput = lastUser.content;
      editedImages = plain(lastUser.extra || []);
      await tick();
      editField?.focus();
      editField?.scrollIntoView({ block: "nearest" });
    } catch (e) {
      error = String(e);
    }
  }
  async function send(mode: "send" | "continue" | "regenerate" | "edit" = "send") {
    if (busy || loading) return;
    if (editingId && mode !== "edit") return;
    const regenerate = mode === "regenerate", editing = mode === "edit";
    if ((regenerate || editing) && !lastUser) return;
    if (editing && editingId !== lastUser.id) return;
    const content = mode === "continue" ? t("请从刚才停止的位置继续。") : editing ? editedInput : input;
    const attachments = plain(editing ? editedImages : images);
    if (!regenerate && !content.trim() && !attachments.length) return;
    const prefix = regenerate || editing ? messages.slice(0, lastUserIndex + (regenerate ? 1 : 0)) : messages;
    // Lock the preflight too: context refresh awaits network I/O.
    loading = true;
    try {
      await refreshContext();
      if (ctx?.state !== "ready") throw Error(t("请先启动模型"));
      prefix.forEach(wire);
      if (
        ((!regenerate && attachments.some((x: Row) => x.type === "image")) ||
          prefix.some((m) => m.extra?.some((x: Row) => x.type === "image"))) &&
        !ctx.capabilities?.vision
      ) throw Error(t("当前模型不支持图片"));
    } catch (e) {
      error = String(e);
      return;
    } finally {
      loading = false;
    }
    error = "";
    notice = "";
    busy = true;
    loading = true;
    let preparing = true;
    let conversationId = active?.id;
    reply = null;
    controller = new AbortController();
    requestId = crypto.randomUUID();
    live = { state: "waiting", started_at_ms: Date.now() };
    try {
      if (!active) {
        active = await DB.createConversation(
          input.trim().slice(0, 40) || t("新对话"),
        );
        set("kvmem-console-active", active.id);
        await saveSettings();
      }
      conversationId = active.id;
      generatingConversation = active;
      generationErrors.delete(conversationId);
      const user = regenerate ? lastUser : await DB.createMessageBranch(
        {
          convId: conversationId,
          type: "text",
          role: "user",
          content,
          timestamp: Date.now(),
          children: [],
          parent: null,
          extra: attachments,
        } as any,
        prefix.at(-1)?.id ?? null,
      );
      const nextMessages = regenerate ? prefix : [...prefix, user];
      if (!regenerate) {
        messages = nextMessages;
        if (editing) cancelEdit();
        else {
          input = "";
          images = [];
          set(draftKey(), "");
          set("kvmem-console-draft:new", "");
          draftAttachments.delete(draftKey());
          draftAttachments.delete("kvmem-console-draft:new");
        }
      }
      const history = nextMessages
          .filter((m) => m.content || m.reasoningContent || m.extra?.length)
          .map(wire),
        opts = options();
      reply = await DB.createMessageBranch(
        {
          convId: conversationId,
          type: "text",
          role: "assistant",
          content: "",
          reasoningContent: "",
          timestamp: Date.now() + 1,
          children: [],
          parent: user.id,
          uiStatus: "pending",
          requestId,
          execution: {
            backend: ctx.backend,
            model: ctx.model,
            displayModel: ctx.display_model,
            effective: Object.fromEntries(
              [
                ...new Set([
                  ...Object.keys(ctx.effective),
                  ...Object.keys(overrides),
                ]),
              ].map((k) => [k, value(k)]),
            ),
          },
        } as any,
        user.id,
      );
      generationMessages = [...nextMessages, reply];
      messages = [...generationMessages];
      await refreshHistory(conversationId);
      await tick();
      bottom?.scrollIntoView();
      preparing = false;
      loading = false;
      saveTimer = setInterval(persistReply, 750);
      await streamChat(
        { ...opts, messages: history },
        controller.signal,
        (data) => {
          const choice = data.choices?.[0],
            delta = choice?.delta;
          if (delta?.tool_calls)
            throw Error(t("此聊天页不执行工具，请使用支持工具的客户端"));
          reply.content += delta?.content ?? "";
          reply.reasoningContent += delta?.reasoning_content ?? "";
          if (delta?.reasoning_content || delta?.content) {
            live = {
              ...live,
              state: delta.reasoning_content ? "thinking" : "answering",
              first_token_at_ms: live.first_token_at_ms ?? Date.now(),
            };
            if (reply.thinkOpen === undefined)
              reply.autoOpen = !!delta.reasoning_content;
          }
          if (choice?.finish_reason) reply.finishReason = choice.finish_reason;
          if (data.timings) reply.timings = data.timings;
          if (data.usage) reply.usage = data.usage;
          if (data.kvmem?.speculative) reply.speculative = data.kvmem.speculative;
          draw();
        },
        requestId,
        ctx.instance,
      );
      reply.uiStatus = "complete";
    } catch (e) {
      if (reply)
        reply.uiStatus = controller?.signal.aborted ? "interrupted" : "error";
      if (!controller?.signal.aborted) reportGenerationError(String(e), conversationId);
    } finally {
      clearInterval(saveTimer);
      if (drawTimer) {
        clearTimeout(drawTimer);
        drawTimer = null;
      }
      if (reply) {
        const follow =
          scroll &&
          scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 140;
        reply.autoOpen = false;
        reply.elapsedMs = Date.now() - live.started_at_ms;
        reply.firstTokenMs = live.first_token_at_ms
          ? live.first_token_at_ms - live.started_at_ms
          : null;
        persistReply();
        await writes;
        if (active?.id === conversationId) {
          messages = messages.map((m) => (m.id === reply.id ? { ...reply } : m));
          if (follow) await tick().then(() => {
            if (active?.id === conversationId) bottom?.scrollIntoView();
          });
        }
      }
      try {
        await cancelRequest(requestId);
        if (conversationId)
          await DB.updateConversation(conversationId, { lastModified: Date.now() });
        await refreshHistory(conversationId);
      } catch (e) {
        reportGenerationError(t("保存历史失败") + ": " + e, conversationId);
      } finally {
        if (preparing) loading = false;
        busy = false;
        generatingConversation = null;
        controller = null;
        reply = null;
      }
    }
  }
  function stop() {
    controller?.abort();
    void cancelRequest(requestId);
  }
  function finish(m: Row) {
    if (m.uiStatus === "pending") return t("生成中");
    if (m.uiStatus === "interrupted") return t("已停止或中断");
    if (m.uiStatus === "error") return t("生成失败");
    if (m.finishReason === "length")
      return t(
        m.reasoningContent && !m.content
          ? "思考阶段达到输出上限"
          : "达到输出上限",
      );
    return t("正常结束");
  }
  function speed(m: Row) {
    const n = m.timings?.predicted_n ?? m.usage?.completion_tokens,
      r = m.timings?.predicted_per_second;
    return n > 1 && Number.isFinite(r) && r > 0 ? r.toFixed(1) : null;
  }
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      error = String(e);
    }
  }
  function download(v: any, name: string, type = "application/json") {
    const url = URL.createObjectURL(
      new Blob([typeof v === "string" ? v : JSON.stringify(v, null, 2)], {
        type,
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function exportAll() {
    const all = await DB.getAllConversations(),
      data = await DB.getConversationsWithMessages(all.map((c) => c.id));
    download(
      {
        schema_version: 1,
        source: location.origin,
        conversations: Array.from(data.values()),
      },
      "kvmem-chats.json",
    );
  }
  async function importFile(f: File | undefined) {
    if (!f) return;
    loading = true;
    error = "";
    try {
      if (f.size > 128 * 1024 * 1024) throw Error(t("历史文件过大"));
      const c = await normalizeHistory(JSON.parse(await f.text()));
      const existing = await DB.getConversationsWithMessages(
          (await DB.getAllConversations()).map((x) => x.id),
        ),
        ids = new Set(
          Array.from(existing.values()).flatMap((x) =>
            x.messages.map((m) => m.id),
          ),
        );
      if (
        c.items.some(
          (x: Row) =>
            !existing.has(x.conv.id) &&
            x.messages.some((m: Row) => ids.has(m.id)),
        )
      )
        throw Error("Message identity collision");
      const result = await DB.importConversations(c.items);
      notice = `${t("导入完成")} · ${result.imported.length} ${t("对话")} · ${result.skipped.length} ${t("重复已跳过")} · ${c.items.reduce((n: number, x: Row) => n + x.messages.length, 0)} ${t("消息")}`;
      if (c.warnings || c.legacy)
        notice +=
          " · " + t("工具或特殊附件原始数据已保留，当前页面不能继续此类对话");
      await refreshHistory();
    } catch (e) {
      error = t("导入失败，原记录未更改") + ": " + e;
    } finally {
      loading = false;
    }
  }
  onMount(() => {
    settingsStore.initialize();
    const viewport = matchMedia("(max-width: 760px)");
    const resize = () => {
      mobile = viewport.matches;
      if (!mobile) historyOpen = false;
    };
    resize();
    viewport.addEventListener("change", resize);
    const pref = localStorage.getItem("kvmem-monitor-theme") || "dark";
    appearance({
      theme: pref,
      density: localStorage.getItem("kvmem-monitor-density") || "comfortable",
      language: localStorage.getItem("kvmem-monitor-language") || "zh-CN",
    });
    const message = (e: MessageEvent) => {
      if (
        e.origin === location.origin &&
        e.source === parent &&
        e.data?.type === "kvmem-appearance"
      )
        appearance(e.data);
    };
    window.addEventListener("message", message);
    const systemTheme = matchMedia("(prefers-color-scheme: dark)");
    const syncSystemTheme = () => {
      if (theme === "system") appearance({});
    };
    systemTheme.addEventListener("change", syncSystemTheme);
    parent.postMessage({ type: "kvmem-chat-ready" }, location.origin);
    void refreshContext();
    void refreshHistory().then(() =>
      selectConversation(
        conversations.find((c) => c.id === get("kvmem-console-active", null)) ??
          null,
      ),
    );
    let mounted = true;
    let pollingRequest = "";
    const interval = setInterval(() => {
      now = Date.now();
      if (!busy) void refreshContext();
      else if (requestId && pollingRequest !== requestId) {
        const polledId = requestId;
        pollingRequest = polledId;
        void requestRecord(polledId).then((r) => {
          if (mounted && busy && requestId === polledId && r?.id === polledId)
            live = { ...live, ...r };
        }).catch(() => {
          // Metrics are optional; a failed poll must not interrupt the stream.
        }).finally(() => {
          if (pollingRequest === polledId) pollingRequest = "";
        });
      }
    }, 1000);
    const leave = () => {
      set(draftKey(), input);
      if (busy) {
        persistReply();
        stop();
      }
    };
    window.addEventListener("pagehide", leave);
    return () => {
      mounted = false;
      leave();
      clearInterval(interval);
      clearInterval(saveTimer);
      if (drawTimer) clearTimeout(drawTimer);
      window.removeEventListener("message", message);
      viewport.removeEventListener("change", resize);
      systemTheme.removeEventListener("change", syncSystemTheme);
      window.removeEventListener("pagehide", leave);
    };
  });
</script>

<svelte:window
  ondragenter={(e) => { if (fileDrag(e)) { e.preventDefault(); dragDepth += 1; } }}
  ondragover={dragOver}
  ondragleave={(e) => { if (fileDrag(e)) dragDepth = Math.max(0, dragDepth - 1); }}
  ondrop={dropFiles}
  ondragend={() => (dragDepth = 0)}
/>
<svelte:head
  ><title>KVMem · Chat</title><link
    rel="stylesheet"
    href="/monitor/monitor.css"
  /></svelte:head
>
<div id="app" data-theme={theme} data-density={density}>
  {#if historyOpen && mobile}<button
      class="history-backdrop"
      tabindex="-1"
      aria-label={t("关闭面板")}
      onclick={() => (historyOpen = false)}
    ></button>{/if}
  <div class="chat-shell">
    <aside
      class:mobile-open={historyOpen}
      aria-label={t("聊天历史")}
      tabindex="-1"
      role={historyOpen && mobile ? "dialog" : undefined}
      aria-modal={historyOpen && mobile ? true : undefined}
      use:drawer={{
        enabled: historyOpen && mobile,
        trigger: historyButton,
        close: () => {
          historyOpen = false;
        },
      }}
    >
      <div class="side-head">
        <strong>{t("聊天历史")}</strong><button
          class="mobile-only"
          onclick={() => (historyOpen = false)}
          aria-label={t("关闭面板")}><X size={18} /></button
        >
      </div>
      <button
        class="s-primary new-chat"
        disabled={loading}
        onclick={() => selectConversation(null)}
        ><Plus size={17} /> {t("新建对话")}</button
      >
      <nav aria-label={t("聊天历史")}>
        {#each historyGroups as group (group.label)}
          <div class="history-group">
            <span class="history-date">{t(group.label)}</span>
            {#each group.items as c (c.id)}<div
                class="history-row"
                class:selected={active?.id === c.id}
                class:generating={busy && generatingConversation?.id === c.id}
              >
                <button
                  class="history-item"
                  aria-current={active?.id === c.id ? "page" : undefined}
                  disabled={loading}
                  onclick={() => selectConversation(c)}
                  title={c.name}><span class="history-name">{c.name}</span>
                  {#if busy && generatingConversation?.id === c.id}<span class="generation-badge"><span class="pulse" aria-hidden="true"></span>{t("生成中")}</span>{/if}</button
                >
                <details class="history-menu" use:disclosure>
                  <summary
                    aria-label={t("对话操作") + "：" + c.name}
                    title={t("对话操作")}><MoreHorizontal size={17} /></summary
                  >
                  <div class="conversation-actions">
                    <button disabled={busy || loading} onclick={() => remove(c)}
                      ><Trash2 size={14} />{t("删除对话")}</button
                    >
                  </div>
                </details>
              </div>{/each}
          </div>
        {:else}<p class="history-empty">{t("对话会保存在这里")}</p>{/each}
      </nav>
      <div class="history-actions">
        <button
          disabled={busy || loading}
          onclick={() => {
            migrationOpen = !migrationOpen;
            historyOpen = false;
          }}><Download size={15} />{t("导入与备份")}</button
        ><small>{t("历史仅保存在此浏览器")}</small>
      </div>
    </aside>
    <section
      class="chat-main"
      class:empty={!messages.length && !migrationOpen}
      aria-label={t("聊天")}
      inert={historyOpen && mobile}
    >
      <header class="chat-header">
        <button
          bind:this={historyButton}
          class="mobile-only"
          onclick={() => (historyOpen = true)}
          aria-label={t("聊天历史")}><PanelLeft size={18} /></button
        >
        <div class="model-heading">
          <strong title={ctx?.display_model || ctx?.model}
            >{modelLabel(ctx?.display_model || ctx?.model) ||
              t("尚未启动模型")}</strong
          ><small
            >{backendLabel(ctx?.backend)}{#if ctx?.context}
              · {Number(ctx.context).toLocaleString()} {t("上下文")}{/if}</small
          >
        </div>
        <span
          class="chat-state"
          class:ready={ctx?.state === "ready"}
          class:working={busy}
          >{t(
            busy ? (viewingGeneration ? "生成中" : "后台生成中") : ctx?.state === "ready" ? "就绪" : "未就绪",
          )}</span
        ><button
          onclick={() => (settingsOpen = true)}
          disabled={busy || ctx?.state !== "ready"}
          ><SlidersHorizontal size={15} />{t("参数")}</button
        >
      </header>
      <div class="chat-scroll" bind:this={scroll}>
        {#if migrationOpen}<section class="migration s-card">
            <strong>{t("导入与备份")}</strong>
            <p>
              {t("迁移不会删除旧记录。先在旧地址导出 JSON，再在这里导入。")}
            </p>
            <div class="row">
              <button onclick={exportAll}>{t("导出全部历史")}</button><label
                class="button"
                >{t("导入历史")}<input
                  type="file"
                  accept=".json,application/json"
                  disabled={busy || loading}
                  onchange={(e) => {
                    void importFile(e.currentTarget.files?.[0]);
                    e.currentTarget.value = "";
                  }}
                /></label
              >{#if ctx?.history_export_url}<a
                  target="_blank"
                  rel="noopener"
                  href={ctx.history_export_url}>{t("打开旧地址导出页")} ↗</a
                >{/if}
            </div>
            <small
              >{t(
                "支持 KVMem、llama-ui 和 PolyStrata 导出；工具与特殊附件原始数据会保留。",
              )}</small
            >
          </section>{/if}
        {#if !messages.length}<div class="welcome">
            <div class="welcome-mark" aria-hidden="true">
              <MessageSquare size={26} strokeWidth={1.5} />
            </div>
            <h1>{t("开始一次对话")}</h1>
            <p>
              {t(
                ctx?.state === "ready"
                  ? "从一个问题、一个想法或一段代码开始。"
                  : "启动本地模型，即可开始聊天。",
              )}
            </p>
            {#if ctx?.state !== "ready"}<button
                onclick={() => navigate("config")}>{t("前往启动配置")}</button
              >{/if}
          </div>{/if}
        <div class="messages">
          {#each messages as m (m.id)}<article class:user={m.role === "user"} class:editing={editingId === m.id}>
              <div class="role">
                {m.role === "user"
                  ? t("你")
                  : modelLabel(
                      m.execution?.displayModel || m.execution?.model,
                    ) || "KVMem"}
              </div>
              {#if m.reasoningContent}<details
                  class="thinking"
                  open={m.thinkOpen ?? m.autoOpen ?? false}
                >
                  <summary
                    onclick={(e) => {
                      e.preventDefault();
                      m.thinkOpen = !(m.thinkOpen ?? m.autoOpen ?? false);
                      messages = [...messages];
                    }}
                    ><Brain size={15} /><span
                      >{t(
                        m.uiStatus === "pending" && live?.state === "thinking"
                          ? "思考中"
                          : m.uiStatus === "complete" || m.finishReason
                            ? "思考过程"
                            : "思考",
                      )}</span
                    ><ChevronDown size={14} class="thinking-chevron" /></summary
                  >{#if m.thinkOpen ?? m.autoOpen ?? false}<div
                      class="reasoning"
                    >
                      {m.reasoningContent}
                    </div>{/if}
                </details>{/if}
              {#if editingId === m.id}<form
                  class="message-editor"
                  onsubmit={(e) => { e.preventDefault(); void send("edit"); }}
                >
                  <label for="edit-last-message">{t("重新编辑")}</label>
                  <textarea
                    id="edit-last-message"
                    bind:this={editField}
                    bind:value={editedInput}
                    rows="4"
                    disabled={busy || loading}
                    onkeydown={(e) => {
                      if (e.key === "Escape" && !busy && !loading) cancelEdit();
                      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
                        e.preventDefault(); void send("edit");
                      }
                    }}
                  ></textarea>
                  {#if editedImages.length}<div class="attachments">
                      {#each editedImages as im, i}<button type="button"
                          disabled={busy || loading}
                          aria-label={t(im.type === "image" ? "移除图片" : "移除文件") + " " + im.name}
                          onclick={() => (editedImages = editedImages.filter((_, j) => i !== j))}
                          >{im.name || t("文件")} ×</button>{/each}
                    </div>{/if}
                  <div class="edit-actions">
                    <button type="button" onclick={cancelEdit} disabled={busy || loading}>{t("取消")}</button>
                    <button type="submit" class="s-primary"
                      disabled={busy || loading || ctx?.state !== "ready" || (!editedInput.trim() && !editedImages.length)}
                      >{t("保存并重新生成")}</button>
                  </div>
                </form>
              {:else if m.role === "assistant"}<MarkdownContent
                  content={m.content}
                />{:else}<div class="user-text">
                  {m.content}
                </div>{/if}
              {#each (editingId === m.id ? editedImages : m.extra) ?? [] as item}{#if item.type === "image" && item.base64Url}<img
                    class="attachment"
                    src={item.base64Url}
                    alt={item.name || t("图片")}
                  />{:else if item.type === "text"}<details class="text-attachment">
                    <summary>{item.name} · {t("文本文件")}</summary>
                    <pre>{item.content}</pre>
                  </details>{/if}{/each}
              {#if m.role === "assistant" && m.uiStatus !== "pending"}{@const mtp = mtpStats(m)}<footer
                  class="message-footer"
                >
                  <div class="message-meta">
                    <span
                      class:finish-warning={m.uiStatus === "error" ||
                        m.uiStatus === "interrupted" ||
                        m.finishReason === "length"}>{finish(m)}</span
                    >{#if speed(m)}<span>{t("平均")} {speed(m)} tok/s</span
                      >{/if}{#if m.timings?.predicted_n ?? m.usage?.completion_tokens}<span
                        >{m.timings?.predicted_n ?? m.usage?.completion_tokens} token</span
                      >{/if}
                    {#if mtp}<span>{t(mtp.generic ? "草稿接受 / 提议" : "MTP 接受 / 提议")} {mtp.accepted} / {mtp.offered}</span>{/if}
                  </div>
                  <div class="message-actions">
                    <button onclick={() => copy(m.content)} title={t("复制")}
                      ><Copy size={14} />{t("复制")}</button
                    ><button
                      onclick={() =>
                        download(m.content, "answer.md", "text/markdown")}
                      ><Download size={14} />{t("下载")}</button
                    >
                  </div>
                  <details>
                    <summary>{t("详情")}</summary>
                    <div class="message-details">
                      <div>{m.execution?.backend} · {m.execution?.model}</div>
                      {#if m.timings}<div>
                          {t("新算 / 复用 token")}：{m.timings.prompt_n ?? "—"} /
                          {m.timings.cache_n ?? "—"}
                        </div>
                        <div>
                          {t("预填充")}：{m.timings.prompt_ms != null
                            ? (m.timings.prompt_ms / 1000).toFixed(2) + "s"
                            : "—"}
                        </div>
                        {#if Number.isFinite(m.timings.predicted_ms)}<div>
                          {t("解码耗时")}：{(m.timings.predicted_ms / 1000).toFixed(2)}s
                        </div>{/if}
                        {#if Number.isFinite(m.timings.draft_n) && Number.isFinite(m.timings.draft_n_accepted)}<div>
                          {t("推测解码提议 / 接受")}：{m.timings.draft_n} / {m.timings.draft_n_accepted}
                        </div>{/if}
                      {/if}
                      <div>
                        {t("首个输出等待（含排队）")}：{m.firstTokenMs != null
                          ? (m.firstTokenMs / 1000).toFixed(2) + "s"
                          : "—"}
                      </div>
                      <div>
                        {t("总耗时")}：{m.elapsedMs != null
                          ? (m.elapsedMs / 1000).toFixed(2) + "s"
                          : "—"}
                      </div>
                      {#if m.execution?.effective}<div>
                          {t("本轮参数")}：{JSON.stringify(
                            m.execution.effective,
                          )}
                        </div>{/if}
                    </div>
                  </details>
                </footer>{/if}
              {#if m.role === "user" && m.id === lastUser?.id && !viewingGeneration}<footer class="user-message-actions">
                  <div class="message-actions">
                    <button onclick={() => send("regenerate")}
                      disabled={busy || loading || !!editingId || ctx?.state !== "ready"}
                      ><RotateCcw size={14} />{t("重新生成")}</button>
                    <button onclick={editLastMessage} disabled={busy || loading || !!editingId}
                      ><Pencil size={14} />{t("重新编辑")}</button>
                  </div>
                </footer>{/if}
            </article>{/each}
          <div bind:this={bottom}></div>
        </div>
      </div>
      <div class="composer-wrap" class:drag-active={dragDepth > 0 && !busy && !loading}>
        {#if dragDepth > 0 && !busy && !loading}<div class="drop-hint" role="status">{t(editingId ? "松开文件，添加到正在编辑的消息" : "松开文件以添加附件")}</div>{/if}
        {#if error && !settingsOpen}<div class="chat-error" role="alert">
            {error}
          </div>{/if}{#if notice}<div class="chat-notice" role="status">
            {notice}<button
              onclick={() => (notice = "")}
              aria-label={t("关闭面板")}><X size={16} /></button
            >
          </div>{/if}
        {#if busy && !viewingGeneration}<div class="background-generation" role="status">
            <span>{t("另一会话正在生成，可查看历史记录")}</span>
            <button onclick={() => selectConversation(generatingConversation)} disabled={loading}>{t("返回生成中的会话")}</button>
          </div>{/if}
        {#if viewingGeneration}<div class="live-line" role="status">
            <span class="pulse"></span>{t(
              live?.state === "thinking"
                ? "思考中"
                : live?.state === "answering"
                  ? "输出正文"
                  : live?.metrics?.status === "queued" ? "排队中"
                    : live?.metrics?.status === "reading" ? "预填充中"
                    : live?.metrics?.status === "generating" ? "生成中" : "等待首个输出",
            )} · {(
              Math.max(0, now - (live?.started_at_ms || now)) / 1000
            ).toFixed(0)}s{#if live?.metrics?.decode_tps}<span
                >{t("实时")} {live.metrics.decode_tps.toFixed(1)} tok/s</span
              >{/if}{#if live?.metrics?.decode_tps_mean}<span
                >{t("平均")}
                {live.metrics.decode_tps_mean.toFixed(1)} tok/s</span
              >{/if}<button onclick={() => navigate("monitor")}
              >{t("查看监控")}</button
            >
          </div>{/if}
        <form
          onsubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          {#if images.length}<div class="attachments">
              {#each images as im, i}<button
                  type="button"
                  disabled={busy || loading}
                  aria-label={t(im.type === "image" ? "移除图片" : "移除文件") + " " + im.name}
                  title={im.name}
                  onclick={() => (images = images.filter((_, j) => i !== j))}
                  >{im.name} ×</button
                >{/each}
            </div>{/if}<textarea
            aria-label={t("消息输入")}
            bind:value={input}
            use:autosize={input}
            oninput={() => set(draftKey(), input)}
            disabled={loading || busy}
            placeholder={t("输入消息…")}
            rows="1"
            onkeydown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
                e.preventDefault();
                void send();
              }
            }}
            onpaste={(e) => {
              const f = Array.from(e.clipboardData?.files || []);
              if (f.length) {
                e.preventDefault();
                void addFiles(f);
              }
            }}
          ></textarea>
          <div class="compose-actions">
            <div class="quick-controls">
              <label
                  class="button"
                  title={t("添加文件：图片、文本和代码；最多 4 个")}
                  ><Paperclip size={17} /><span class="sr-only"
                    >{t("添加文件")}</span
                  ><input
                    type="file"
                    accept={fileAccept}
                    multiple
                    disabled={busy || loading}
                    onchange={(e) => {
                      void addFiles(Array.from(e.currentTarget.files || []));
                      e.currentTarget.value = "";
                    }}
                  /></label
                ><button
                type="button"
                class="thinking-control"
                onclick={() => (settingsOpen = true)}
                disabled={busy || ctx?.state !== "ready"}
                ><Brain size={15} />{t("思考")}：{value("enable_thinking") ==
                null
                  ? t("未报告")
                  : value("enable_thinking")
                    ? t("开启")
                    : t(
                        "关闭",
                      )}{#if value("enable_thinking") && value("reasoning_effort")}<span
                    class="effort">· {value("reasoning_effort")}</span
                  >{/if}</button
              >
              {#if value("reasoning_budget_tokens") != null}<span
                  class="thinking-budget"
                  >{t("预算")}
                  {value("reasoning_budget_tokens") === -1
                    ? t("不限（-1）")
                    : value("reasoning_budget_tokens")}</span
                >{/if}
              {#if Object.keys(overrides).length}<button
                  type="button"
                  class="changed"
                  disabled={busy}
                  onclick={() => (settingsOpen = true)}
                  title={t("本对话已修改")}
                  aria-label={t("本对话已修改") +
                    " " +
                    Object.keys(overrides).length}
                  ><SlidersHorizontal size={13} />{Object.keys(overrides)
                    .length}</button
                >{/if}
            </div>
            <div class="send-actions">
              {#if viewingGeneration}<button
                  type="button"
                  class="stop"
                  onclick={stop}
                  aria-label={t("停止生成")}
                  title={t("停止生成")}
                  ><Square size={14} fill="currentColor" /></button
                >{:else}{#if !busy && !editingId && (messages.at(-1)?.finishReason === "length" || messages.at(-1)?.uiStatus === "interrupted")}<button
                    type="button"
                    disabled={loading || ctx?.state !== "ready"}
                    onclick={() => send("continue")}>{t("续写（新请求）")}</button
                  >{/if}<button
                  type="submit"
                  class="s-primary"
                  aria-label={t("发送")}
                  title={t("发送")}
                  disabled={busy || loading || !!editingId ||
                    ctx?.state !== "ready" ||
                    (!input.trim() && !images.length)}
                  ><ArrowUp size={19} /></button
                >{/if}
            </div>
          </div>
        </form>
        <div class="composer-hint">{t("Enter 发送 · Shift+Enter 换行")}</div>
      </div>
    </section>
  </div>
  {#if settingsOpen}<div
      class="drawer-backdrop"
      role="presentation"
      onclick={(e) => {
        if (e.target === e.currentTarget) settingsOpen = false;
      }}
    ></div>
    <div
      class="settings-drawer"
      use:drawer
      tabindex="-1"
      role="dialog"
      aria-label={t("生成参数")}
      aria-modal="true"
    >
      <header>
        <strong>{t("生成与思考")}</strong><button
          onclick={() => (settingsOpen = false)}
          aria-label={t("关闭面板")}><X size={16} /></button
        >
      </header>
      <div class="settings-body">
        {#if error}<div class="chat-error" role="alert">{error}</div>{/if}
        <p>{t("修改仅用于当前对话，下一次发送生效。")}</p>
        <label
          >{t("思考")}<small>{source("enable_thinking")}</small><select
            aria-label={t("思考")}
            value={value("enable_thinking") == null
              ? "unknown"
              : String(value("enable_thinking"))}
            onchange={(e) =>
              edit("enable_thinking", e.currentTarget.value === "true")}
            ><option value="unknown" disabled>{t("未报告")}</option><option
              value="true">{t("开启")}</option
            ><option value="false">{t("关闭")}</option></select
          ></label
        >
        {#if ctx?.capabilities?.efforts?.length}<label
            >{t("思考强度")}<small>{source("reasoning_effort")}</small><select
              aria-label={t("思考强度")}
              disabled={value("enable_thinking") === false}
              value={value("reasoning_effort") || ""}
              onchange={(e) => edit("reasoning_effort", e.currentTarget.value)}
              ><option value="" disabled>{t(ctx?.uses_template_effort ? "跟随模型" : "未报告")}</option
              >{#if value("reasoning_effort") === "none"}<option value="none"
                  >none</option
                >{/if}{#each ctx.capabilities.efforts as effort}<option
                  value={effort}>{effort}</option
                >{/each}</select
            ></label
          >{/if}
        {#each [{ key: "reasoning_budget_tokens", label: "思考预算 · tokens", min: ctx?.backend === "llamacpp" ? -1 : (ctx?.budget_min ?? 0) }, { key: "max_tokens", label: "总输出上限 · tokens", min: 1 }] as f}<label
            >{t(f.label)}<small>{source(f.key)}</small><input
              aria-label={t(f.label)}
              type="number"
              min={f.min}
              max={ctx?.limit}
              step="1"
              value={f.key === "max_tokens" && value(f.key) != null && value(f.key) <= 0 ? "" : (value(f.key) ?? "")}
              placeholder={t(f.key === "max_tokens" && value(f.key) != null && value(f.key) <= 0 ? "不限制（跟随服务）" : "未报告")}
              onchange={(e) => editNumber(f.key, e.currentTarget)}
            /></label
          >{/each}<small
          >{t(
            ctx?.backend === "ninfer"
              ? "思考预算必须为正整数；总输出包含思考和正文。"
              : ctx?.backend === "polystrata"
                ? "思考预算 0 表示不设思考预算；总输出仍限制思考和正文。"
                : "思考预算 0 会立即结束思考；-1 恢复服务设置。",
          )}</small
        >
        {#if ctx?.capabilities?.preserve}<label class="check"
            ><input
              type="checkbox"
              checked={value("preserve_thinking") === true}
              onchange={(e) =>
                edit("preserve_thinking", e.currentTarget.checked)}
            />{t("保留历史思考")}<small
              >{value("preserve_thinking") == null
                ? t("未报告")
                : source("preserve_thinking")}</small
            ></label
          ><small
            >{t(
              "保留历史思考可能增加上下文占用；不影响思考内容的展开或折叠。",
            )}</small
          >{/if}
        <details>
          <summary>{t("采样参数")}</summary
          >{#each ["temperature", "top_p", "top_k", "min_p", "seed"] as key}<label
              >{key}<small>{source(key)}</small><input
                aria-label={key}
                type="number"
                step={["seed", "top_k"].includes(key) ? 1 : 0.05}
                value={shown(key) ?? ""}
                placeholder={t("未报告")}
                onchange={(e) => editNumber(key, e.currentTarget)}
              /></label
            >{/each}
        </details>
      </div>
      <footer>
        <button onclick={reset}>{t("恢复服务设置")}</button><button
          class="s-primary"
          onclick={() => (settingsOpen = false)}>{t("完成")}</button
        >
      </footer>
    </div>{/if}
</div>

<style>
  :global(html),
  :global(body) {
    height: 100%;
    overflow: hidden;
    margin: 0;
  }
  #app {
    height: 100dvh;
    color: var(--s-ink);
    background: var(--s-bg);
    /* Upstream Markdown widgets inherit the console palette too. */
    --background: var(--s-bg);
    --foreground: var(--s-ink);
    --card: var(--s-surface);
    --card-foreground: var(--s-ink);
    --muted: var(--s-soft);
    --muted-foreground: var(--s-muted);
    --primary: var(--s-info-text);
    --accent: var(--s-soft);
    --accent-foreground: var(--s-ink);
    --border: var(--s-line);
    --ring: var(--s-info);
    --code-background: var(--s-surface);
    --code-foreground: var(--s-ink);
  }
  #app[data-theme] {
    background-image: none;
  }
  .chat-shell {
    --chat-width: 880px;
    display: grid;
    grid-template-columns: 232px minmax(0, 1fr);
    height: 100%;
    font-family: var(--s-font);
  }
  button,
  .button,
  select,
  input,
  textarea {
    font: inherit;
    color: var(--s-ink);
    border: 1px solid var(--s-line);
    background: var(--s-soft);
    border-radius: 8px;
  }
  button,
  .button {
    padding: 6px 12px;
    min-height: 34px;
    background: transparent;
    border-radius: var(--s-button-radius);
    font-weight: var(--s-button-weight);
    cursor: pointer;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
  }
  button:hover:not(:disabled),
  .button:hover:not(:has(input:disabled)) {
    background: var(--s-button-hover);
  }
  button:disabled,
  .button:has(input:disabled) {
    opacity: 0.45;
    cursor: default;
  }
  button:focus-visible,
  input:focus-visible,
  select:focus-visible,
  textarea:focus-visible,
  summary:focus-visible {
    outline: 2px solid var(--s-info);
    outline-offset: 2px;
  }
  small {
    color: var(--s-muted);
  }
  a {
    color: var(--s-info);
  }
  :global(#app svg.lucide) {
    flex-shrink: 0;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }
  aside {
    background: var(--s-surface);
    border-right: 1px solid var(--s-line-soft);
    padding: 20px 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    min-height: 0;
  }
  .side-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: var(--s-muted);
  }
  .side-head strong {
    font-weight: 500;
  }
  .new-chat {
    min-height: 38px;
  }
  aside nav {
    overflow: auto;
    flex: 1;
    min-height: 0;
    scrollbar-width: thin;
  }
  .history-group + .history-group {
    margin-top: 20px;
  }
  .history-date {
    display: block;
    padding: 0 10px 8px;
    color: var(--s-muted);
    font-size: 11px;
  }
  .history-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 30px;
    align-items: center;
    margin-bottom: 3px;
    border-radius: 8px;
  }
  .history-row:hover,
  .history-row:focus-within {
    background: var(--s-soft);
  }
  .history-row.selected {
    background: color-mix(in srgb, var(--s-info) 11%, var(--s-surface));
  }
  .history-row.generating {
    background: color-mix(in srgb, var(--s-accent) 16%, var(--s-surface));
    box-shadow: inset 3px 0 var(--s-accent);
  }
  .history-name { overflow: hidden; text-overflow: ellipsis; flex: 1; }
  .generation-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
    color: var(--s-info-text);
    font-size: 10px;
  }
  #app .history-item {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    min-height: 38px;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    background: transparent;
    border: 0;
    padding: 9px 10px;
    font-weight: 400;
  }
  #app .selected .history-item {
    color: var(--s-info-text);
    font-weight: 500;
  }
  .history-menu {
    display: contents;
  }
  .history-menu summary {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    min-height: 32px;
    cursor: pointer;
    border-radius: 5px;
    color: var(--s-muted);
  }
  summary {
    cursor: pointer;
  }
  .history-menu summary,
  .thinking summary,
  .message-footer summary {
    list-style: none;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  .conversation-actions {
    grid-column: 1 / -1;
    padding: 4px 6px 7px;
    border-top: 1px solid var(--s-line-soft);
  }
  #app .conversation-actions button {
    width: 100%;
    justify-content: flex-start;
    color: var(--s-error-text);
    border: 0;
    font-size: 12px;
  }
  .history-empty {
    font-size: 12px;
    line-height: 1.6;
    color: var(--s-muted);
    padding: 0 10px;
  }
  .history-actions {
    display: flex;
    flex-direction: column;
    gap: 9px;
    padding-top: 12px;
    border-top: 1px solid var(--s-line-soft);
  }
  #app .history-actions button {
    justify-content: flex-start;
    border: 0;
    color: var(--s-muted);
  }
  .history-actions small {
    font-size: 11px;
    padding: 0 12px;
  }
  .chat-main {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    background: var(--s-bg);
  }
  .chat-header {
    display: flex;
    gap: 14px;
    align-items: center;
    min-height: 60px;
    border-bottom: 1px solid var(--s-line-soft);
    padding: 10px 24px;
  }
  .model-heading {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 12px;
  }
  .model-heading strong {
    font-size: 14px;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .model-heading small {
    font-size: 11px;
    white-space: nowrap;
  }
  .chat-state {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--s-muted);
    font-size: 11px;
    white-space: nowrap;
  }
  .chat-state::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--s-muted);
  }
  .chat-state.ready::before {
    background: var(--s-accent);
  }
  .chat-state.working {
    color: var(--s-info-text);
  }
  .chat-scroll {
    overflow: auto;
    flex: 1;
    min-height: 0;
    padding: 30px 32px 12px;
    scrollbar-gutter: stable both-edges;
  }
  .messages,
  .welcome,
  .migration {
    max-width: var(--chat-width);
    width: 100%;
    margin: 0 auto;
  }
  .welcome {
    text-align: center;
    padding: 28px 0;
  }
  .welcome-mark {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    margin: 0 auto 20px;
    border-radius: 16px;
    background: var(--s-surface);
    color: var(--s-info-text);
  }
  .welcome h1 {
    font-size: clamp(24px, 2.5vw, 30px);
    font-weight: 600;
    letter-spacing: -0.035em;
  }
  .welcome p {
    font-size: 14px;
    color: var(--s-muted);
    margin: 12px 0 20px;
    line-height: 1.7;
  }
  .empty .chat-scroll {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
  }
  .empty .composer-wrap {
    padding-bottom: clamp(32px, 18vh, 190px);
  }
  article {
    margin: 0 0 32px;
    padding: 0;
    overflow-wrap: anywhere;
    min-width: 0;
    font-size: 15px;
    line-height: 1.8;
  }
  article.user {
    width: fit-content;
    max-width: 86%;
    margin-left: auto;
    background: var(--s-soft);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
  }
  article.user .role {
    display: none;
  }
  article.user.editing {
    width: 640px;
  }
  .role {
    font-size: 12px;
    font-weight: 500;
    color: var(--s-muted);
    margin-bottom: 12px;
  }
  .user-text,
  .reasoning {
    white-space: pre-wrap;
  }
  .thinking {
    margin: 0 0 14px;
  }
  .thinking summary {
    display: flex;
    align-items: center;
    width: fit-content;
    gap: 8px;
    font-size: 12px;
    color: var(--s-muted);
    padding: 5px 2px;
    border-radius: 5px;
  }
  .thinking summary:hover {
    color: var(--s-ink);
  }
  .thinking :global(.thinking-chevron) {
    transition: transform 0.15s ease;
  }
  .thinking[open] :global(.thinking-chevron) {
    transform: rotate(180deg);
  }
  .reasoning {
    font-size: 13px;
    line-height: 1.8;
    color: var(--s-muted);
    max-height: 320px;
    overflow: auto;
    padding: 4px 16px;
    margin: 10px 0 0 8px;
    border-left: 1px solid var(--s-line);
  }
  .attachment {
    max-width: min(320px, 100%);
    max-height: 240px;
    border-radius: 10px;
    margin: 8px 0;
  }
  .message-footer,
  .user-message-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 14px;
    margin-top: 16px;
    color: var(--s-muted);
    font-size: 11px;
    line-height: 1.6;
  }
  .user-message-actions { margin-top: 8px; justify-content: flex-end; }
  .message-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px 12px;
    font-variant-numeric: tabular-nums;
  }
  .message-meta > span + span::before {
    content: "·";
    margin-right: 12px;
    color: var(--s-muted);
  }
  .finish-warning {
    color: var(--s-warn-text);
  }
  .message-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  #app .message-editor {
    margin-top: 8px;
    width: 100%;
  }
  .message-editor textarea {
    min-height: 96px;
    padding: 12px;
  }
  .message-editor label {
    font-size: 12px;
    color: var(--s-muted);
  }
  .edit-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    padding: 8px;
  }
  #app .message-footer button,
  #app .user-message-actions button {
    border: 0;
    color: var(--s-muted);
    font-size: 11px;
    min-height: 28px;
    padding: 3px 6px;
    gap: 5px;
  }
  #app .message-footer button:hover,
  #app .user-message-actions button:hover {
    background: var(--s-soft);
    color: var(--s-ink);
  }
  .message-footer details {
    display: contents;
  }
  .message-footer summary {
    padding: 4px 2px;
    border-radius: 4px;
  }
  .message-footer summary:hover {
    color: var(--s-ink);
  }
  .message-details {
    flex-basis: 100%;
    padding: 12px 14px;
    line-height: 1.9;
    background: var(--s-surface);
    overflow-wrap: anywhere;
    border: 1px solid var(--s-line-soft);
    border-radius: 10px;
  }
  .composer-wrap {
    padding: 10px 32px 12px;
    width: 100%;
    max-width: calc(var(--chat-width) + 64px);
    margin: 0 auto;
  }
  form {
    border: 1px solid var(--s-line);
    border-radius: 18px;
    background: var(--s-surface);
    overflow: hidden;
    transition: border-color 0.15s;
  }
  form:focus-within {
    border-color: color-mix(in srgb, var(--s-info) 65%, var(--s-line));
  }
  textarea {
    resize: none;
    display: block;
    width: 100%;
    min-height: 58px;
    max-height: 200px;
    padding: 16px 18px 10px;
    border: 0;
    background: transparent;
    font-size: 15px;
    line-height: 1.7;
    overflow-y: auto;
  }
  #app textarea:focus {
    outline: 0;
    box-shadow: none;
  }
  textarea::placeholder {
    color: var(--s-muted);
  }
  .compose-actions {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 8px;
    padding: 4px 10px 10px;
  }
  .quick-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 3px 8px;
    min-width: 0;
    font-size: 11px;
    color: var(--s-muted);
  }
  #app .quick-controls button,
  #app .quick-controls .button {
    font-size: 12px;
    padding: 5px 7px;
    border: 0;
    color: var(--s-muted);
    gap: 5px;
  }
  #app .quick-controls button:hover {
    color: var(--s-ink);
  }
  .thinking-budget {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  #app .quick-controls .changed {
    color: var(--s-info-text);
  }
  .send-actions {
    display: flex;
    flex-shrink: 0;
    align-items: center;
    gap: 6px;
  }
  .send-actions .s-primary,
  .send-actions .stop {
    width: 36px;
    min-height: 36px;
    padding: 0;
    border-radius: 10px;
  }
  .composer-hint {
    text-align: right;
    padding: 8px 4px 0;
    font-size: 10px;
    color: var(--s-muted);
  }
  .button input {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    border: 0;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }
  #app .button:has(input:focus-visible) {
    outline: 2px solid var(--s-info);
    outline-offset: 2px;
  }
  #app .button input:focus-visible {
    outline: 0;
  }
  #app button.stop {
    color: var(--s-error-text);
    border-color: var(--s-error-text);
    background: var(--s-error-tint);
  }
  .chat-error,
  .chat-notice {
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 12px;
    margin-bottom: 10px;
    overflow-wrap: anywhere;
    line-height: 1.6;
  }
  .chat-error {
    background: var(--s-error-tint);
    color: var(--s-error-text);
  }
  .chat-notice {
    background: var(--s-info-tint);
    color: var(--s-info-text);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .chat-notice button {
    padding: 0 6px;
    background: transparent;
    border: 0;
    min-height: 26px;
  }
  .background-generation {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--s-info-text);
  }
  #app .background-generation button { font-size: 12px; padding: 4px 8px; }
  .live-line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--s-muted);
    margin-bottom: 10px;
    flex-wrap: wrap;
    font-variant-numeric: tabular-nums;
  }
  .live-line button {
    margin-left: auto;
    font-size: 11px;
    padding: 3px 6px;
    min-height: 28px;
    border: 0;
  }
  .pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--s-accent);
  }
  .drag-active form {
    border-color: var(--s-accent);
    outline: 2px dashed var(--s-accent);
    outline-offset: 3px;
  }
  .drop-hint {
    padding: 8px 12px;
    color: var(--s-info);
    font-size: 13px;
  }
  .text-attachment {
    margin-top: 8px;
    font-size: 12px;
  }
  .text-attachment summary { overflow-wrap: anywhere; cursor: pointer; }
  .text-attachment pre {
    max-height: 240px;
    overflow: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    padding: 8px;
    background: var(--s-surface);
  }
  .attachments button { max-width: 100%; overflow-wrap: anywhere; white-space: normal; }
  .attachments {
    padding: 10px 12px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .migration {
    margin-bottom: 24px;
    font-size: 13px;
  }
  .migration p {
    margin: 10px 0;
  }
  .row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .drawer-backdrop {
    position: fixed;
    inset: 0;
    background: #0005;
    z-index: 20;
  }
  #app .history-backdrop {
    position: fixed;
    inset: 0;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: #0006;
    z-index: 24;
  }
  .settings-drawer {
    position: fixed;
    right: 0;
    top: 0;
    bottom: 0;
    width: min(390px, 100%);
    background: var(--s-surface);
    border-left: 1px solid var(--s-line);
    display: flex;
    flex-direction: column;
    z-index: 21;
    box-shadow: var(--s-shadow);
  }
  .settings-drawer header,
  .settings-drawer footer {
    padding: 18px;
    border-bottom: 1px solid var(--s-line);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .settings-drawer footer {
    border-top: 1px solid var(--s-line);
    border-bottom: 0;
  }
  .settings-body {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 20px;
  }
  .settings-body p {
    font-size: 13px;
    color: var(--s-muted);
    margin-bottom: 18px;
  }
  .settings-body label {
    display: block;
    font-size: 13px;
    margin: 0 0 18px;
  }
  .settings-body label small {
    float: right;
    font-size: 11px;
  }
  .settings-body input:not([type="checkbox"]),
  .settings-body select {
    display: block;
    width: 100%;
    padding: 9px 10px;
    margin-top: 7px;
  }
  .settings-body > small {
    display: block;
    font-size: 11px;
    margin: -8px 0 18px;
    line-height: 1.7;
  }
  .settings-body .check {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  #app .settings-body .check input {
    appearance: none;
    position: relative;
    flex: none;
    width: 42px;
    height: 24px;
    margin: 0;
    padding: 0;
    border: 1px solid var(--s-muted);
    border-radius: 999px;
    background: var(--s-line);
    cursor: pointer;
  }
  #app .settings-body .check input::before {
    content: "";
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--s-muted);
  }
  #app .settings-body .check input:checked {
    background: var(--s-info);
    border-color: var(--s-info);
  }
  #app .settings-body .check input:checked::before {
    background: #fff;
    transform: translateX(18px);
  }
  .settings-body details summary {
    font-size: 13px;
    margin: 10px 0 20px;
  }
  .settings-body details {
    border-top: 1px solid var(--s-line);
    padding-top: 10px;
  }
  :global(#app .markdown-content :is(td, th)) {
    font-family: var(--s-font);
    font-size: 13px;
    line-height: 1.65;
    padding: 10px 12px;
  }
  :global(#app .markdown-content .table-wrapper) {
    margin: 18px 0;
    border: 1px solid var(--s-line-soft);
    border-radius: 10px;
  }
  :global(#app .markdown-content table:hover) {
    box-shadow: none;
  }
  :global(#app .markdown-content blockquote),
  :global(#app .markdown-content blockquote:hover) {
    margin: 18px 0;
    padding: 3px 16px;
    border-left: 2px solid var(--s-line);
    border-radius: 0;
    background: transparent;
    font-style: normal;
    color: var(--s-muted);
    transform: none;
    transition: none;
  }
  :global(#app .markdown-content blockquote::before),
  :global(#app .markdown-content hr::after) {
    content: none;
  }
  :global(#app .markdown-content hr) {
    height: 1px;
    background: var(--s-line-soft);
    margin: 24px 0;
  }

  .mobile-only {
    display: none;
  }
  #app[data-density="compact"] aside {
    padding: 16px 12px 12px;
    gap: 14px;
  }
  #app[data-density="compact"] .chat-header {
    min-height: 54px;
    padding: 8px 24px;
  }
  #app[data-density="compact"] .chat-scroll {
    padding-top: 22px;
  }
  #app[data-density="compact"] article {
    margin-bottom: 24px;
  }
  #app[data-density="compact"] .composer-wrap {
    padding-top: 8px;
  }
  :global(#app .markdown-content) {
    font-size: 15px;
    line-height: 1.8;
    color: var(--s-ink);
  }
  :global(#app .markdown-content p) {
    margin-block: 0.85em;
  }
  :global(#app .markdown-content :is(h1, h2, h3, h4)) {
    line-height: 1.4;
    letter-spacing: -0.015em;
    margin-top: 1.4em;
    margin-bottom: 0.65em;
  }
  :global(#app .markdown-content h1) {
    font-size: 1.55em;
  }
  :global(#app .markdown-content h2) {
    font-size: 1.35em;
  }
  :global(#app .markdown-content h3) {
    font-size: 1.15em;
  }
  :global(#app .markdown-content :is(ul, ol)) {
    margin-block: 0.8em;
  }
  :global(#app .markdown-content li) {
    margin-block: 0.3em;
  }
  :global(#app .markdown-content pre) {
    max-width: 100%;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.7;
  }
  :global(#app .markdown-content .code-block-wrapper) {
    border-color: var(--s-line-soft);
    border-radius: 10px;
    box-shadow: none;
    margin-block: 18px;
  }
  :global(#app .markdown-content .code-block-header) {
    color: var(--s-muted);
    background: var(--s-surface);
    border-bottom: 1px solid var(--s-line-soft);
    font-size: 11px;
  }
  :global(#app .markdown-content th) {
    background: var(--s-surface);
  }
  :global(#app .chat-main table) {
    display: table;
    max-width: 100%;
    font-size: 13px;
  }
  :global(#app .chat-main :is(iframe, img)) {
    max-width: 100%;
  }
  @media (min-width: 761px) and (hover: hover) and (pointer: fine) {
    .history-menu summary {
      opacity: 0;
    }
    .history-row:hover .history-menu summary,
    .history-row:focus-within .history-menu summary,
    .history-menu[open] summary {
      opacity: 1;
    }
  }
  @media (max-width: 1080px) {
    .model-heading {
      display: block;
    }
    .model-heading strong {
      display: block;
    }
    .model-heading small {
      display: block;
      margin-top: 2px;
    }
  }
  @media (max-width: 760px) {
    .chat-shell {
      grid-template-columns: 1fr;
    }
    aside {
      display: none;
    }
    aside.mobile-open {
      display: flex;
      position: absolute;
      inset: 0 14% 0 0;
      z-index: 25;
    }
    .mobile-only {
      display: inline-flex;
    }
    .chat-header,
    #app[data-density="compact"] .chat-header {
      padding: 9px 12px;
      gap: 8px;
      min-height: 58px;
    }
    .chat-header .chat-state {
      display: none;
    }
    .model-heading strong {
      font-size: 13px;
    }
    .model-heading small {
      font-size: 10px;
    }
    .chat-header > button {
      padding: 5px 8px;
    }
    .chat-scroll,
    #app[data-density="compact"] .chat-scroll {
      padding: 22px 16px 8px;
      scrollbar-gutter: auto;
    }
    .composer-wrap {
      padding: 8px 12px 12px;
    }
    .empty .composer-wrap {
      padding-bottom: clamp(24px, 12vh, 100px);
    }
    .welcome {
      padding: 20px 0;
    }
    .welcome-mark {
      margin-bottom: 16px;
    }
    .welcome p {
      font-size: 13px;
    }
    article,
    #app[data-density="compact"] article {
      margin-bottom: 26px;
    }
    article.user {
      max-width: 92%;
      padding: 10px 14px;
    }
    .reasoning {
      padding-right: 6px;
      max-height: 240px;
    }
    .message-footer {
      gap: 6px 10px;
    }
    textarea {
      padding: 14px 14px 8px;
      font-size: 16px;
      min-height: 56px;
    }
    .compose-actions {
      padding: 4px 8px 8px;
      gap: 4px;
    }
    .quick-controls {
      gap: 2px 4px;
    }
    #app .quick-controls button {
      font-size: 11px;
      padding-inline: 5px;
      gap: 4px;
    }
    .thinking-budget {
      font-size: 10px;
    }
    .send-actions > button:not(.s-primary):not(.stop) {
      max-width: 90px;
      white-space: normal;
      font-size: 10px;
      padding: 4px;
    }
    .composer-hint {
      display: none;
    }
    .settings-drawer {
      width: min(390px, 100%);
    }
  }
  @media (max-height: 560px) {
    .empty .composer-wrap {
      padding-bottom: 12px;
    }
    .welcome-mark {
      display: none;
    }
    .welcome {
      padding: 12px 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    form,
    .thinking :global(.thinking-chevron) {
      transition: none;
    }
  }
</style>
