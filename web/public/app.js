// // web/public/app.js
// const state = {
//   activeTab: "phase1",
//   backendUrl: null,
//   tagColors: { user: "#8ecbff", think: "#e6e6e6", answer: "#b9f6ca" },
//   tagVisible: { user: true, think: true, answer: true },
//   chats: {
//     phase1: [],
//     phase2: [],
//     phase3: [],
//     phase4: [],
//   },
// };

// const el = {
//   backendPortValue: document.getElementById("backendPortValue"),
//   refreshPortBtn: document.getElementById("refreshPortBtn"),
//   tagCheckboxes: document.getElementById("tagCheckboxes"),
//   tabs: Array.from(document.querySelectorAll(".tab")),
//   clearChatBtn: document.getElementById("clearChatBtn"),
//   promptInput: document.getElementById("promptInput"),
//   sendBtn: document.getElementById("sendBtn"),
//   chatPhase1: document.getElementById("chatPhase1"),
//   chatPhase2: document.getElementById("chatPhase2"),
//   chatPhase3: document.getElementById("chatPhase3"),
//   chatPhase4: document.getElementById("chatPhase4"),
// };

// function escapeHtml(s) {
//   return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
// }

// function parseTaggedResponse(text) {
//   const out = [];
//   const patterns = [
//     { tag: "think", re: /<think>([\s\S]*?)<\/think>/gi },
//     { tag: "answer", re: /<answer>([\s\S]*?)<\/answer>/gi },
//   ];

//   const matches = [];
//   for (const p of patterns) {
//     let m;
//     while ((m = p.re.exec(text)) !== null) {
//       matches.push({
//         tag: p.tag,
//         start: m.index,
//         end: m.index + m[0].length,
//         content: m[1].trim(),
//       });
//     }
//   }
//   matches.sort((a, b) => a.start - b.start);

//   if (matches.length === 0) return [{ tag: "answer", content: text.trim() }];

//   for (const m of matches) {
//     if (m.content.length > 0) out.push({ tag: m.tag, content: m.content });
//   }

//   const stripped = text
//     .replaceAll(/<think>[\s\S]*?<\/think>/gi, "")
//     .replaceAll(/<answer>[\s\S]*?<\/answer>/gi, "")
//     .trim();
//   if (stripped.length > 0) out.push({ tag: "answer", content: stripped });

//   return out;
// }

// function activeChatEl() {
//   if (state.activeTab === "phase1") return el.chatPhase1;
//   if (state.activeTab === "phase2") return el.chatPhase2;
//   if (state.activeTab === "phase3") return el.chatPhase3;
//   return el.chatPhase4;
// }

// function renderCheckboxes() {
//   const tags = Object.keys(state.tagColors);
//   for (const t of tags) {
//     if (!(t in state.tagVisible)) state.tagVisible[t] = true;
//   }

//   el.tagCheckboxes.innerHTML = "";
//   for (const tag of tags) {
//     const wrap = document.createElement("label");
//     wrap.className = "checkbox";

//     const cb = document.createElement("input");
//     cb.type = "checkbox";
//     cb.checked = Boolean(state.tagVisible[tag]);
//     cb.addEventListener("change", () => {
//       state.tagVisible[tag] = cb.checked;
//       renderChat();
//     });

//     const dot = document.createElement("span");
//     dot.className = "tag-dot";
//     dot.style.background = state.tagColors[tag];

//     const text = document.createElement("span");
//     text.textContent = tag;

//     wrap.appendChild(cb);
//     wrap.appendChild(dot);
//     wrap.appendChild(text);
//     el.tagCheckboxes.appendChild(wrap);
//   }
// }

// function renderChat() {
//   const chatEl = activeChatEl();
//   const messages = state.chats[state.activeTab];

//   chatEl.innerHTML = "";

//   for (const msg of messages) {
//     if (msg.role === "user") {
//       if (!state.tagVisible.user) continue;

//       const msgWrap = document.createElement("div");
//       msgWrap.className = "message user";
//       msgWrap.appendChild(makeBubble("user", msg.text));
//       chatEl.appendChild(msgWrap);
//       continue;
//     }

//     const segments = Array.isArray(msg.segments) ? msg.segments : [];
//     const msgWrap = document.createElement("div");
//     msgWrap.className = "message model";

//     for (const seg of segments) {
//       if (!state.tagVisible[seg.tag]) continue;
//       msgWrap.appendChild(makeBubble(seg.tag, seg.content));
//     }

//     if (msgWrap.childElementCount > 0) chatEl.appendChild(msgWrap);
//   }

//   chatEl.scrollTop = chatEl.scrollHeight;
// }

// function makeBubble(tag, content) {
//   const bubble = document.createElement("div");
//   bubble.className = "bubble";

//   const tagColor = state.tagColors[tag] ?? "#ffffff";
//   bubble.style.background = tagColor;
//   bubble.style.color = "#101318";

//   const tagEl = document.createElement("div");
//   tagEl.className = "tag";
//   tagEl.textContent = tag;
//   tagEl.style.color = "inherit";

//   const contentEl = document.createElement("div");
//   contentEl.className = "content";
//   contentEl.innerHTML = escapeHtml(content);

//   bubble.appendChild(tagEl);
//   bubble.appendChild(contentEl);
//   return bubble;
// }

// function setActiveTab(tab) {
//   state.activeTab = tab;

//   for (const b of el.tabs) {
//     b.classList.toggle("active", b.dataset.tab === tab);
//   }

//   el.chatPhase1.classList.toggle("active", tab === "phase1");
//   el.chatPhase2.classList.toggle("active", tab === "phase2");
//   el.chatPhase3.classList.toggle("active", tab === "phase3");
//   el.chatPhase4.classList.toggle("active", tab === "phase4");

//   renderChat();
// }

// async function refreshConfig() {
//   const resp = await fetch("/api/config");
//   const data = await resp.json();

//   state.backendUrl = data.backendUrl ?? null;
//   state.tagColors = data.tagColors ?? state.tagColors;

//   el.backendPortValue.textContent =
//     state.backendUrl == null ? "unknown" : String(state.backendUrl);

//   for (const tag of Object.keys(state.tagColors)) {
//     if (!(tag in state.tagVisible)) state.tagVisible[tag] = true;
//   }

//   renderCheckboxes();
//   renderChat();
// }

// function endpointForTab(tab) {
//   if (tab === "phase1") return "/api/phase1/reasoning";
//   if (tab === "phase2") return "/api/phase2/tools";
//   if (tab === "phase3") return "/api/phase3/rag";
//   return "/api/phase4/agent";
// }

// async function sendPrompt() {
//   const prompt = el.promptInput.value.trim();
//   if (prompt.length === 0) return;

//   state.chats[state.activeTab].push({ role: "user", text: prompt });
//   el.promptInput.value = "";
//   renderChat();

//   const endpoint = endpointForTab(state.activeTab);

//   let json;
//   try {
//     const resp = await fetch(endpoint, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ prompt }),
//     });
//     json = await resp.json();
//   } catch (err) {
//     state.chats[state.activeTab].push({
//       role: "model",
//       segments: [{ tag: "answer", content: `Network error: ${String(err)}` }],
//     });
//     renderChat();
//     return;
//   }

//   const raw =
//     typeof json?.response === "string" ? json.response : JSON.stringify(json, null, 2);

//   const segments = parseTaggedResponse(raw);

//   state.chats[state.activeTab].push({
//     role: "model",
//     raw,
//     segments,
//   });

//   renderChat();
// }

// function clearChat() {
//   state.chats[state.activeTab] = [];
//   renderChat();
// }

// function setupEvents() {
//   for (const b of el.tabs) {
//     b.addEventListener("click", () => setActiveTab(b.dataset.tab));
//   }

//   el.refreshPortBtn.addEventListener("click", () => refreshConfig());
//   el.clearChatBtn.addEventListener("click", () => clearChat());
//   el.sendBtn.addEventListener("click", () => sendPrompt());

//   el.promptInput.addEventListener("keydown", (e) => {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault();
//       sendPrompt();
//     }
//   });
// }

// setupEvents();
// await refreshConfig();
// setActiveTab("phase1");













// web/public/app.js
const state = {
  activeTab: "phase1",
  backendUrl: null,
  tagColors: { user: "#8ecbff", think: "#e6e6e6", answer: "#b9f6ca" },
  tagVisible: { user: true, think: true, answer: true },
  chats: {
    phase1: [],
    phase2: [],
    phase3: [],
    phase4: [],
  },
  isSending: false,
};

const el = {
  backendPortValue: document.getElementById("backendPortValue"),
  refreshPortBtn: document.getElementById("refreshPortBtn"),
  tagCheckboxes: document.getElementById("tagCheckboxes"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  clearChatBtn: document.getElementById("clearChatBtn"),
  promptInput: document.getElementById("promptInput"),
  sendBtn: document.getElementById("sendBtn"),
  chatPhase1: document.getElementById("chatPhase1"),
  chatPhase2: document.getElementById("chatPhase2"),
  chatPhase3: document.getElementById("chatPhase3"),
  chatPhase4: document.getElementById("chatPhase4"),
};

function setSending(isSending) {
  state.isSending = Boolean(isSending);

  el.sendBtn.disabled = state.isSending;
  el.promptInput.disabled = state.isSending;

  // Change button appearance with existing classes (no CSS required).
  // While sending, it becomes a "disabled-like" button.
  el.sendBtn.classList.toggle("primary", !state.isSending);
  el.sendBtn.classList.toggle("danger", state.isSending);

  el.sendBtn.textContent = state.isSending ? "Sending..." : "Send";
}

function escapeHtml(s) {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function parseTaggedResponse(text) {
  const out = [];
  const patterns = [
    { tag: "think", re: /<think>([\s\S]*?)<\/think>/gi },
    { tag: "answer", re: /<answer>([\s\S]*?)<\/answer>/gi },
  ];

  const matches = [];
  for (const p of patterns) {
    let m;
    while ((m = p.re.exec(text)) !== null) {
      matches.push({
        tag: p.tag,
        start: m.index,
        end: m.index + m[0].length,
        content: m[1].trim(),
      });
    }
  }
  matches.sort((a, b) => a.start - b.start);

  if (matches.length === 0) return [{ tag: "answer", content: text.trim() }];

  for (const m of matches) {
    if (m.content.length > 0) out.push({ tag: m.tag, content: m.content });
  }

  const stripped = text
    .replaceAll(/<think>[\s\S]*?<\/think>/gi, "")
    .replaceAll(/<answer>[\s\S]*?<\/answer>/gi, "")
    .trim();
  if (stripped.length > 0) out.push({ tag: "answer", content: stripped });

  return out;
}

function activeChatEl() {
  if (state.activeTab === "phase1") return el.chatPhase1;
  if (state.activeTab === "phase2") return el.chatPhase2;
  if (state.activeTab === "phase3") return el.chatPhase3;
  return el.chatPhase4;
}

function renderCheckboxes() {
  const tags = Object.keys(state.tagColors);
  for (const t of tags) {
    if (!(t in state.tagVisible)) state.tagVisible[t] = true;
  }

  el.tagCheckboxes.innerHTML = "";
  for (const tag of tags) {
    const wrap = document.createElement("label");
    wrap.className = "checkbox";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = Boolean(state.tagVisible[tag]);
    cb.addEventListener("change", () => {
      state.tagVisible[tag] = cb.checked;
      renderChat();
    });

    const dot = document.createElement("span");
    dot.className = "tag-dot";
    dot.style.background = state.tagColors[tag];

    const text = document.createElement("span");
    text.textContent = tag;

    wrap.appendChild(cb);
    wrap.appendChild(dot);
    wrap.appendChild(text);
    el.tagCheckboxes.appendChild(wrap);
  }
}

function renderChat() {
  const chatEl = activeChatEl();
  const messages = state.chats[state.activeTab];

  chatEl.innerHTML = "";

  for (const msg of messages) {
    if (msg.role === "user") {
      if (!state.tagVisible.user) continue;

      const msgWrap = document.createElement("div");
      msgWrap.className = "message user";
      msgWrap.appendChild(makeBubble("user", msg.text));
      chatEl.appendChild(msgWrap);
      continue;
    }

    const segments = Array.isArray(msg.segments) ? msg.segments : [];
    const msgWrap = document.createElement("div");
    msgWrap.className = "message model";

    for (const seg of segments) {
      if (!state.tagVisible[seg.tag]) continue;
      msgWrap.appendChild(makeBubble(seg.tag, seg.content));
    }

    if (msgWrap.childElementCount > 0) chatEl.appendChild(msgWrap);
  }

  chatEl.scrollTop = chatEl.scrollHeight;
}

function makeBubble(tag, content) {
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const tagColor = state.tagColors[tag] ?? "#ffffff";
  bubble.style.background = tagColor;
  bubble.style.color = "#101318";

  const tagEl = document.createElement("div");
  tagEl.className = "tag";
  tagEl.textContent = tag;
  tagEl.style.color = "inherit";

  const contentEl = document.createElement("div");
  contentEl.className = "content";
  contentEl.innerHTML = escapeHtml(content);

  bubble.appendChild(tagEl);
  bubble.appendChild(contentEl);
  return bubble;
}

function setActiveTab(tab) {
  state.activeTab = tab;

  for (const b of el.tabs) {
    b.classList.toggle("active", b.dataset.tab === tab);
  }

  el.chatPhase1.classList.toggle("active", tab === "phase1");
  el.chatPhase2.classList.toggle("active", tab === "phase2");
  el.chatPhase3.classList.toggle("active", tab === "phase3");
  el.chatPhase4.classList.toggle("active", tab === "phase4");

  renderChat();
}

async function refreshConfig() {
  const resp = await fetch("/api/config");
  const data = await resp.json();

  state.backendUrl = data.backendUrl ?? null;
  state.tagColors = data.tagColors ?? state.tagColors;

  el.backendPortValue.textContent =
    state.backendUrl == null ? "unknown" : String(state.backendUrl);

  for (const tag of Object.keys(state.tagColors)) {
    if (!(tag in state.tagVisible)) state.tagVisible[tag] = true;
  }

  renderCheckboxes();
  renderChat();
}

function endpointForTab(tab) {
  if (tab === "phase1") return "/api/phase1/reasoning";
  if (tab === "phase2") return "/api/phase2/tools";
  if (tab === "phase3") return "/api/phase3/rag";
  return "/api/phase4/agent";
}

async function sendPrompt() {
  if (state.isSending) return;

  const prompt = el.promptInput.value.trim();
  if (prompt.length === 0) return;

  setSending(true);

  state.chats[state.activeTab].push({ role: "user", text: prompt });
  el.promptInput.value = "";
  renderChat();

  const endpoint = endpointForTab(state.activeTab);

  let json;
  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    json = await resp.json();
  } catch (err) {
    state.chats[state.activeTab].push({
      role: "model",
      segments: [{ tag: "answer", content: `Network error: ${String(err)}` }],
    });
    renderChat();
    setSending(false);
    return;
  }

  const raw =
    typeof json?.response === "string" ? json.response : JSON.stringify(json, null, 2);

  const segments = parseTaggedResponse(raw);

  state.chats[state.activeTab].push({
    role: "model",
    raw,
    segments,
  });

  renderChat();
  setSending(false);
}

function clearChat() {
  state.chats[state.activeTab] = [];
  renderChat();
}

function setupEvents() {
  for (const b of el.tabs) {
    b.addEventListener("click", () => setActiveTab(b.dataset.tab));
  }

  el.refreshPortBtn.addEventListener("click", () => refreshConfig());
  el.clearChatBtn.addEventListener("click", () => clearChat());
  el.sendBtn.addEventListener("click", () => sendPrompt());

  el.promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });
}

setupEvents();
setSending(false);
await refreshConfig();
setActiveTab("phase1");
