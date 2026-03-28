const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const promptBox = document.getElementById("prompt");
const statusEl = document.getElementById("status");
const clearBtn = document.getElementById("clearBtn");
const restartBtn = document.getElementById("restartBtn");

// Replace this with your UNO Q host IP if it changes
const API_BASE = "http://192.168.1.226:8090";

function addMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "UNO Q";

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;

  msg.appendChild(roleEl);
  msg.appendChild(body);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  const text = await res.text();

  try {
    return JSON.parse(text);
  } catch {
    return { ok: false, error: `Non-JSON response: ${text}` };
  }
}

async function checkHealth() {
  try {
    const data = await apiGet("/api/health");

    if (data.ok) {
      statusEl.textContent = "Backend ready";
    } else {
      statusEl.textContent =
        data.error ||
        (data.errors ? data.errors.join(" | ") : "Backend not ready");
    }
  } catch (err) {
    statusEl.textContent = `Cannot reach backend: ${err.message}`;
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();

  const prompt = promptBox.value.trim();
  if (!prompt) return;

  addMessage("user", prompt);
  promptBox.value = "";
  statusEl.textContent = "Generating...";

  try {
    const data = await apiGet(`/api/chat/${encodeURIComponent(prompt)}`);

    if (!data.ok) {
      addMessage("assistant", `Error: ${data.error || JSON.stringify(data)}`);
      statusEl.textContent = "Error";
      return;
    }

    addMessage("assistant", data.answer || "(No answer field returned)");
    statusEl.textContent = "Ready";
  } catch (err) {
    addMessage("assistant", `Network error: ${err.message}`);
    statusEl.textContent = "Network error";
  }
});

restartBtn.addEventListener("click", async () => {
  statusEl.textContent = "Restarting model...";

  try {
    const data = await apiGet("/api/restart");

    if (!data.ok) {
      statusEl.textContent = `Restart error: ${data.error || JSON.stringify(data)}`;
      return;
    }

    statusEl.textContent = "Model restarted";
  } catch (err) {
    statusEl.textContent = `Restart error: ${err.message}`;
  }
});

clearBtn.addEventListener("click", () => {
  chat.innerHTML = "";
});

checkHealth();