/* Message Intelligence dashboard.
 *
 * Renders sanitized data fetched from the /api endpoints. All dynamic content
 * is written with textContent / createElement - never innerHTML - so no value
 * from the API can be interpreted as HTML.
 */
"use strict";

(function () {
  const PAGE_SIZE = 100;

  const state = {
    search: "",
    category: "",
    sensitive: "",
    offset: 0,
  };

  async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
      let detail = "Request failed";
      try {
        const body = await response.json();
        if (body && body.detail) detail = String(body.detail);
      } catch (_err) {
        /* keep default message */
      }
      throw new Error(`HTTP ${response.status}: ${detail}`);
    }
    return response.json();
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value === null || value === undefined ? "—" : String(value);
  }

  function clearChildren(id) {
    const node = document.getElementById(id);
    if (node) node.replaceChildren();
  }

  function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.replaceChildren(el("p", "error", message));
  }

  /* ------------------------------------------------------------- stats */

  async function loadStats() {
    try {
      const data = await fetchJSON("/api/stats");
      setText("stat-total", data.total_messages);
      setText("stat-classified", data.classified_messages);
      setText("stat-tasks", data.task_event_count);
      setText("stat-sensitive", data.sensitive_messages);

      const validation = await fetchJSON("/api/validation");
      setText("status-messages", validation.summary.total_messages);
      setText("status-processed", validation.summary.classified_messages);
      setText("status-validation", validation.validation_status);
      setText("status-sensitive", data.sensitive_messages);
      setText("status-tasks", data.task_event_count);
      setText(
        "status-mandatory",
        `${validation.report.mandatory_messages_processed}/${validation.report.mandatory_messages_found}`
      );
      setText("status-leak", validation.report.sensitive_value_leak_check);
    } catch (err) {
      showError("summary-cards", err.message);
      showError("pipeline-status", err.message);
    }
  }

  /* ----------------------------------------------------------- messages */

  async function loadMessages() {
    const table = document.getElementById("message-rows");
    table.replaceChildren(
      el("tr", "empty-row", null).appendChild(el("td", null, "Loading…")).parentNode
    );
    const params = new URLSearchParams({
      limit: String(PAGE_SIZE),
      offset: String(state.offset),
    });
    if (state.search) params.set("search", state.search);
    if (state.category) params.set("category", state.category);
    if (state.sensitive) params.set("sensitive", state.sensitive);

    try {
      const data = await fetchJSON(`/api/messages?${params.toString()}`);
      renderMessageRows(data.items, data.total);
      renderPagination(data.total);
    } catch (err) {
      table.replaceChildren(el("tr", "empty-row", null).appendChild(el("td", "error", err.message)).parentNode);
    }
  }

  function renderMessageRows(items, total) {
    const table = document.getElementById("message-rows");
    table.replaceChildren();
    if (items.length === 0) {
      const row = el("tr", "empty-row");
      const cell = el("td", null, total === 0 ? "No messages match the filters." : "No results on this page.");
      cell.colSpan = 6;
      row.appendChild(cell);
      table.appendChild(row);
      return;
    }
    for (const item of items) {
      const row = el("tr", "clickable");
      row.appendChild(el("td", null, item.message_id));
      row.appendChild(el("td", null, formatTimestamp(item.timestamp)));
      row.appendChild(el("td", null, item.sender));
      row.appendChild(el("td", null, item.category.replace(/_/g, " ")));
      row.appendChild(el("td", null, formatConfidence(item.confidence)));
      row.appendChild(
        el("td", null).appendChild(
          el("span", item.has_sensitive ? "tag sensitive" : "tag clean", item.has_sensitive ? "SENSITIVE" : "clean")
        ).parentNode
      );
      row.addEventListener("click", () => loadMessageDetail(item.message_id));
      table.appendChild(row);
    }
  }

  function renderPagination(total) {
    const pagination = document.getElementById("message-pagination");
    pagination.hidden = total === 0;
    const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    const current = Math.floor(state.offset / PAGE_SIZE) + 1;
    setText("page-info", `Page ${current} of ${pages} · ${total} messages`);
    document.getElementById("page-prev").disabled = current <= 1;
    document.getElementById("page-next").disabled = current >= pages;
  }

  function firstPage() {
    state.offset = 0;
  }

  /* ---------------------------------------------------- message detail */

  async function loadMessageDetail(messageId) {
    const panel = document.getElementById("message-detail");
    const body = document.getElementById("detail-body");
    panel.hidden = false;
    setText("detail-title", `Message ${messageId}`);
    body.replaceChildren(el("p", "hint", "Loading message…"));

    try {
      const data = await fetchJSON(`/api/messages/${encodeURIComponent(messageId)}`);
      renderDetail(body, data);
    } catch (err) {
      body.replaceChildren(el("p", "error", err.message));
    }
  }

  function renderDetail(body, data) {
    body.replaceChildren();

    body.appendChild(el("div", "safe-message", data.safe_message));

    const facts = el("dl", "detail-grid");
    facts.appendChild(fact("Category", data.classification.category.replace(/_/g, " ")));
    facts.appendChild(fact("Confidence", formatConfidence(data.classification.confidence)));
    facts.appendChild(fact("Method", data.classification.method));
    facts.appendChild(fact("Reason", data.classification.reason));
    facts.appendChild(
      fact("Security", data.security.has_detection ? "SENSITIVE" : "No sensitive values detected")
    );
    body.appendChild(facts);

    if (data.extracted_items && data.extracted_items.length > 0) {
      body.appendChild(el("h3", null, "Extracted tasks / events"));
      const list = el("ul", "items-list");
      for (const item of data.extracted_items) {
        const li = el("li");
        li.appendChild(el("strong", null, item.title));
        li.appendChild(el("span", "tag", item.type));
        li.appendChild(el("span", "tag", item.priority));
        if (item.date) li.appendChild(el("span", "tag", `date: ${item.date}`));
        if (item.deadline) li.appendChild(el("span", "tag", `deadline: ${item.deadline}`));
        if (item.time) li.appendChild(el("span", "tag", `time: ${item.time}`));
        list.appendChild(li);
      }
      body.appendChild(list);
    } else {
      body.appendChild(el("p", "hint", "No tasks or events were extracted from this message."));
    }

    if (data.security.detections && data.security.detections.length > 0) {
      body.appendChild(el("h3", null, "Security detections (masked)"));
      const list = el("ul", "items-list");
      for (const detection of data.security.detections) {
        const li = el("li");
        li.appendChild(el("span", "tag", detection.sensitivity_type.replace(/_/g, " ")));
        li.appendChild(el("span", "tag", detection.risk));
        li.appendChild(document.createTextNode(` ${detection.masked_text} `));
        li.appendChild(el("span", null, detection.recommended_action));
        list.appendChild(li);
      }
      body.appendChild(list);
    }
  }

  function fact(label, value) {
    const wrapper = document.createElement("div");
    const dt = el("dt", null, label);
    const dd = el("dd", null, value);
    wrapper.appendChild(dt);
    wrapper.appendChild(dd);
    return wrapper;
  }

  /* ---------------------------------------------------------- sensitive */

  async function loadSensitive() {
    const rows = document.getElementById("sensitive-rows");
    rows.replaceChildren(el("tr", "empty-row", null).appendChild(el("td", null, "Loading…")).parentNode);
    try {
      const data = await fetchJSON(`/api/sensitive?limit=100&offset=0`);
      rows.replaceChildren();
      if (data.items.length === 0) {
        const row = el("tr", "empty-row");
        const cell = el("td", null, "No sensitive messages.");
        cell.colSpan = 5;
        row.appendChild(cell);
        rows.appendChild(row);
        return;
      }
      for (const result of data.items) {
        for (const detection of result.detections) {
          const row = el("tr", "clickable");
          row.appendChild(el("td", null, result.message_id));
          row.appendChild(el("td", null, detection.sensitivity_type.replace(/_/g, " ")));
          row.appendChild(el("td", null, detection.risk));
          row.appendChild(el("td", null, detection.masked_text));
          row.appendChild(el("td", null, detection.recommended_action));
          row.addEventListener("click", () => loadMessageDetail(result.message_id));
          rows.appendChild(row);
        }
      }
    } catch (err) {
      rows.replaceChildren(el("tr", "empty-row", null).appendChild(el("td", "error", err.message)).parentNode);
    }
  }

  /* -------------------------------------------------------- mandatory */

  async function loadMandatory() {
    const list = document.getElementById("mandatory-list");
    list.replaceChildren(el("p", "hint", "Loading mandatory messages…"));
    try {
      const data = await fetchJSON("/api/demo/mandatory");
      setText("mandatory-count", `${data.processed}/${data.found}`);
      list.replaceChildren();
      data.results.forEach((message, index) => {
        const card = el("div", "mandatory-item");
        const number = el(
          "span",
          "mandatory-number",
          `${String(index + 1).padStart(2, "0")} of ${String(data.found).padStart(2, "0")}`
        );
        const heading = el("h4", null, message.message_id);
        heading.appendChild(el("span", "tag", message.classification.category.replace(/_/g, " ")));
        card.appendChild(number);
        card.appendChild(heading);
        card.appendChild(
          el("p", "meta", `${formatTimestamp(message.timestamp)} · ${message.sender}`)
        );
        card.appendChild(
          el("p", "meta", `Confidence ${formatConfidence(message.classification.confidence)}`)
        );
        const summary = message.extracted_items && message.extracted_items.length > 0
          ? message.extracted_items.map((item) => `${item.type}: ${item.title}`).join(" · ")
          : "No extracted tasks/events";
        card.appendChild(el("p", "meta", summary));
        card.appendChild(
          el(
            "span",
            message.security.has_detection ? "tag sensitive" : "tag clean",
            message.security.has_detection ? "SENSITIVE" : "clean"
          )
        );
        card.appendChild(el("p", "masked-preview", message.safe_message));
        card.addEventListener("click", () => {
          loadMessageDetail(message.message_id);
          scrollToDetail();
        });
        list.appendChild(card);
      });
    } catch (err) {
      showError("mandatory-list", err.message);
    }
  }

  /* ---------------------------------------------------------- validation */

  async function loadValidation() {
    try {
      const data = await fetchJSON("/api/validation");
      setText("val-processed", `${data.summary.classified_messages}/${data.summary.total_messages}`);
      setText("val-missing", data.report.missing_message_ids);
      setText("val-duplicates", data.report.duplicate_message_ids);
      setText("val-leak", data.report.sensitive_value_leak_check);
      setText("val-status", data.validation_status);
      const issues = document.getElementById("validation-issues");
      issues.replaceChildren();
      if (data.report.issues && data.report.issues.length > 0) {
        for (const issue of data.report.issues) {
          issues.appendChild(el("p", "error", `[${issue.code}] ${issue.detail}`));
        }
      } else {
        issues.appendChild(el("p", "hint", "No validation issues reported."));
      }
    } catch (err) {
      showError("validation-facts", err.message);
    }
  }

  /* -------------------------------------------------------------- misc */

  function formatTimestamp(value) {
    if (!value) return "—";
    return String(value).replace("T", " ").slice(0, 16);
  }

  function formatConfidence(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : "—";
  }

  function scrollToDetail() {
    const panel = document.getElementById("message-detail");
    if (panel) panel.scrollIntoView({ block: "start" });
  }

  function bindControls() {
    document.getElementById("apply-filters").addEventListener("click", () => {
      state.search = document.getElementById("search-input").value.trim();
      state.category = document.getElementById("category-filter").value;
      state.sensitive = document.getElementById("sensitive-filter").value;
      firstPage();
      loadMessages();
    });
    document.getElementById("search-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter") document.getElementById("apply-filters").click();
    });
    document.getElementById("page-prev").addEventListener("click", () => {
      state.offset = Math.max(0, state.offset - PAGE_SIZE);
      loadMessages();
    });
    document.getElementById("page-next").addEventListener("click", () => {
      state.offset += PAGE_SIZE;
      loadMessages();
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    bindControls();
    loadStats();
    loadMessages();
    loadSensitive();
    loadMandatory();
    loadValidation();
  });
})();
