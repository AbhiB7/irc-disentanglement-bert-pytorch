/**
 * IRC Conversation Link Visualization
 *
 * Three-pane layout:
 *   Left (20%) - Thread legend
 *   Center (50%) - Chat messages with thread-colored left borders
 *   Right (30%) - Static node-link graph
 *
 * Hover on any message or graph node highlights the full thread
 * in both the chat panel and the graph panel.
 */

var DATA_URL = "data/tiny.dev.json";
var messagesContainer = document.getElementById("messages");
var chatHeader = document.getElementById("chat-header");
var threadLegend = document.getElementById("thread-legend");
var svg = document.getElementById("links-svg");

var conversation = null;
var threadColors = {};

var NODE_RADIUS = 5;
var NODE_X = 50;
var msgYPositions = [];

function hashToHue(seed) {
    return ((seed * 137.508) % 360 + 360) % 360;
}

function escapeHtml(text) {
    var s = text;
    s = s.replace(/&/g, "&#38;");
    s = s.replace(/</g, "&#60;");
    s = s.replace(/>/g, "&#62;");
    s = s.replace(/"/g, "&#34;");
    s = s.replace(/'/g, "&#39;");
    return s;
}

function computeThreadLookup(n, links) {
    var uf = [];
    for (var i = 0; i < n; i++) uf[i] = i;
    function find(x) {
        while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    }
    function union(a, b) {
        var ra = find(a);
        var rb = find(b);
        if (ra !== rb) uf[rb] = ra;
    }
    for (var li = 0; li < links.length; li++) {
        union(links[li].child, links[li].parent);
    }
    var lookup = [];
    for (var i = 0; i < n; i++) lookup[i] = find(i);
    return lookup;
}

// ===== CHAT PANEL =====

function createMessageElement(msg, threadColor) {
    var div = document.createElement("div");
    div.className = "message";
    if (msg.is_system) div.className += " system";
    div.dataset.index = msg.index;
    div.style.borderLeftColor = threadColor || "transparent";
    if (!msg.is_system) {
        var frag = document.createDocumentFragment();
        function addSpan(className, text) {
            var sp = document.createElement("span");
            sp.className = className;
            sp.textContent = text;
            frag.appendChild(sp);
        }
        addSpan("index", msg.index + " ");
        addSpan("timestamp", "[" + (msg.timestamp || "--:--") + "]");
        addSpan("speaker", "<" + msg.speaker + ">");
        var tx = document.createElement("span");
        tx.className = "text";
        tx.innerHTML = escapeHtml(msg.text);
        frag.appendChild(tx);
        div.appendChild(frag);
        div.addEventListener("mouseenter", function () { onChatHover(msg.index); });
        div.addEventListener("mouseleave", resetAll);
    } else {
        var tx = document.createElement("span");
        tx.className = "text";
        tx.innerHTML = escapeHtml(msg.text);
        div.appendChild(tx);
    }
    return div;
}

function buildChat(data, childToRoot) {
    var msgs = data.messages;
    chatHeader.innerHTML = "<strong>" + escapeHtml(data.name) + "</strong> &mdash; "
        + msgs.length + " msgs, " + data.links.length + " links, "
        + data.threads.length + " threads";
    messagesContainer.innerHTML = "";
    var frag = document.createDocumentFragment();
    for (var i = 0; i < msgs.length; i++) {
        var msg = msgs[i];
        var color = threadColors[childToRoot[i]] || "";
        frag.appendChild(createMessageElement(msg, color));
    }
    messagesContainer.appendChild(frag);
    syncYPositions();
}

function buildLegend(data) {
    threadLegend.innerHTML = "";
    for (var t = 0; t < data.threads.length; t++) {
        var th = data.threads[t];
        var color = threadColors[th.id];
        var li = document.createElement("li");
        li.className = "thread-legend-item";
        var dot = document.createElement("span");
        dot.className = "thread-legend-color";
        dot.style.background = color;
        li.appendChild(dot);
        li.appendChild(document.createTextNode("Thread " + th.id + " (" + th.size + " msgs)"));
        threadLegend.appendChild(li);
    }
}

function syncYPositions() {
    var chats = document.querySelectorAll(".message:not(.system)");
    msgYPositions = [];
    for (var i = 0; i < chats.length; i++) {
        msgYPositions[parseInt(chats[i].dataset.index)] = chats[i].offsetTop + chats[i].offsetHeight / 2;
    }
}

function updateGraphPositions() {
    syncYPositions();
    var panel = document.getElementById("graph-panel");
    var panelW = panel.clientWidth || 300;
    // Find max Y
    var maxY = 0;
    for (var i = 0; i < msgYPositions.length; i++) {
        if (msgYPositions[i] && msgYPositions[i] > maxY) maxY = msgYPositions[i];
    }
    if (maxY < 10) maxY = 300;
    var height = maxY + 40;

    svg.setAttribute("viewBox", "0 0 " + panelW + " " + height);
    svg.style.height = height + "px";

    // Update edges
    var edges = svg.querySelectorAll("path");
    for (var li = 0; li < edges.length; li++) {
        var c = parseInt(edges[li].dataset.child);
        var p = parseInt(edges[li].dataset.parent);
        var y1 = msgYPositions[c] || 10;
        var y2 = msgYPositions[p] || 10;
        var cpOff = Math.max(Math.abs(y2 - y1) * 0.5, 15);
        var d = "M " + NODE_X + " " + y1 + " C " + (NODE_X + cpOff) + " " + y1 + ", " + (NODE_X + cpOff) + " " + y2 + ", " + NODE_X + " " + y2;
        edges[li].setAttribute("d", d);
    }

    // Update circles
    var circles = svg.querySelectorAll("circle");
    for (var i = 0; i < circles.length; i++) {
        var idx = parseInt(circles[i].dataset.index);
        var y = msgYPositions[idx] || 10;
        circles[i].setAttribute("cy", y);
    }

    // Update labels
    var labels = svg.querySelectorAll("text");
    for (var i = 0; i < labels.length; i++) {
        var txt = labels[i].textContent.trim();
        if (txt !== "") {
            var idx = parseInt(txt);
            var y = msgYPositions[idx] || 10;
            labels[i].setAttribute("y", y + 3);
        }
    }
}

// ===== GRAPH PANEL =====

function buildGraph(data, childToRoot) {
    var n = data.messages.length;
    var panel = document.getElementById("graph-panel");
    var panelW = panel.clientWidth || 300;

    svg.innerHTML = "";

    var ns = "http://www.w3.org/2000/svg";

    // Bezier edges (child -> parent)
    for (var li = 0; li < data.links.length; li++) {
        var c = data.links[li].child;
        var p = data.links[li].parent;
        if (c === p || data.messages[c].is_system) continue;

        var y1 = msgYPositions[c] || 10;
        var y2 = msgYPositions[p] || 10;
        var color = threadColors[childToRoot[c]] || "#888";

        var cpOff = Math.max(Math.abs(y2 - y1) * 0.5, 15);
        var d = "M " + NODE_X + " " + y1 + " C " + (NODE_X + cpOff) + " " + y1 + ", " + (NODE_X + cpOff) + " " + y2 + ", " + NODE_X + " " + y2;

        var path = document.createElementNS(ns, "path");
        path.setAttribute("d", d);
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "1.5");
        path.setAttribute("fill", "none");
        path.setAttribute("opacity", "0.7");
        path.dataset.child = c;
        path.dataset.parent = p;
        svg.appendChild(path);
    }

    // Node circles
    for (var i = 0; i < n; i++) {
        if (data.messages[i].is_system) continue;
        var y = msgYPositions[i] || 10;
        var color = threadColors[childToRoot[i]] || "#666";

        var label = document.createElementNS(ns, "text");
        label.setAttribute("x", "5");
        label.setAttribute("y", y + 3);
        label.setAttribute("fill", "#8080a0");
        label.setAttribute("font-size", "8");
        label.setAttribute("font-family", "monospace");
        label.textContent = i % 10 === 0 ? i : "";
        svg.appendChild(label);

        var circle = document.createElementNS(ns, "circle");
        circle.setAttribute("cx", NODE_X);
        circle.setAttribute("cy", y);
        circle.setAttribute("r", NODE_RADIUS);
        circle.setAttribute("fill", color);
        circle.dataset.index = i;
        circle.style.cursor = "pointer";
        svg.appendChild(circle);

        circle.addEventListener("mouseenter", function () {
            highlightThread(parseInt(this.dataset.index));
        });
        circle.addEventListener("mouseleave", resetAll);
    }
}

// ===== HIGHLIGHT =====

function buildChildToRoot() {
    return computeThreadLookup(conversation.messages.length, conversation.links);
}
function getRoot(msgIndex) {
    var n = conversation.messages.length;
    var ctr = computeThreadLookup(n, conversation.links);
    return ctr[msgIndex];
}

function highlightThread(msgIndex) {
    var n = conversation.messages.length;
    var ctr = computeThreadLookup(n, conversation.links);
    var root = ctr[msgIndex];

    // Chat messages
    var chats = document.querySelectorAll(".message:not(.system)");
    for (var i = 0; i < chats.length; i++) {
        var idx = parseInt(chats[i].dataset.index);
        var match = (ctr[idx] === root);
        if (match) {
            chats[i].classList.add("message-highlight");
            chats[i].classList.remove("message-dimming");
        } else {
            chats[i].classList.add("message-dimming");
            chats[i].classList.remove("message-highlight");
        }
    }

    // Graph edges
    var edges = svg.querySelectorAll("path");
    for (var i = 0; i < edges.length; i++) {
        var child = parseInt(edges[i].dataset.child);
        edges[i].setAttribute("opacity", (ctr[child] === root) ? "0.8" : "0.08");
    }

    // Graph nodes + labels
    var circles = svg.querySelectorAll("circle");
    for (var i = 0; i < circles.length; i++) {
        var idx = parseInt(circles[i].dataset.index);
        circles[i].setAttribute("opacity", (ctr[idx] === root) ? "1.0" : "0.1");
    }
    var labels = svg.querySelectorAll("text");
    for (var i = 0; i < labels.length; i++) {
        var txt = labels[i].textContent.trim();
        if (txt !== "") {
            labels[i].setAttribute("opacity", (ctr[parseInt(txt)] === root) ? "1.0" : "0.1");
        }
    }

    // Scroll sync from chat -> graph
    var node = svg.querySelector('circle[data-index="' + msgIndex + '"]');
    if (node) node.scrollIntoView({ block: "center", behavior: "smooth" });
}

function resetAll() {
    var chats = document.querySelectorAll(".message");
    for (var i = 0; i < chats.length; i++) {
        chats[i].classList.remove("message-dimming", "message-highlight");
    }
    var edges = svg.querySelectorAll("path");
    for (var i = 0; i < edges.length; i++) {
        edges[i].setAttribute("opacity", "0.7");
    }
    var all = svg.querySelectorAll("circle, text");
    for (var i = 0; i < all.length; i++) {
        all[i].setAttribute("opacity", "1.0");
    }
}

// Chat hover handler (dispatches to highlightThread)
function onChatHover(msgIndex) {
    highlightThread(msgIndex);
}

// ===== BUILD =====

function build(data) {
    conversation = data;
    var n = data.messages.length;
    var ctr = computeThreadLookup(n, data.links);

    threadColors = {};
    for (var t = 0; t < data.threads.length; t++) {
        var tid = data.threads[t].id;
        threadColors[tid] = "hsl(" + hashToHue(tid) + ", 70%, 55%)";
    }

    buildLegend(data);
    buildChat(data, ctr);
    buildGraph(data, ctr);
    updateGraphPositions();
}

fetch(DATA_URL)
    .then(function (r) { return r.json(); })
    .then(build)
    .catch(function (err) {
        messagesContainer.innerHTML = "<p style=\"color: #ff6b6b; padding: 20px;\">"
            + "Failed to load conversation: " + err.message + ".<br>"
            + "Make sure you've run scripts/export_chat_json.py first."
            + "</p>";
    });