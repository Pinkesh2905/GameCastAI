/* ==========================================================================
   Charts, drawn by hand in SVG.

   No chart library, for two reasons. The worm is cricket's own graph and has
   conventions no general-purpose library knows about - territory shaded to
   whichever side is ahead of the 50% line, wickets as ticks along the bottom,
   overs rather than time on the x axis. And a library's default palette is
   the fastest way to make a page look like every other page.

   Everything scales with viewBox so the same code serves a phone and a
   desktop; nothing here measures the DOM.
   ========================================================================== */

(function (window, document) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function svg(width, height, label) {
    var node = el("svg", {
      viewBox: "0 0 " + width + " " + height,
      preserveAspectRatio: "none",
      role: "img",
    });
    node.setAttribute("preserveAspectRatio", "xMidYMid meet");
    if (label) {
      var title = el("title", {}, node);
      title.textContent = label;
    }
    return node;
  }

  /* A chart authored at desktop proportions becomes a 100px-tall smear on a
     phone. Each chart therefore declares a wide and a narrow geometry and
     picks between them from the width it has actually been given. */
  var NARROW = 520;

  function pickSize(host, wide, narrow) {
    var width = host.getBoundingClientRect().width || wide[0];
    return width < NARROW ? narrow : wide;
  }

  function path(points, close) {
    if (!points.length) return "";
    var d = "M" + points[0][0].toFixed(2) + " " + points[0][1].toFixed(2);
    for (var i = 1; i < points.length; i++) {
      d += "L" + points[i][0].toFixed(2) + " " + points[i][1].toFixed(2);
    }
    return close ? d + "Z" : d;
  }

  /* --------------------------------------------------------------------
     Score curve: win probability across every score the side could be on.
     The point of this one is the cliff - you can see the exact score where
     the chase stops being a coin flip.
     -------------------------------------------------------------------- */

  function scoreCurve(host, data, current) {
    host.textContent = "";
    if (!data || !data.length) return;

    var size = pickSize(host, [560, 230], [380, 250]);
    var W = size[0], H = size[1];
    var padL = 34, padR = 12, padT = 12, padB = 26;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var maxScore = data[data.length - 1].score || 1;
    var x = function (s) { return padL + (s / maxScore) * plotW; };
    var y = function (p) { return padT + (1 - p) * plotH; };

    var node = svg(W, H, "Win probability against score");

    // Same two-tone convention as the worm: amber where the chasing side is
    // ahead of even money, leather red where it is behind.
    var defs = el("defs", {}, node);
    var clipTop = el("clipPath", { id: "curve-clip-top" }, defs);
    el("rect", { x: 0, y: 0, width: W, height: y(0.5) }, clipTop);
    var clipBottom = el("clipPath", { id: "curve-clip-bottom" }, defs);
    el("rect", { x: 0, y: y(0.5), width: W, height: H - y(0.5) }, clipBottom);

    // Horizontal grid at quarters, with the even-money line called out.
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
      el("line", {
        x1: padL, x2: W - padR, y1: y(p), y2: y(p),
        class: p === 0.5 ? "even-line" : "grid-line",
      }, node);
      el("text", {
        x: padL - 6, y: y(p) + 3, "text-anchor": "end", class: "axis-label",
      }, node).textContent = Math.round(p * 100);
    });

    var points = data.map(function (d) { return [x(d.score), y(d.probability)]; });

    // Fill down to the even-money line rather than to zero, so the shape
    // reads as "ahead" and "behind" instead of as a generic area chart.
    var mid = y(0.5);
    var area = points.slice();
    area.push([x(maxScore), mid]);
    area.unshift([x(0), mid]);
    el("path", { d: path(area, true), fill: "var(--bat-soft)",
      "clip-path": "url(#curve-clip-top)" }, node);
    el("path", { d: path(area, true), fill: "var(--bowl-soft)",
      "clip-path": "url(#curve-clip-bottom)" }, node);

    var line = path(points);
    el("path", { d: line, fill: "none", stroke: "var(--bat)", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      "clip-path": "url(#curve-clip-top)" }, node);
    el("path", { d: line, fill: "none", stroke: "var(--bowl)", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      "clip-path": "url(#curve-clip-bottom)" }, node);

    if (current && typeof current.score === "number") {
      var cx = x(current.score);
      el("line", {
        x1: cx, x2: cx, y1: padT, y2: padT + plotH,
        stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "2 3",
      }, node);
      el("circle", {
        cx: cx, cy: y(current.probability), r: 4,
        fill: "var(--ground)",
        stroke: current.probability >= 0.5 ? "var(--bat)" : "var(--bowl)",
        "stroke-width": 2,
      }, node);
      el("text", {
        x: Math.min(cx + 7, W - padR - 42), y: Math.max(y(current.probability) - 9, padT + 9),
        class: "axis-label", fill: "var(--ink)",
      }, node).textContent = "now";
    }

    [0, Math.round(maxScore / 2), maxScore].forEach(function (s, i) {
      el("text", {
        x: x(s), y: H - 8,
        "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle",
        class: "axis-label",
      }, node).textContent = s;
    });

    host.appendChild(node);
  }

  /* --------------------------------------------------------------------
     Wicket sensitivity: the same score with a different number of wickets
     standing. Drawn as columns because the variable is discrete.
     -------------------------------------------------------------------- */

  function wicketBars(host, data, currentWicketsLeft) {
    host.textContent = "";
    if (!data || !data.length) return;

    var size = pickSize(host, [400, 180], [360, 200]);
    var W = size[0], H = size[1];
    var padL = 30, padR = 8, padT = 10, padB = 24;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var slot = plotW / data.length;
    var barW = Math.max(slot - 6, 4);

    var node = svg(W, H, "Win probability by wickets in hand");

    [0, 0.5, 1].forEach(function (p) {
      var yy = padT + (1 - p) * plotH;
      el("line", { x1: padL, x2: W - padR, y1: yy, y2: yy,
        class: p === 0.5 ? "even-line" : "grid-line" }, node);
      el("text", { x: padL - 6, y: yy + 3, "text-anchor": "end", class: "axis-label" },
        node).textContent = Math.round(p * 100);
    });

    data.forEach(function (d, i) {
      var isNow = d.wickets_left === currentWicketsLeft;
      var h = d.probability * plotH;
      el("rect", {
        x: padL + i * slot + (slot - barW) / 2,
        y: padT + plotH - h,
        width: barW, height: Math.max(h, 1),
        fill: isNow ? "var(--bat)" : "var(--surface-3)",
      }, node);
      el("text", {
        x: padL + i * slot + slot / 2, y: H - 8,
        "text-anchor": "middle", class: "axis-label",
        fill: isNow ? "var(--bat)" : "var(--ink-3)",
      }, node).textContent = d.wickets_left;
    });

    host.appendChild(node);
  }

  /* --------------------------------------------------------------------
     Projection band: where the innings is heading, and how wide the honest
     range around it is.
     -------------------------------------------------------------------- */

  function projectionBand(host, projection, par, score) {
    host.textContent = "";
    if (!projection) return;

    // The drawing is a single horizontal axis with labels above and below it,
    // so the box is sized to that band rather than left with dead space under.
    var size = pickSize(host, [480, 86], [380, 92]);
    var W = size[0], H = size[1];
    var padL = 10, padR = 10, padT = 34;
    var plotW = W - padL - padR;

    var lo = Math.min(projection.low, par, score) - 12;
    var hi = Math.max(projection.high, par) + 12;
    var span = Math.max(hi - lo, 1);
    var x = function (v) { return padL + ((v - lo) / span) * plotW; };
    var axis = padT + 12;

    var node = svg(W, H, "Projected total with its range");

    el("line", { x1: padL, x2: W - padR, y1: axis, y2: axis,
      stroke: "var(--rule-strong)", "stroke-width": 1 }, node);

    el("rect", {
      x: x(projection.low), y: axis - 9,
      width: Math.max(x(projection.high) - x(projection.low), 2), height: 18,
      fill: "var(--bat-soft)",
    }, node);

    [["low", projection.low], ["high", projection.high]].forEach(function (pair) {
      el("line", { x1: x(pair[1]), x2: x(pair[1]), y1: axis - 9, y2: axis + 9,
        stroke: "var(--bat-line)", "stroke-width": 1 }, node);
      el("text", { x: x(pair[1]), y: axis + 24, "text-anchor": "middle", class: "axis-label" },
        node).textContent = pair[1];
    });

    // Venue par, so the projection reads as "big" or "small" for this ground.
    el("line", { x1: x(par), x2: x(par), y1: axis - 15, y2: axis + 15,
      stroke: "var(--bowl)", "stroke-width": 1, "stroke-dasharray": "3 2" }, node);
    el("text", { x: x(par), y: axis - 21, "text-anchor": "middle",
      class: "axis-label", fill: "var(--bowl)" }, node).textContent = "par " + Math.round(par);

    el("circle", { cx: x(projection.projected), cy: axis, r: 5,
      fill: "var(--bat)", stroke: "var(--ground)", "stroke-width": 2 }, node);

    host.appendChild(node);
  }

  /* --------------------------------------------------------------------
     THE WORM.

     Cricket's own win-probability graph. Territory is shaded toward whoever
     is ahead of the even-money line, wickets appear as ticks along the
     bottom axis, and the whole thing is scrubbable: the returned controller
     exposes setIndex so the player and the keyboard can drive it.
     -------------------------------------------------------------------- */

  function worm(host, balls, options) {
    host.textContent = "";
    if (!balls || !balls.length) return null;

    options = options || {};
    // Portrait-leaning on a phone so the line has room to move and the
    // playhead is wide enough to catch with a thumb.
    var size = pickSize(host, [720, 260], [380, 280]);
    var W = size[0], H = size[1];
    var padL = 30, padR = 10, padT = 14, padB = 34;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var last = balls.length - 1;
    var x = function (i) { return padL + (i / Math.max(last, 1)) * plotW; };
    var y = function (p) { return padT + (1 - p) * plotH; };
    var mid = y(0.5);

    var node = svg(W, H, "Win probability through the chase");
    node.setAttribute("class", "worm-svg");
    node.style.touchAction = "pan-y";   // let the page scroll, we take the drag

    var defs = el("defs", {}, node);
    // Two clips so one line can be painted amber above the halfway mark and
    // leather red below it, which is the convention the worm follows.
    var clipTop = el("clipPath", { id: "worm-clip-top" }, defs);
    el("rect", { x: 0, y: 0, width: W, height: mid }, clipTop);
    var clipBottom = el("clipPath", { id: "worm-clip-bottom" }, defs);
    el("rect", { x: 0, y: mid, width: W, height: H - mid }, clipBottom);

    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
      el("line", { x1: padL, x2: W - padR, y1: y(p), y2: y(p),
        class: p === 0.5 ? "even-line" : "grid-line" }, node);
      el("text", { x: padL - 6, y: y(p) + 3, "text-anchor": "end", class: "axis-label" },
        node).textContent = Math.round(p * 100);
    });

    var points = balls.map(function (b, i) { return [x(i), y(b.probability)]; });

    var above = points.slice();
    above.push([x(last), mid]);
    above.unshift([x(0), mid]);
    el("path", { d: path(above, true), fill: "var(--bat-soft)",
      "clip-path": "url(#worm-clip-top)" }, node);
    el("path", { d: path(above, true), fill: "var(--bowl-soft)",
      "clip-path": "url(#worm-clip-bottom)" }, node);

    var line = path(points);
    el("path", { d: line, fill: "none", stroke: "var(--bat)", "stroke-width": 1.8,
      "stroke-linejoin": "round", "clip-path": "url(#worm-clip-top)" }, node);
    el("path", { d: line, fill: "none", stroke: "var(--bowl)", "stroke-width": 1.8,
      "stroke-linejoin": "round", "clip-path": "url(#worm-clip-bottom)" }, node);

    // Wickets as ticks on the floor of the chart - the scorebook convention.
    balls.forEach(function (b, i) {
      if (!b.wicket) return;
      el("line", { x1: x(i), x2: x(i), y1: padT + plotH, y2: padT + plotH + 7,
        stroke: "var(--bowl)", "stroke-width": 1.5 }, node);
    });

    // Over markers along the bottom.
    var totalBalls = options.totalBalls || 120;
    var overStep = W < 500 ? 10 : 5;
    for (var over = overStep; over * 6 <= totalBalls; over += overStep) {
      var idx = balls.findIndex(function (b) { return b.ball_number >= over * 6; });
      if (idx < 0) continue;
      el("text", { x: x(idx), y: H - 8, "text-anchor": "middle", class: "axis-label" },
        node).textContent = over;
    }
    el("text", { x: W - padR, y: H - 8, "text-anchor": "end", class: "axis-label" },
      node).textContent = "overs";

    var playhead = el("line", {
      x1: x(last), x2: x(last), y1: padT, y2: padT + plotH,
      stroke: "var(--ink)", "stroke-width": 1, opacity: 0.55,
    }, node);
    var marker = el("circle", {
      cx: x(last), cy: y(balls[last].probability), r: 4.5,
      fill: "var(--ground)", stroke: "var(--ink)", "stroke-width": 2,
    }, node);

    var index = last;

    function setIndex(next, notify) {
      next = Math.max(0, Math.min(Math.round(next), last));
      if (next === index && notify !== true) return;
      index = next;
      var point = points[index];
      playhead.setAttribute("x1", point[0]);
      playhead.setAttribute("x2", point[0]);
      marker.setAttribute("cx", point[0]);
      marker.setAttribute("cy", point[1]);
      var ahead = balls[index].probability >= 0.5;
      marker.setAttribute("stroke", ahead ? "var(--bat)" : "var(--bowl)");
      if (options.onChange) options.onChange(index, balls[index]);
    }

    function indexFromClientX(clientX) {
      var box = node.getBoundingClientRect();
      var ratio = (clientX - box.left) / box.width;
      var svgX = ratio * W;
      return ((svgX - padL) / plotW) * last;
    }

    var dragging = false;
    function begin(event) {
      dragging = true;
      node.setPointerCapture(event.pointerId);
      setIndex(indexFromClientX(event.clientX));
      if (options.onScrubStart) options.onScrubStart();
    }
    function move(event) {
      if (!dragging) return;
      event.preventDefault();
      setIndex(indexFromClientX(event.clientX));
    }
    function end(event) {
      if (!dragging) return;
      dragging = false;
      try { node.releasePointerCapture(event.pointerId); } catch (err) {}
      if (options.onScrubEnd) options.onScrubEnd(index);
    }

    node.addEventListener("pointerdown", begin);
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", end);
    node.addEventListener("pointercancel", end);
    node.style.cursor = "ew-resize";

    // Keyboard scrubbing, so the signature element is not mouse-only.
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "slider");
    node.setAttribute("aria-label", "Scrub through the chase, ball by ball");
    node.setAttribute("aria-valuemin", "0");
    node.setAttribute("aria-valuemax", String(last));
    node.addEventListener("keydown", function (event) {
      var step = event.shiftKey ? 6 : 1;
      if (event.key === "ArrowRight") { setIndex(index + step); event.preventDefault(); }
      else if (event.key === "ArrowLeft") { setIndex(index - step); event.preventDefault(); }
      else if (event.key === "Home") { setIndex(0); event.preventDefault(); }
      else if (event.key === "End") { setIndex(last); event.preventDefault(); }
    });

    host.appendChild(node);

    return {
      setIndex: setIndex,
      get index() { return index; },
      get last() { return last; },
      node: node,
    };
  }

  /* --------------------------------------------------------------------
     Reliability: predicted against observed. A perfect model sits on the
     diagonal, so the diagonal is drawn and the dots are measured against it.
     -------------------------------------------------------------------- */

  function reliability(host, rows) {
    host.textContent = "";
    if (!rows || !rows.length) return;

    var W = 300, H = 300;
    var pad = 34;
    var plot = W - pad * 2;
    var x = function (p) { return pad + p * plot; };
    var y = function (p) { return H - pad - p * plot; };

    var node = svg(W, H, "Predicted probability against observed win rate");

    el("rect", { x: pad, y: pad, width: plot, height: plot,
      fill: "none", stroke: "var(--rule)", "stroke-width": 1 }, node);
    el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1),
      stroke: "var(--rule-strong)", "stroke-width": 1, "stroke-dasharray": "4 3" }, node);

    var maxCount = Math.max.apply(null, rows.map(function (r) { return r.count; }));
    var linePoints = rows.map(function (r) { return [x(r.predicted), y(r.observed)]; });
    el("path", { d: path(linePoints), fill: "none",
      stroke: "var(--bat)", "stroke-width": 1.5 }, node);

    rows.forEach(function (r) {
      el("circle", {
        cx: x(r.predicted), cy: y(r.observed),
        // Radius carries the sample size, so a lonely bin cannot mislead.
        r: 3 + 4 * Math.sqrt(r.count / maxCount),
        fill: "var(--bat)", opacity: 0.85,
      }, node);
    });

    [0, 0.5, 1].forEach(function (p) {
      el("text", { x: x(p), y: H - pad + 15, "text-anchor": "middle", class: "axis-label" },
        node).textContent = Math.round(p * 100);
      el("text", { x: pad - 7, y: y(p) + 3, "text-anchor": "end", class: "axis-label" },
        node).textContent = Math.round(p * 100);
    });
    el("text", { x: W / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" },
      node).textContent = "predicted %";
    el("text", { x: 11, y: H / 2, "text-anchor": "middle", class: "axis-label",
      transform: "rotate(-90 11 " + H / 2 + ")" }, node).textContent = "observed %";

    host.appendChild(node);
  }

  window.Charts = {
    scoreCurve: scoreCurve,
    wicketBars: wicketBars,
    projectionBand: projectionBand,
    worm: worm,
    reliability: reliability,
  };
})(window, document);
