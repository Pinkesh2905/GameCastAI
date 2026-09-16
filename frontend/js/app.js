/* ==========================================================================
   GameCastAI application.

   Four views over one API. The chase view is the default and carries its
   whole state in the URL, so any situation someone reaches is a link they can
   send to somebody else - which is the only sharing mechanism this app needs
   and the only one that works without an account or a database.
   ========================================================================== */

(function (window, document) {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    league: "ipl",
    leagues: [],
    detail: null,
    view: "chase",
    lastPrediction: null,
    replay: null,
    worm: null,
    player: null,
  };

  /* ── helpers ───────────────────────────────────────────────────────── */

  function oversLabel(balls) {
    return Math.floor(balls / 6) + "." + (balls % 6);
  }

  function pct(value) { return (value * 100).toFixed(1); }

  function deltaClass(value) {
    if (value > 0.005) return "delta-up";
    if (value < -0.005) return "delta-down";
    return "delta-flat";
  }

  function signed(value) {
    var v = value * 100;
    return (v >= 0 ? "+" : "") + v.toFixed(1);
  }

  var toastTimer = null;
  function toast(message) {
    var node = $("toast");
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { node.hidden = true; }, 2600);
  }

  /* Teams come back split into sides that still exist and sides that do not.
     Both belong in the list - an old match needs its old teams - but a folded
     franchise should never be what the page opens on. */
  function fillTeamSelect(select, detail, selected) {
    select.textContent = "";
    var groups = [
      ["Current teams", detail.filter(function (t) { return t.active; })],
      ["Former teams", detail.filter(function (t) { return !t.active; })],
    ];
    groups.forEach(function (pair) {
      if (!pair[1].length) return;
      var group = document.createElement("optgroup");
      group.label = pair[0];
      pair[1].forEach(function (team) {
        var option = document.createElement("option");
        option.value = team.name;
        option.textContent = team.active
          ? team.name
          : team.name + " (to " + team.last_season + ")";
        if (team.name === selected) option.selected = true;
        group.appendChild(option);
      });
      select.appendChild(group);
    });
  }

  function fillSelect(select, values, selected) {
    select.textContent = "";
    values.forEach(function (value) {
      var option = document.createElement("option");
      if (typeof value === "string") {
        option.value = value;
        option.textContent = value;
      } else {
        option.value = value.value;
        option.textContent = value.label;
      }
      if (option.value === selected) option.selected = true;
      select.appendChild(option);
    });
  }

  /* ── URL state ─────────────────────────────────────────────────────── */

  /* The chase form round-trips through the query string. Short keys keep the
     link short enough to paste into a message without it wrapping. */
  var URL_KEYS = {
    l: "league", t: "target", s: "score", b: "balls", w: "wickets",
    o: "totalOvers", bt: "batting", bw: "bowling", v: "venue",
    fr: "formRuns", fw: "formWickets",
  };

  function readUrl() {
    var params = new URLSearchParams(location.search);
    var out = {};
    Object.keys(URL_KEYS).forEach(function (key) {
      if (params.has(key)) out[URL_KEYS[key]] = params.get(key);
    });
    if (params.has("view")) out.view = params.get("view");
    if (params.has("match")) out.match = params.get("match");
    return out;
  }

  function writeUrl(replace) {
    var params = new URLSearchParams();
    if (state.view !== "chase") params.set("view", state.view);

    if (state.view === "chase") {
      params.set("l", state.league);
      params.set("t", $("target").value);
      params.set("s", $("score").value);
      params.set("b", $("balls").value);
      params.set("w", $("wickets").value);
      params.set("o", $("total-overs").value);
      if ($("batting-team").value) params.set("bt", $("batting-team").value);
      if ($("bowling-team").value) params.set("bw", $("bowling-team").value);
      if ($("venue").value) params.set("v", $("venue").value);
      params.set("fr", $("form-runs").value);
      params.set("fw", $("form-wickets").value);
    } else if (state.view === "replay" && state.replay) {
      params.set("l", state.league);
      params.set("match", state.replay.match.match_id);
    } else {
      params.set("l", state.league);
    }

    var next = location.pathname + "?" + params.toString();
    // replaceState while a slider moves, pushState only on real navigation,
    // otherwise the back button has to be pressed sixty times.
    if (replace) history.replaceState(null, "", next);
    else history.pushState(null, "", next);
  }

  /* ── chase view ────────────────────────────────────────────────────── */

  var predictTimer = null;

  function chasePayload() {
    var totalOvers = parseInt($("total-overs").value, 10) || 20;
    var totalBalls = totalOvers * 6;
    var balls = Math.min(parseInt($("balls").value, 10) || 0, totalBalls);

    return {
      league: state.league,
      batting_team: $("batting-team").value,
      bowling_team: $("bowling-team").value,
      venue: $("venue").value,
      target: parseInt($("target").value, 10) || 1,
      score: parseInt($("score").value, 10) || 0,
      balls_bowled: balls,
      wickets_fallen: parseInt($("wickets").value, 10) || 0,
      total_balls: totalBalls,
      runs_last_30: parseInt($("form-runs").value, 10) || 0,
      wickets_last_30: parseInt($("form-wickets").value, 10) || 0,
    };
  }

  function syncChaseBounds() {
    var totalOvers = parseInt($("total-overs").value, 10) || 20;
    var totalBalls = totalOvers * 6;
    var target = parseInt($("target").value, 10) || 1;

    var balls = $("balls");
    balls.max = String(totalBalls);
    if (+balls.value > totalBalls) balls.value = String(totalBalls);

    var score = $("score");
    // A chase that has passed the target is over, so the slider stops one
    // short of it and the readout never has to explain a finished game.
    score.max = String(Math.max(target - 1, 0));
    if (+score.value > +score.max) score.value = score.max;

    $("score-out").textContent = score.value;
    $("balls-out").textContent = oversLabel(+balls.value);
    $("wickets-out").textContent = $("wickets").value;
    $("form-runs-out").textContent = $("form-runs").value;
    $("form-wickets-out").textContent = $("form-wickets").value;
  }

  function renderReadout(data) {
    var probability = data.probability;
    var batPct = pct(probability);
    var bowlPct = pct(1 - probability);

    $("pct").textContent = probability >= 0.5 ? batPct : bowlPct;
    $("readout").classList.toggle("is-bowling", probability < 0.5);
    $("readout-team").textContent = probability >= 0.5 ? data.batting_team : data.bowling_team;
    $("readout-note").textContent = probability >= 0.5
      ? "to chase this down"
      : "to defend this total";

    $("split-bat").style.transform = "scaleX(" + probability.toFixed(4) + ")";
    $("split-bowl").style.transform = "scaleX(" + (1 - probability).toFixed(4) + ")";
    $("legend-bat-pct").textContent = batPct + "%";
    $("legend-bowl-pct").textContent = bowlPct + "%";
    $("legend-bat-name").textContent = data.batting_team;
    $("legend-bowl-name").textContent = data.bowling_team;
    $("split-figure").setAttribute(
      "aria-label",
      data.batting_team + " " + batPct + " percent, " + data.bowling_team + " " + bowlPct + " percent"
    );

    var confidence = $("confidence");
    if (data.decided) {
      confidence.innerHTML = "<b>Settled</b> - this match is already decided.";
    } else {
      confidence.innerHTML = "Reading is <b>" + data.confidence + "</b>" +
        (data.venue_par ? " &middot; par at this ground is about " + Math.round(data.venue_par) : "");
    }
  }

  function renderScoreboard(data) {
    var s = data.situation;
    var cells = [
      ["Score", s.score_line],
      ["Overs", s.overs_label],
      ["Need", s.runs_left],
      ["Balls left", s.balls_left],
      ["Run rate", s.current_run_rate.toFixed(2)],
      ["Required", s.required_run_rate === null ? "-" : s.required_run_rate.toFixed(2)],
    ];
    var host = $("scoreboard");
    host.textContent = "";
    cells.forEach(function (cell) {
      var wrap = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = cell[0];
      var dd = document.createElement("dd");
      dd.textContent = cell[1];
      wrap.appendChild(dt);
      wrap.appendChild(dd);
      host.appendChild(wrap);
    });

    var narrative = $("narrative");
    narrative.textContent = "";
    data.narrative.forEach(function (line) {
      var p = document.createElement("p");
      p.textContent = line;
      narrative.appendChild(p);
    });
  }

  function renderScenarios(data) {
    var host = $("scenarios");
    host.textContent = "";

    if (!data.scenarios.length) {
      var li = document.createElement("li");
      li.className = "scenario";
      li.innerHTML = '<span class="scenario-label">Nothing left to simulate - the match is decided.</span>';
      host.appendChild(li);
      return;
    }

    data.scenarios.forEach(function (scenario) {
      var li = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.className = "scenario";
      button.innerHTML =
        '<span class="scenario-label">' + scenario.label + "</span>" +
        '<span class="scenario-prob">' + pct(scenario.probability) + "%</span>" +
        '<span class="scenario-delta ' + deltaClass(scenario.delta) + '">' +
        signed(scenario.delta) + "</span>";

      button.addEventListener("click", function () {
        // Play the over out for real: move the inputs and re-score.
        var totalBalls = (parseInt($("total-overs").value, 10) || 20) * 6;
        $("score").value = String(Math.min(
          +$("score").value + scenario.runs, +$("score").max
        ));
        $("balls").value = String(Math.min(+$("balls").value + 6, totalBalls));
        $("wickets").value = String(Math.min(+$("wickets").value + scenario.wickets, 10));
        syncChaseBounds();
        runPrediction(false);
        window.Analytics.track("scenario_click", {
          league: state.league,
          scenario: scenario.label,
          probability: window.Analytics.bucket(scenario.probability),
        });
      });

      li.appendChild(button);
      host.appendChild(li);
    });
  }

  function renderMatrix(rows) {
    var body = $("matrix").querySelector("tbody");
    body.textContent = "";
    if (!rows.length) {
      var tr = document.createElement("tr");
      tr.innerHTML = '<td colspan="3">Nothing left to break down.</td>';
      body.appendChild(tr);
      return;
    }
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + row.over + "</td>" +
        '<td class="need">' + row.required_this_over.toFixed(1) + "</td>" +
        "<td>" + row.overs_after + "</td>";
      body.appendChild(tr);
    });
  }

  function showChaseError(message) {
    var narrative = $("narrative");
    narrative.textContent = "";
    var p = document.createElement("p");
    p.className = "error-note";
    p.textContent = message;
    narrative.appendChild(p);
  }

  async function runPrediction(replaceUrl) {
    syncChaseBounds();
    var payload = chasePayload();
    var output = document.querySelector(".col-output");
    output.classList.add("is-loading");

    try {
      var data = await window.Api.predict(payload);
      state.lastPrediction = data;

      renderReadout(data);
      renderScoreboard(data);
      renderScenarios(data);
      renderMatrix(data.required_matrix);

      window.Charts.scoreCurve($("curve-chart"), data.score_curve, {
        score: payload.score,
        probability: data.probability,
      });
      window.Charts.wicketBars($("wickets-chart"), data.wickets_curve,
        data.situation.wickets_left);

      writeUrl(replaceUrl !== false);

      window.Analytics.track("predict_run", {
        league: state.league,
        phase: data.situation.phase,
        probability: window.Analytics.bucket(data.probability),
      });
    } catch (err) {
      if (err.name === "AbortError") return;    // a newer request won
      showChaseError(err.message);
    } finally {
      output.classList.remove("is-loading");
    }
  }

  function schedulePrediction() {
    syncChaseBounds();
    clearTimeout(predictTimer);
    // Long enough to coalesce a slider drag, short enough to feel live.
    predictTimer = setTimeout(function () { runPrediction(true); }, 120);
  }

  /* ── first innings view ────────────────────────────────────────────── */

  var projectTimer = null;

  async function runProjection() {
    var balls = parseInt($("i-balls").value, 10) || 0;
    $("i-score-out").textContent = $("i-score").value;
    $("i-balls-out").textContent = oversLabel(balls);
    $("i-wickets-out").textContent = $("i-wickets").value;
    $("i-form-runs-out").textContent = $("i-form-runs").value;
    $("i-form-wickets-out").textContent = $("i-form-wickets").value;

    try {
      var data = await window.Api.project({
        league: state.league,
        batting_team: $("i-batting").value,
        bowling_team: $("i-bowling").value,
        venue: $("i-venue").value,
        score: parseInt($("i-score").value, 10) || 0,
        balls_bowled: balls,
        wickets_fallen: parseInt($("i-wickets").value, 10) || 0,
        total_balls: 120,
        runs_last_30: parseInt($("i-form-runs").value, 10) || 0,
        wickets_last_30: parseInt($("i-form-wickets").value, 10) || 0,
      });

      $("proj-number").textContent = data.projection.projected;
      $("proj-band").textContent = data.projection.low + " to " + data.projection.high;

      window.Charts.projectionBand($("proj-chart"), data.projection, data.venue_par,
        parseInt($("i-score").value, 10) || 0);

      var over = data.versus_par;
      var coverage = Math.round((data.projection.confidence || 0.8) * 100);
      $("proj-note").textContent =
        (over >= 0
          ? "That is " + Math.abs(over).toFixed(0) + " above par for this ground"
          : "That is " + Math.abs(over).toFixed(0) + " below par for this ground") +
        ". The range holds the real total about " + coverage +
        "% of the time on matches the model never saw.";
    } catch (err) {
      if (err.name === "AbortError") return;
      $("proj-note").textContent = err.message;
    }
  }

  function scheduleProjection() {
    clearTimeout(projectTimer);
    projectTimer = setTimeout(runProjection, 120);
  }

  /* ── replay view ───────────────────────────────────────────────────── */

  var searchTimer = null;

  async function loadMatches() {
    var host = $("match-list");
    host.innerHTML = '<li style="padding:14px"><span class="skeleton" style="display:block;width:70%"></span></li>';

    try {
      var data = await window.Api.matches({
        league: state.league,
        search: $("match-search").value,
        season: $("match-season").value,
        limit: 60,
      });

      if ($("match-season").options.length <= 1 && data.seasons.length) {
        fillSelect($("match-season"),
          [{ value: "", label: "All" }].concat(
            data.seasons.map(function (s) { return { value: s, label: s }; })
          ), "");
      }

      $("match-count").textContent = data.total
        ? data.total.toLocaleString() + " matches" + (data.total > 60 ? ", showing the 60 most recent" : "")
        : "";

      host.textContent = "";

      if (!data.matches.length) {
        var li = document.createElement("li");
        li.innerHTML =
          '<div class="empty-state" style="border:0;padding:28px 16px">' +
          '<p class="empty-title">No matches found</p>' +
          '<p class="empty-note">Nothing matches that search. Try a team name, a ground, or clear the filter.</p>' +
          "</div>";
        host.appendChild(li);
        return;
      }

      data.matches.forEach(function (match) {
        var li = document.createElement("li");
        var button = document.createElement("button");
        button.type = "button";
        button.className = "match-item";
        button.dataset.matchId = match.match_id;
        button.innerHTML =
          '<span class="match-teams">' + match.first_team + " v " + match.second_team + "</span>" +
          '<span class="match-line">' + (match.first_innings || "") + "  chased  " +
          (match.second_innings || "") + "</span>" +
          '<span class="match-sub">' + match.date + " &middot; " + match.venue + "</span>";
        button.addEventListener("click", function () { openReplay(match.match_id); });
        li.appendChild(button);
        host.appendChild(li);
      });
    } catch (err) {
      host.innerHTML = '<li style="padding:14px"><p class="error-note">' + err.message + "</p></li>";
    }
  }

  function stopPlayer() {
    if (state.player) {
      clearInterval(state.player);
      state.player = null;
    }
    var button = document.querySelector(".play-btn");
    if (button) button.setAttribute("aria-label", "Play the chase");
    var icon = document.querySelector(".play-btn svg");
    if (icon) icon.innerHTML = '<path d="M2 1 L13 7.5 L2 14 Z" fill="currentColor"/>';
  }

  function renderReplay(data) {
    var stage = $("replay-stage");
    stage.textContent = "";

    var head = document.createElement("section");
    head.className = "panel";
    head.innerHTML =
      '<div class="scrub-head">' +
      '<span class="scrub-score" id="scrub-score"></span>' +
      '<span class="scrub-prob" id="scrub-prob"></span>' +
      "</div>" +
      '<p class="scrub-detail" id="scrub-detail"></p>' +
      '<figure class="chart" id="worm-chart"></figure>' +
      '<div class="playbar">' +
      '<button class="play-btn" type="button" aria-label="Play the chase">' +
      '<svg viewBox="0 0 15 15" aria-hidden="true"><path d="M2 1 L13 7.5 L2 14 Z" fill="currentColor"/></svg>' +
      "</button>" +
      '<p class="scrub-detail" style="flex:1">' +
      data.batting_team + " chasing " + data.target + " against " + data.bowling_team +
      " &middot; " + data.match.venue + ", " + data.match.date +
      "</p>" +
      "</div>";
    stage.appendChild(head);

    var moments = document.createElement("section");
    moments.className = "panel";
    moments.innerHTML = '<div class="panel-head"><h2 class="panel-title">Where it turned</h2>' +
      '<p class="panel-sub">The deliveries that moved the number most. Tap one to jump there.</p></div>' +
      '<ul class="moments" id="moments"></ul>';
    stage.appendChild(moments);

    var scoreEl = $("scrub-score");
    var probEl = $("scrub-prob");
    var detailEl = $("scrub-detail");

    function paint(index, ball) {
      scoreEl.textContent = ball.score_line + "  (" + ball.over_label + ")";
      probEl.textContent = pct(ball.probability) + "%";
      probEl.style.color = ball.probability >= 0.5 ? "var(--bat)" : "var(--bowl)";

      var parts = [];
      if (ball.start) {
        parts.push("Start of the chase, " + data.target + " to win.");
      } else {
        if (ball.wicket) parts.push("WICKET - " + ball.player_out + ", " + ball.wicket_kind + ".");
        else if (ball.batter) parts.push(ball.bowler + " to " + ball.batter + ", " + ball.runs + ".");
        parts.push(ball.runs_left + " needed from " + ball.balls_left + ".");
      }
      detailEl.textContent = parts.join("  ");
      state.worm && state.worm.node.setAttribute("aria-valuenow", String(index));
      state.worm && state.worm.node.setAttribute("aria-valuetext",
        ball.over_label + ", " + ball.score_line + ", " + pct(ball.probability) + " percent");
    }

    state.worm = window.Charts.worm($("worm-chart"), data.balls, {
      totalBalls: data.total_balls,
      onChange: paint,
      onScrubStart: stopPlayer,
      onScrubEnd: function () {
        window.Analytics.track("replay_scrub", { league: state.league });
      },
    });

    var momentHost = $("moments");
    data.key_moments.forEach(function (moment) {
      var li = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.className = "moment";
      button.innerHTML =
        '<span class="moment-over">' + moment.over_label + "</span>" +
        '<span class="moment-text">' + moment.headline + "</span>" +
        '<span class="moment-delta ' + deltaClass(moment.delta) + '">' + signed(moment.delta) + "</span>";
      button.addEventListener("click", function () {
        stopPlayer();
        var index = data.balls.findIndex(function (b) {
          return b.ball_number === moment.ball_number && b.over_label === moment.over_label;
        });
        if (index >= 0) state.worm.setIndex(index, true);
        state.worm.node.focus();
      });
      li.appendChild(button);
      momentHost.appendChild(li);
    });

    document.querySelector(".play-btn").addEventListener("click", function () {
      var button = this;
      if (state.player) { stopPlayer(); return; }
      if (state.worm.index >= state.worm.last) state.worm.setIndex(0, true);

      button.querySelector("svg").innerHTML =
        '<rect x="2.5" y="1.5" width="3.5" height="12" fill="currentColor"/>' +
        '<rect x="9" y="1.5" width="3.5" height="12" fill="currentColor"/>';
      button.setAttribute("aria-label", "Pause");

      state.player = setInterval(function () {
        if (state.worm.index >= state.worm.last) { stopPlayer(); return; }
        state.worm.setIndex(state.worm.index + 1, true);
      }, 110);
    });

    state.worm.setIndex(data.balls.length - 1, true);
  }

  async function openReplay(matchId) {
    stopPlayer();
    document.querySelectorAll(".match-item").forEach(function (node) {
      node.classList.toggle("is-selected", node.dataset.matchId === String(matchId));
    });

    var stage = $("replay-stage");
    stage.innerHTML = '<div class="panel"><p class="scrub-detail">Scoring every delivery…</p>' +
      '<span class="skeleton" style="display:block;height:210px;margin-top:14px"></span></div>';

    try {
      var data = await window.Api.replay(matchId, state.league);
      state.replay = data;
      renderReplay(data);
      writeUrl(false);

      // On a phone the list sits above the chart, so opening a match has to
      // carry the user to the thing they just asked for.
      if (window.matchMedia("(max-width: 979px)").matches) {
        stage.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      window.Analytics.track("replay_open", { league: state.league, match_id: String(matchId) });
    } catch (err) {
      stage.innerHTML = '<div class="panel"><p class="error-note">' + err.message + "</p></div>";
    }
  }

  /* ── model card ────────────────────────────────────────────────────── */

  function metric(value, label, note) {
    return '<div class="metric"><span class="metric-value">' + value + "</span>" +
      '<span class="metric-label">' + label + "</span>" +
      (note ? '<span class="metric-note">' + note + "</span>" : "") + "</div>";
  }

  function renderModelCard(detail) {
    var card = detail.model_card;
    var overall = card.overall;
    var host = $("model-card");

    var phaseRows = Object.keys(card.by_phase).map(function (phase) {
      var p = card.by_phase[phase];
      return "<tr><td>" + phase + "</td><td>" + p.rows.toLocaleString() + "</td><td>" +
        p.roc_auc.toFixed(3) + "</td><td>" + (p.accuracy * 100).toFixed(1) + "%</td><td>" +
        p.brier.toFixed(3) + "</td></tr>";
    }).join("");

    var maxImportance = Math.max.apply(null,
      card.importances.map(function (f) { return f.importance; }));
    var importanceRows = card.importances.slice(0, 8).map(function (f) {
      var width = Math.max((f.importance / maxImportance) * 100, 0);
      return '<li class="bar-row"><span class="bar-name">' + f.label + "</span>" +
        '<span class="bar-track"><span class="bar-fill" style="width:' + width.toFixed(1) + '%"></span></span>' +
        '<span class="bar-value">' + f.importance.toFixed(3) + "</span></li>";
    }).join("");

    var setting = card.setting;

    host.innerHTML =
      '<div class="metric-grid">' +
      metric((overall.accuracy * 100).toFixed(1) + "%", "Accuracy", "on unseen matches") +
      metric(overall.roc_auc.toFixed(3), "ROC-AUC", "1.0 is perfect") +
      metric(overall.brier.toFixed(3), "Brier score", "lower is better") +
      metric(card.matches_total.toLocaleString(), "Matches", card.date_from.slice(0, 4) + "-" + card.date_to.slice(0, 4)) +
      "</div>" +

      '<section class="model-section">' +
      '<h2 class="panel-title">How it was tested</h2>' +
      '<div class="model-prose">' +
      "<p>The archive was split by <strong>date</strong>, not at random. The model trained on " +
      card.train_matches.toLocaleString() + " matches and was measured on the " +
      card.test_matches.toLocaleString() + " most recent ones it had never seen. " +
      "A random split would have put deliveries from the same game on both sides - consecutive balls are near-duplicates, " +
      "and the score that comes back from that is not real.</p>" +
      "<p>The model that actually answers your requests was then refitted on all " +
      (card.shipped_matches || card.matches_total).toLocaleString() +
      " matches, because there is no reason to serve predictions from a model denied the most recent seasons. " +
      "The numbers above describe the method; the shipped model has seen more.</p>" +
      "<p>Algorithm: <strong>" + card.algorithm + "</strong>" +
      (card.calibrated ? " with isotonic calibration" : "") +
      ". Team identity was tested as an input and <strong>removed</strong> - it made out-of-sample accuracy worse. " +
      "A T20 squad is rebuilt every auction, so the name is not a stable thing to learn from. The ground is, and it stays in.</p>" +
      "</div></section>" +

      '<section class="model-section">' +
      '<h2 class="panel-title">Is 70% really 70%?</h2>' +
      '<div class="model-prose"><p>Ranking matches correctly is not enough: if the page shows 70%, those situations ' +
      "should win about seven times in ten. Each dot is a probability band, sized by how many deliveries fell in it. " +
      "The dashed line is perfection.</p>" +
      (card.holdout_bias > 0.02
        ? "<p>The dots sit <strong>above</strong> the line, by " + (card.holdout_bias * 100).toFixed(0) +
          " points on average. That is real and worth knowing: on the most recent seasons, chasing sides won more often " +
          "than the older data says they should. Random splits show no such bias, so this is drift rather than a bug - " +
          "teams now chase down positions that used to be losing ones. Retraining as new matches land is the fix.</p>"
        : "") +
      "</div>" +
      '<figure class="chart" id="reliability-chart" style="max-width:320px;margin-top:14px"></figure>' +
      "</section>" +

      '<section class="model-section">' +
      '<h2 class="panel-title">Where it is confident, and where it is not</h2>' +
      '<div class="table-scroll"><table class="data-table">' +
      "<thead><tr><th>Phase</th><th>Deliveries</th><th>ROC-AUC</th><th>Accuracy</th><th>Brier</th></tr></thead>" +
      "<tbody>" + phaseRows + "</tbody></table></div>" +
      '<div class="model-prose" style="margin-top:12px"><p>The powerplay is genuinely hard and the death overs are nearly ' +
      "arithmetic. A single headline accuracy hides that, which is why it is broken out.</p></div>" +
      "</section>" +

      '<section class="model-section">' +
      '<h2 class="panel-title">What the model leans on</h2>' +
      '<ul class="bar-list">' + importanceRows + "</ul>" +
      '<div class="model-prose" style="margin-top:12px"><p>Measured by permutation: each feature is shuffled and the drop ' +
      "in Brier score recorded. It answers what the model actually uses, not what it was given.</p></div>" +
      "</section>" +

      (setting ? '<section class="model-section">' +
        '<h2 class="panel-title">First-innings projection</h2>' +
        '<div class="metric-grid">' +
        metric(setting.mae.toFixed(1), "Mean error", "runs, averaged over the innings") +
        metric((setting.band_coverage * 100).toFixed(0) + "%", "Range holds", "against " + (setting.band_nominal * 100).toFixed(0) + "% aimed for") +
        metric(Math.round(setting.band_mean_width), "Range width", "runs") +
        "</div>" +
        '<div class="model-prose"><p>The range comes from conformal prediction - the width is read off errors the model ' +
        "made on matches it had not trained on, so the coverage is earned rather than asserted. Quantile regression was " +
        "tried first and claimed an 85% band that held the truth only a quarter of the time.</p></div>" +
        "</section>" : "") +

      '<section class="model-section">' +
      '<h2 class="panel-title">Data</h2>' +
      '<div class="model-prose">' +
      "<p>" + card.matches_total.toLocaleString() + " matches from " + card.date_from + " to " + card.date_to +
      ", across " + card.seasons + " seasons and " + card.venues + " grounds. Source: " + card.data_source +
      ". Rain-revised (D/L) matches are excluded from training, because the archive stores only the final revised " +
      "target and earlier deliveries would carry a target that was not yet in force. Super overs are excluded too.</p>" +
      "<p>Chasing sides win " + (card.base_rate * 100).toFixed(1) + "% of these matches. Any model worth using has to beat that.</p>" +
      "</div></section>";

    window.Charts.reliability($("reliability-chart"), card.reliability);
  }

  /* ── views ─────────────────────────────────────────────────────────── */

  var VIEW_TITLES = {
    chase: "Chase win probability",
    innings: "First-innings projection",
    replay: "Match replay",
    model: "Model card",
  };

  function setView(view, skipUrl) {
    if (!VIEW_TITLES[view]) view = "chase";
    state.view = view;
    stopPlayer();

    document.querySelectorAll(".tab").forEach(function (tab) {
      var current = tab.dataset.view === view;
      tab.classList.toggle("is-current", current);
      if (current) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    });
    document.querySelectorAll(".view").forEach(function (node) {
      node.hidden = node.id !== "view-" + view;
    });

    if (view === "innings") runProjection();
    if (view === "replay" && !$("match-list").children.length) loadMatches();
    if (view === "model" && state.detail) renderModelCard(state.detail);

    if (!skipUrl) writeUrl(false);
    window.Analytics.page("/" + view, VIEW_TITLES[view]);
    window.Analytics.track("view_change", { section: view, league: state.league });
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  /* ── league ────────────────────────────────────────────────────────── */

  async function applyLeague(code, initial) {
    state.league = code;
    $("league").value = code;

    var detail = await window.Api.league(code);
    state.detail = detail;

    var teamDetail = detail.team_detail || detail.teams.map(function (name) {
      return { name: name, active: true, matches: 0, last_season: "" };
    });
    var active = teamDetail.filter(function (t) { return t.active; });
    var defaults = active.length >= 2 ? active : teamDetail;

    var venues = [{ value: "", label: "Any ground (league average)" }].concat(
      detail.venues.slice(0, 60).map(function (v) {
        return { value: v.venue, label: v.venue + " (" + v.matches + ")" };
      })
    );

    var keep = initial || {};
    fillTeamSelect($("batting-team"), teamDetail, keep.batting || defaults[0].name);
    fillTeamSelect($("bowling-team"), teamDetail, keep.bowling || defaults[1].name);
    fillSelect($("venue"), venues, keep.venue || "");
    fillTeamSelect($("i-batting"), teamDetail, defaults[0].name);
    fillTeamSelect($("i-bowling"), teamDetail, defaults[1].name);
    fillSelect($("i-venue"), venues, "");

    $("venue-hint").textContent = detail.league_par
      ? "league par " + Math.round(detail.league_par)
      : "";

    // A new league means a new archive and a new season list.
    $("match-season").innerHTML = '<option value="">All</option>';
    $("match-list").textContent = "";
    state.replay = null;

    if (state.view === "model") renderModelCard(detail);
    if (state.view === "replay") loadMatches();
  }

  /* ── wiring ────────────────────────────────────────────────────────── */

  function bind() {
    document.querySelectorAll(".tab").forEach(function (tab) {
      tab.addEventListener("click", function () { setView(tab.dataset.view); });
    });

    ["score", "balls", "wickets", "form-runs", "form-wickets"].forEach(function (id) {
      $(id).addEventListener("input", schedulePrediction);
    });
    ["target", "total-overs", "batting-team", "bowling-team", "venue"].forEach(function (id) {
      $(id).addEventListener("change", function () { runPrediction(true); });
    });
    $("target").addEventListener("input", syncChaseBounds);

    ["i-score", "i-balls", "i-wickets", "i-form-runs", "i-form-wickets"].forEach(function (id) {
      $(id).addEventListener("input", scheduleProjection);
    });
    ["i-batting", "i-bowling", "i-venue"].forEach(function (id) {
      $(id).addEventListener("change", runProjection);
    });

    $("league").addEventListener("change", async function () {
      await applyLeague(this.value);
      if (state.view === "chase") runPrediction(false);
      if (state.view === "innings") runProjection();
      window.Analytics.track("league_change", { league: this.value });
    });

    $("match-search").addEventListener("input", function () {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(loadMatches, 220);
    });
    $("match-season").addEventListener("change", loadMatches);

    $("copy-link").addEventListener("click", async function () {
      var link = location.href;
      try {
        await navigator.clipboard.writeText(link);
        toast("Link copied - anyone who opens it lands on this exact situation.");
      } catch (err) {
        // Clipboard is blocked without a secure context or a user gesture on
        // some browsers, so fall back to selecting it for a manual copy.
        window.prompt("Copy this link", link);
      }
      window.Analytics.track("share_copy", { league: state.league, section: state.view });
    });

    window.addEventListener("popstate", function () { boot(true); });

    /* Charts choose their proportions from the width they are handed, so a
       rotation or a resized window has to redraw them. Only a crossing of the
       narrow breakpoint matters, which keeps this off the resize hot path. */
    var wasNarrow = window.innerWidth < 520;
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        var narrow = window.innerWidth < 520;
        if (narrow === wasNarrow) return;
        wasNarrow = narrow;

        if (state.view === "chase" && state.lastPrediction) runPrediction(true);
        if (state.view === "innings") runProjection();
        if (state.view === "replay" && state.replay) {
          var keep = state.worm ? state.worm.index : null;
          renderReplay(state.replay);
          if (keep !== null) state.worm.setIndex(keep, true);
        }
      }, 180);
    });
  }

  /* ── boot ──────────────────────────────────────────────────────────── */

  async function boot(fromHistory) {
    var url = readUrl();

    if (!state.leagues.length) {
      try {
        var payload = await window.Api.leagues();
        state.leagues = payload.leagues;
        fillSelect($("league"), payload.leagues.map(function (l) {
          return { value: l.code, label: l.short + " - " + l.matches.toLocaleString() + " matches" };
        }), url.league || payload.default);
      } catch (err) {
        document.querySelector("main").innerHTML =
          '<p class="error-note">' + err.message +
          " The API may not be running - start it with <code>uvicorn backend.app.main:app</code>.</p>";
        return;
      }
    }

    var code = url.league && state.leagues.some(function (l) { return l.code === url.league; })
      ? url.league
      : $("league").value;

    await applyLeague(code, url);

    if (url.target) $("target").value = url.target;
    if (url.totalOvers) $("total-overs").value = url.totalOvers;
    syncChaseBounds();
    if (url.score) $("score").value = url.score;
    if (url.balls) $("balls").value = url.balls;
    if (url.wickets) $("wickets").value = url.wickets;
    if (url.formRuns) $("form-runs").value = url.formRuns;
    if (url.formWickets) $("form-wickets").value = url.formWickets;

    setView(url.view || "chase", true);
    await runPrediction(true);

    if (url.view === "replay" && url.match) openReplay(url.match);
    if (!fromHistory) bind();
  }

  document.addEventListener("DOMContentLoaded", function () { boot(false); });
})(window, document);
