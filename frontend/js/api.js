/* ==========================================================================
   API client.

   Thin wrapper over fetch with three things the naive version lacks: it
   aborts a request that a newer one has superseded (dragging a slider fires
   many), it surfaces the server's own error message rather than a generic
   one, and it caches the responses that never change within a session.
   ========================================================================== */

(function (window) {
  "use strict";

  var cache = new Map();
  var inflight = new Map();

  function url(path, params) {
    var query = new URLSearchParams();
    Object.keys(params || {}).forEach(function (key) {
      var value = params[key];
      if (value !== undefined && value !== null && value !== "") {
        query.set(key, value);
      }
    });
    var qs = query.toString();
    return qs ? path + "?" + qs : path;
  }

  function describe(response, body) {
    if (body && body.detail) return body.detail;
    if (response.status === 404) return "Not found.";
    if (response.status >= 500) return "The server had a problem. Try again in a moment.";
    return "Request failed (" + response.status + ").";
  }

  async function request(path, options) {
    options = options || {};
    var response;
    try {
      response = await fetch(path, options);
    } catch (err) {
      if (err.name === "AbortError") throw err;
      // Distinguish "offline" from "the server said no", because the fix
      // is different and the user can act on the difference.
      var offline = new Error("Cannot reach the server. Check your connection.");
      offline.offline = true;
      throw offline;
    }

    var body = null;
    try {
      body = await response.json();
    } catch (err) {
      body = null;
    }

    if (!response.ok) {
      var failure = new Error(describe(response, body));
      failure.status = response.status;
      failure.body = body;
      throw failure;
    }
    return body;
  }

  /** GET that caches for the life of the page. For data that cannot change. */
  async function getCached(path, params) {
    var full = url(path, params);
    if (cache.has(full)) return cache.get(full);
    if (inflight.has(full)) return inflight.get(full);

    var promise = request(full).then(function (value) {
      cache.set(full, value);
      inflight.delete(full);
      return value;
    }).catch(function (err) {
      inflight.delete(full);
      throw err;
    });

    inflight.set(full, promise);
    return promise;
  }

  function get(path, params) {
    return request(url(path, params));
  }

  /**
   * POST that cancels any earlier call sharing the same key. A slider drag
   * produces a burst of requests and only the last answer is wanted; without
   * this they can also arrive out of order and paint a stale number.
   */
  var controllers = new Map();
  function post(path, payload, key) {
    if (key && controllers.has(key)) {
      controllers.get(key).abort();
    }
    var controller = new AbortController();
    if (key) controllers.set(key, controller);

    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    }).finally(function () {
      if (key && controllers.get(key) === controller) controllers.delete(key);
    });
  }

  window.Api = {
    leagues: function () { return getCached("/api/leagues"); },
    league: function (code) { return getCached("/api/leagues/" + encodeURIComponent(code)); },
    predict: function (payload) { return post("/api/predict", payload, "predict"); },
    project: function (payload) { return post("/api/project", payload, "project"); },
    matches: function (params) { return get("/api/matches", params); },
    replay: function (id, league) {
      return getCached("/api/matches/" + encodeURIComponent(id) + "/replay", { league: league });
    },
  };
})(window);
