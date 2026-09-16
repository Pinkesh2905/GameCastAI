/* ==========================================================================
   Google Analytics 4

   HOW TO TURN THIS ON
   -------------------
   1. Go to analytics.google.com and create a property (or open an existing
      one). Admin -> Data streams -> Add stream -> Web. Enter the URL you will
      deploy to and give the stream a name.
   2. The stream page shows a MEASUREMENT ID that looks like G-XXXXXXXXXX.
      Copy it.
   3. Paste it into MEASUREMENT_ID below and deploy. That is the whole setup.
      Nothing else on the page needs changing.

   Until a real ID is filled in, every function here is a no-op: no script is
   loaded, no cookie is set and no request leaves the browser. That keeps local
   development clean and means a fork of this repo does not silently report
   into somebody else's property.

   WHAT GETS MEASURED
   ------------------
   GA4 collects page_view automatically once gtag is loaded. On top of that we
   send a small number of custom events, chosen because each one answers a
   question worth asking about this particular app:

     predict_run       someone scored a chase       -> is the core tool used?
     scenario_click    tried a what-if over         -> do people explore?
     replay_open       opened a real match          -> is the archive valued?
     replay_scrub      dragged the worm             -> is the signature used?
     share_copy        copied a scenario link       -> does it spread?
     view_change       switched section             -> what do people come for?
     league_change     switched competition         -> is multi-league earning its keep?

   Each carries a few parameters. To see them in GA4 as dimensions you can
   filter on, register them once under Admin -> Custom definitions -> Create
   custom dimension, with Scope = Event and the parameter name below.

   PARAMETERS SENT
     league        ipl / t20i / bbl / psl / wpl
     phase         powerplay / middle / death
     probability   rounded to the nearest 5 so it buckets cleanly
     section       chase / innings / replay / model

   PRIVACY
   -------
   No personal data is collected here: no names, no identifiers, no free text.
   The venue and team values are public sporting facts, and the situation
   numbers are the user's own hypothetical. Global Privacy Control and Do Not
   Track are both honoured, and a page served from localhost never reports.
   ========================================================================== */

(function (window, document) {
  "use strict";

  // Replace with your own G-XXXXXXXXXX. Leave blank to keep analytics off.
  var MEASUREMENT_ID = "";

  // Hosts that should never report, so local work stays out of the numbers.
  var EXCLUDED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", ""];

  var enabled = false;

  function optedOut() {
    // Global Privacy Control is the successor signal; Do Not Track is older
    // but still set by plenty of people. Respecting both costs nothing.
    if (navigator.globalPrivacyControl === true) return true;
    var dnt = navigator.doNotTrack || window.doNotTrack || navigator.msDoNotTrack;
    return dnt === "1" || dnt === "yes";
  }

  function shouldLoad() {
    if (!MEASUREMENT_ID || MEASUREMENT_ID.indexOf("G-") !== 0) return false;
    if (EXCLUDED_HOSTS.indexOf(location.hostname) !== -1) return false;
    if (optedOut()) return false;
    return true;
  }

  window.dataLayer = window.dataLayer || [];
  function gtag() { window.dataLayer.push(arguments); }

  function load() {
    var script = document.createElement("script");
    script.async = true;
    script.src = "https://www.googletagmanager.com/gtag/js?id=" + MEASUREMENT_ID;
    document.head.appendChild(script);

    gtag("js", new Date());
    gtag("config", MEASUREMENT_ID, {
      // The app rewrites the URL as the scenario changes, and every one of
      // those would otherwise land as a separate page view. We send page_view
      // ourselves when the section actually changes instead.
      send_page_view: true,
      anonymize_ip: true,
    });
    enabled = true;
  }

  if (shouldLoad()) {
    load();
  }

  /**
   * Send a custom event. Safe to call whether or not analytics is switched on.
   *
   * @param {string} name   one of the event names documented above
   * @param {object} params flat key/value pairs, no personal data
   */
  function track(name, params) {
    if (!enabled) return;
    try {
      gtag("event", name, params || {});
    } catch (err) {
      /* Analytics must never be able to break the page. */
    }
  }

  /** Record a section change as its own page view, so paths show up in GA4. */
  function page(path, title) {
    if (!enabled) return;
    try {
      gtag("event", "page_view", {
        page_path: path,
        page_title: title,
        page_location: location.origin + path,
      });
    } catch (err) {}
  }

  /** Buckets a 0-1 probability into fives, which keeps cardinality sane. */
  function bucket(probability) {
    return Math.round((probability * 100) / 5) * 5;
  }

  window.Analytics = {
    track: track,
    page: page,
    bucket: bucket,
    get enabled() { return enabled; },
  };
})(window, document);
