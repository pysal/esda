(function () {
  "use strict";

  var indexPromise = null;
  var searchIndex = null;

  function getBaseUrl() {
    var config = document.getElementById("__config");
    if (!config) {
      return ".";
    }
    try {
      return JSON.parse(config.textContent).base || ".";
    } catch (_error) {
      return ".";
    }
  }

  function loadSearchIndex() {
    if (searchIndex) {
      return Promise.resolve(searchIndex);
    }
    if (indexPromise) {
      return indexPromise;
    }

    indexPromise = new Promise(function (resolve, reject) {
      var previousSearch = window.Search || {};
      window.Search = Object.assign({}, previousSearch, {
        setIndex: function (index) {
          searchIndex = index;
          if (typeof previousSearch.setIndex === "function") {
            previousSearch.setIndex(index);
          }
          resolve(index);
        },
      });

      var script = document.createElement("script");
      script.src = getBaseUrl() + "/searchindex.js";
      script.async = true;
      script.onerror = reject;
      document.body.appendChild(script);
    });

    return indexPromise;
  }

  function asArray(value) {
    if (typeof value === "undefined") {
      return [];
    }
    return Array.isArray(value) ? value : [value];
  }

  function docUrl(index, position) {
    if (index.docurls) {
      return index.docurls[position];
    }
    if (index.docnames) {
      return index.docnames[position] + ".html";
    }
    return "#";
  }

  function searchDocuments(query, index) {
    var terms = query
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    var scores = new Map();

    if (!terms.length) {
      return [];
    }

    terms.forEach(function (term) {
      var escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      var matcher = new RegExp(escaped);

      Object.keys(index.titleterms || {}).forEach(function (key) {
        if (key === term || key.match(matcher)) {
          asArray(index.titleterms[key]).forEach(function (doc) {
            scores.set(doc, (scores.get(doc) || 0) + 10);
          });
        }
      });

      Object.keys(index.terms || {}).forEach(function (key) {
        if (key === term || key.match(matcher)) {
          asArray(index.terms[key]).forEach(function (doc) {
            scores.set(doc, (scores.get(doc) || 0) + 3);
          });
        }
      });

      (index.titles || []).forEach(function (title, doc) {
        if (title.toLowerCase().indexOf(term) !== -1) {
          scores.set(doc, (scores.get(doc) || 0) + 20);
        }
      });
    });

    return Array.from(scores.entries())
      .filter(function (entry) {
        var doc = entry[0];
        var haystack = ((index.titles && index.titles[doc]) || "").toLowerCase();
        return terms.every(function (term) {
          return (
            haystack.indexOf(term) !== -1 ||
            asArray(index.terms && index.terms[term]).indexOf(doc) !== -1 ||
            asArray(index.titleterms && index.titleterms[term]).indexOf(doc) !== -1
          );
        });
      })
      .sort(function (left, right) {
        return right[1] - left[1];
      })
      .slice(0, 30)
      .map(function (entry) {
        var doc = entry[0];
        return {
          title: (index.titles && index.titles[doc]) || docUrl(index, doc),
          url: docUrl(index, doc),
          score: entry[1],
        };
      });
  }

  function render(container, query, results) {
    var meta = container.querySelector(".md-search-result__meta");
    var list = container.querySelector(".md-search-result__list");
    if (!meta || !list) {
      return;
    }

    list.innerHTML = "";
    if (!query.trim()) {
      meta.textContent = "Type to start searching";
      return;
    }
    if (!results.length) {
      meta.textContent = "No matching documents";
      return;
    }

    meta.textContent =
      results.length === 1 ? "1 matching document" : results.length + " matching documents";

    results.forEach(function (result) {
      var item = document.createElement("li");
      item.className = "md-search-result__item";

      var link = document.createElement("a");
      link.className = "md-search-result__link";
      link.href = result.url;

      var article = document.createElement("article");
      article.className = "md-search-result__article md-typeset";

      var title = document.createElement("h1");
      title.textContent = result.title;

      var location = document.createElement("p");
      location.textContent = result.url;

      article.appendChild(title);
      article.appendChild(location);
      link.appendChild(article);
      item.appendChild(link);
      list.appendChild(item);
    });
  }

  function initSearchFallback() {
    var search = document.querySelector('[data-md-component="search"]');
    if (!search) {
      return;
    }
    var input = search.querySelector('[data-md-component="search-query"]');
    var result = search.querySelector('[data-md-component="search-result"]');
    if (!input || !result) {
      return;
    }

    var update = function () {
      var query = input.value;
      if (!query.trim()) {
        render(result, query, []);
        return;
      }
      loadSearchIndex().then(function (index) {
        render(result, query, searchDocuments(query, index));
      });
    };

    input.addEventListener("input", update);
    input.addEventListener("keyup", update);
    if (input.value) {
      update();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSearchFallback);
  } else {
    initSearchFallback();
  }
})();
