var LR_PERIOD      = 120;
var LR_DATA_COUNT  = 155;
var LR_COL         = 51;      // AY열 (1-based)
var LR_PROP_PREFIX = "LRT_";
var LR_CACHE_SEC   = 86400;   // 24시간
var LR_SLEEP_MS    = 400;

function _calcLR(lows, period) {
  if (!lows || lows.length < period) return null;

  var y     = lows.slice(-period);
  var n     = period;
  var xMean = (n - 1) / 2;
  var ySum  = 0;
  for (var i = 0; i < n; i++) ySum += y[i];
  var yMean = ySum / n;

  var num = 0, den = 0;
  for (var i = 0; i < n; i++) {
    num += (i - xMean) * (y[i] - yMean);
    den += (i - xMean) * (i - xMean);
  }
  if (den === 0) return null;

  var slope     = num / den;
  var intercept = yMean - slope * xMean;
  var value     = intercept + slope * (n - 1);

  return {
    value: Math.round(value * 100) / 100,
    slope: Math.round(slope * 1000000) / 1000000
  };
}

function _saveLRProp(code, result) {
  if (!result || typeof result.value !== "number") return;
  try {
    PropertiesService.getScriptProperties().setProperty(
      LR_PROP_PREFIX + code,
      JSON.stringify({ value: result.value, slope: result.slope, ts: Date.now() })
    );
  } catch (e) {
    console.log("[LR캐시 저장 오류] " + code + ": " + e.toString());
  }
}

function _loadLRProp(code, allProps) {
  try {
    var raw = allProps
      ? allProps[LR_PROP_PREFIX + code]
      : PropertiesService.getScriptProperties().getProperty(LR_PROP_PREFIX + code);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) { return null; }
}

function _fetchKRLows(code, count) {
  var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code
            + "&timeframe=day&count=" + count + "&requestType=0";
  var opts = { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT_RSI } };
  var text = _fetchAndCache(url, "LR_KR_" + code + "_" + count, opts, true);
  if (!text) return null;

  var lows = [], m, re = /item data="([^"]+)"/g;
  while ((m = re.exec(text)) !== null) {
    var low = parseFloat(m[1].split("|")[3]);
    if (!isNaN(low) && isFinite(low) && low > 0) lows.push(low);
  }
  return lows.length ? lows : null;
}

function _fetchUSLows(symbol, count) {
  var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
            + "?range=" + (Math.ceil(count * 1.5) + "d") + "&interval=1d";
  var opts = {
    muteHttpExceptions: true,
    headers: { "User-Agent": "Mozilla/5.0", "Accept": "application/json", "Cache-Control": "no-cache" }
  };
  var text = _fetchAndCache(url, "LR_US_" + symbol + "_" + count, opts, false);
  if (!text) return null;

  try {
    var quote = JSON.parse(text).chart.result[0].indicators.quote[0];
    if (!quote || !quote.low) return null;
    var lows = quote.low.filter(function(v) { return v !== null && !isNaN(v) && isFinite(v) && v > 0; });
    return lows.length ? lows : null;
  } catch (e) {
    console.log("[LR US 파싱 오류] " + symbol + ": " + e.toString());
    return null;
  }
}

function _calcLRForSymbol(symbol, allProps) {
  var resolved = resolveToTickerForRSI(symbol);
  if (resolved.type === "UNKNOWN") return null;

  var cached = _loadLRProp(resolved.code, allProps);
  if (cached && typeof cached.value === "number" && (Date.now() - (cached.ts || 0)) < LR_CACHE_SEC * 1000) {
    return cached;
  }

  var lows = null;
  for (var attempt = 1; attempt <= 3; attempt++) {
    try {
      lows = resolved.type === "KR"
        ? _fetchKRLows(resolved.code, LR_DATA_COUNT)
        : _fetchUSLows(resolved.code, LR_DATA_COUNT);
      if (lows && lows.length >= LR_PERIOD) break;
    } catch (e) {}
    Utilities.sleep(1500 * attempt);
  }

  if (!lows || lows.length < LR_PERIOD) {
    console.log("[LR 데이터 부족] " + symbol + " (" + (lows ? lows.length : 0) + "/" + LR_PERIOD + ")");
    return null;
  }

  var result = _calcLR(lows, LR_PERIOD);
  if (!result) return null;

  _saveLRProp(resolved.code, result);
  console.log("[LR] " + symbol + " | 추세선: " + result.value + " | 기울기: " + result.slope);
  return result;
}

function updateLRTrendlineAll() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  if (!sheet) { console.log("[LR] 기술분석 시트 없음"); return; }

  var lastRow = sheet.getLastRow();
  if (lastRow < 3) return;

  var rowCount = lastRow - 2;
  var symbols  = sheet.getRange(3, 1, rowCount, 1).getValues();

  // ① 모든 LR 캐시를 한 번에 로드 (종목당 getProperty 개별 호출 → 1회 getProperties)
  var allProps    = PropertiesService.getScriptProperties().getProperties();
  var propsToSave = {};
  var now         = Date.now();

  var results  = [];
  var isUS     = [];
  var successN = 0, skipN = 0, errorN = 0;

  console.log("[LR] 시작 — " + rowCount + "개 종목");

  for (var i = 0; i < rowCount; i++) {
    var symbol = String(symbols[i][0]).trim();
    if (!symbol) { results.push([""]); isUS.push(false); skipN++; continue; }

    try {
      var resolved = resolveToTickerForRSI(symbol);
      if (resolved.type === "UNKNOWN") { results.push([""]); isUS.push(false); skipN++; continue; }

      var cacheKey  = LR_PROP_PREFIX + resolved.code;
      var cachedRaw = allProps[cacheKey];
      var result    = null;
      var wasCached = false;

      // ② 이미 로드된 allProps에서 캐시 확인 (개별 getProperty 제거)
      if (cachedRaw) {
        try {
          var parsed = JSON.parse(cachedRaw);
          if (parsed && typeof parsed.value === "number" && (now - (parsed.ts || 0)) < LR_CACHE_SEC * 1000) {
            result    = parsed;
            wasCached = true;
          }
        } catch (e) {}
      }

      if (!result) {
        // 캐시 미스 → 외부 fetch
        var lows = null;
        for (var attempt = 1; attempt <= 3; attempt++) {
          try {
            lows = resolved.type === "KR"
              ? _fetchKRLows(resolved.code, LR_DATA_COUNT)
              : _fetchUSLows(resolved.code, LR_DATA_COUNT);
            if (lows && lows.length >= LR_PERIOD) break;
          } catch (e) {}
          Utilities.sleep(1500 * attempt);
        }

        if (lows && lows.length >= LR_PERIOD) {
          result = _calcLR(lows, LR_PERIOD);
          if (result) {
            // ③ 즉시 setProperty 대신 배치 객체에 누적
            propsToSave[cacheKey] = JSON.stringify({ value: result.value, slope: result.slope, ts: now });
            console.log("[LR] " + symbol + " | 추세선: " + result.value + " | 기울기: " + result.slope);
          }
        } else {
          console.log("[LR 데이터 부족] " + symbol + " (" + (lows ? lows.length : 0) + "/" + LR_PERIOD + ")");
        }

        // ④ 캐시 미스(실제 fetch)일 때만 sleep — 캐시 히트 시 불필요
        Utilities.sleep(LR_SLEEP_MS);
      }

      if (result && typeof result.value === "number") {
        results.push([result.value]);
        isUS.push(resolved.type === "US");
        successN++;
      } else {
        results.push(["데이터 부족"]);
        isUS.push(false);
        errorN++;
      }
    } catch (e) {
      console.log("[LR 예외] " + symbol + ": " + e.toString());
      results.push(["데이터 부족"]);
      isUS.push(false);
      errorN++;
    }
  }

  // ⑤ 새로 계산된 결과를 한 번에 저장 (종목당 setProperty → setProperties 1회)
  if (Object.keys(propsToSave).length > 0) {
    try {
      PropertiesService.getScriptProperties().setProperties(propsToSave, true);
    } catch (e) {
      console.log("[LR 배치 저장 실패, 개별 저장 시도] " + e.toString());
      Object.keys(propsToSave).forEach(function(k) {
        try { PropertiesService.getScriptProperties().setProperty(k, propsToSave[k]); } catch (e2) {}
      });
    }
  }

  if (results.length > 0) {
    sheet.getRange(3, LR_COL, results.length, 1).setValues(results);

    // ⑥ setNumberFormat 개별 셀 호출(81회) → 연속 범위를 묶어 최소 API 호출
    _applyGroupedNumberFormats(sheet, LR_COL, results, isUS, 3);
  }

  console.log("[LR] 완료 | 성공: " + successN + " | 실패: " + errorN + " | 스킵: " + skipN);
}

/**
 * KR / US 종목을 연속 행 블록으로 묶어 setNumberFormat 호출 횟수를 최소화.
 */
function _applyGroupedNumberFormats(sheet, col, results, isUS, startRow) {
  var groups = { US: [], KR: [] };
  for (var i = 0; i < results.length; i++) {
    var v = results[i][0];
    if (v === "" || v === "데이터 부족") continue;
    groups[isUS[i] ? "US" : "KR"].push(startRow + i);
  }

  [["US", "#,##0.00"], ["KR", "#,##0"]].forEach(function(pair) {
    var key  = pair[0];
    var fmt  = pair[1];
    var rows = groups[key];
    if (!rows.length) return;

    var start = rows[0], prev = rows[0];
    for (var i = 1; i <= rows.length; i++) {
      if (i < rows.length && rows[i] === prev + 1) {
        prev = rows[i];
      } else {
        sheet.getRange(start, col, prev - start + 1, 1).setNumberFormat(fmt);
        if (i < rows.length) { start = rows[i]; prev = rows[i]; }
      }
    }
  });
}

/**
 * 셀 수식: =if(A3<>"",fetchLRTrendline(A3),"")
 * updateLRTrendlineAll() 캐시 적재 후 SpreadsheetApp.flush() 시 읽힘.
 * 캐시 없을 때는 직접 계산 (cold-start 대비).
 * @param {string} symbol
 * @returns {number|string}
 * @customfunction
 */
function fetchLRTrendline(symbol) {
  if (!symbol || String(symbol).trim() === "") return "";
  try {
    var resolved = resolveToTickerForRSI(String(symbol).trim());
    if (resolved.type === "UNKNOWN") return "";

    var cached = _loadLRProp(resolved.code);
    if (cached && typeof cached.value === "number" && cached.value > 0) return cached.value;

    var result = _calcLRForSymbol(resolved.code);
    return (result && result.value > 0) ? result.value : "데이터 부족";
  } catch (e) { return "데이터 부족"; }
}

/**
 * updateInvestmentOpinion에서 C그룹 조건 판단 시 사용.
 * @param {string} symbol
 * @param {Object=} allProps  이미 로드된 getProperties() 결과 (옵션)
 * @returns {number} 양수=상승추세, 음수=하락추세, 0=데이터없음
 */
function getLRSlope(symbol, allProps) {
  try {
    var resolved = resolveToTickerForRSI(symbol);
    if (resolved.type === "UNKNOWN") return 0;
    var cached = _loadLRProp(resolved.code, allProps);
    return (cached && typeof cached.slope === "number") ? cached.slope : 0;
  } catch (e) { return 0; }
}

function clearLRTrendlineCache() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  if (!sheet) return;

  var rows      = sheet.getRange(3, 1, sheet.getLastRow() - 2, 1).getValues();
  var props     = PropertiesService.getScriptProperties();
  var cache     = CacheService.getScriptCache();
  var propKeys  = [], cacheKeys = [];

  for (var i = 0; i < rows.length; i++) {
    var symbol = String(rows[i][0]).trim();
    if (!symbol) continue;
    var resolved = resolveToTickerForRSI(symbol);
    propKeys.push(LR_PROP_PREFIX + resolved.code);
    cacheKeys.push("LR_KR_" + resolved.code + "_" + LR_DATA_COUNT);
    cacheKeys.push("LR_US_" + resolved.code + "_" + LR_DATA_COUNT);
  }

  propKeys.forEach(function(k) { props.deleteProperty(k); });
  cache.removeAll(cacheKeys);
  console.log("[LR캐시 초기화] Props " + propKeys.length + "개 / Cache " + cacheKeys.length + "개 삭제");
}
