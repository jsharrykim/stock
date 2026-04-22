var USER_AGENT_RSI = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";
var USER_AGENT_CCI = USER_AGENT_RSI;
var _CACHE_TTL = 10800;

// ──────────────────────────────────────────────
// PropertiesService 결과 캐시
// ──────────────────────────────────────────────
function _propKey(type, code, period) {
  return "RCCI_" + type + "_" + code + "_" + period;
}

function _loadCachedResult(type, code, period, allProps) {
  try {
    var raw = allProps
      ? allProps[_propKey(type, code, period)]
      : PropertiesService.getScriptProperties().getProperty(_propKey(type, code, period));
    if (!raw) return null;
    return JSON.parse(raw);
  } catch(e) { return null; }
}

function _isValidResultArr(arr) {
  return Array.isArray(arr) && arr.length === 3 &&
    arr.every(function(v) { return typeof v === "number" && isFinite(v) && !isNaN(v); });
}

// ──────────────────────────────────────────────
// [트리거용] 전 종목 순차 캐시 적재
// ──────────────────────────────────────────────
function batchPrefetchPrices() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  if (!sheet) return;
  var lastRow = sheet.getLastRow();
  if (lastRow < 3) return;
  var rows = sheet.getRange(3, 1, lastRow - 2, 1).getValues();

  // ① 기존 캐시를 한 번에 로드 (종목마다 getProperty → 1회 getProperties)
  var allProps   = PropertiesService.getScriptProperties().getProperties();
  var propsToSave = {};
  var now        = Date.now();
  var period     = 14;

  for (var i = 0; i < rows.length; i++) {
    var symbol = String(rows[i][0]).trim();
    if (!symbol) continue;
    try {
      var resolved = resolveToTickerForRSI(symbol);
      if (resolved.type === "UNKNOWN") continue;

      var propKey = _propKey(resolved.type, resolved.code, period);

      // ② 이미 유효한 캐시가 있으면 skip (fetch + sleep 불필요)
      var existing = allProps[propKey];
      if (existing) {
        try {
          var parsed = JSON.parse(existing);
          if (parsed && _isValidResultArr(parsed.rsi) && _isValidResultArr(parsed.cci)
              && (now - (parsed.ts || 0)) < _CACHE_TTL * 1000) {
            continue; // 캐시 유효 → 이 종목 건너뜀
          }
        } catch(e) {}
      }

      var rsiResult, cciResult;
      if (resolved.type === "KR") {
        rsiResult = getKoreanRSI(resolved.code, period);
        cciResult = getKoreanCCI(resolved.code, period);
      } else {
        rsiResult = getUSRSI(resolved.code, period);
        cciResult = getUSCCI(resolved.code, period);
      }

      if (rsiResult && rsiResult[0] && _isValidResultArr(rsiResult[0])
          && cciResult && cciResult[0] && _isValidResultArr(cciResult[0])) {
        // ③ 즉시 setProperty 대신 배치 객체에 누적
        propsToSave[propKey] = JSON.stringify({
          rsi: rsiResult[0], cci: cciResult[0], ts: now
        });
      }

      // ④ 실제 fetch가 발생한 종목에만 sleep (캐시 히트 시 불필요)
      Utilities.sleep(300);
    } catch (e) {}
  }

  // ⑤ 신규/갱신 결과를 한 번에 저장 (종목당 setProperty → setProperties 1회)
  if (Object.keys(propsToSave).length > 0) {
    try {
      PropertiesService.getScriptProperties().setProperties(propsToSave, true);
    } catch (e) {
      console.log("[batchPrefetch 배치 저장 실패, 개별 저장] " + e.toString());
      Object.keys(propsToSave).forEach(function(k) {
        try { PropertiesService.getScriptProperties().setProperty(k, propsToSave[k]); } catch(e2) {}
      });
    }
  }

  Logger.log("batchPrefetchPrices 완료 | 신규 저장: " + Object.keys(propsToSave).length + "건");
}

function _fetchAndCache(url, key, options, isKR) {
  var cache = CacheService.getScriptCache();
  if (cache.get(key)) return cache.get(key);

  for (var attempt = 0; attempt < 3; attempt++) {
    try {
      var res = UrlFetchApp.fetch(url, options);
      if (res.getResponseCode() === 429) {
        Utilities.sleep(2000 * (attempt + 1));
        continue;
      }
      if (res.getResponseCode() === 200) {
        var text = isKR ? res.getContentText("EUC-KR") : res.getContentText();
        cache.put(key, text, _CACHE_TTL);
        return text;
      }
      break;
    } catch (e) {
      Utilities.sleep(1000);
    }
  }
  return null;
}

// ──────────────────────────────────────────────
// 공개 커스텀 함수 (RSI, CCI)
// ──────────────────────────────────────────────
function getRSI(symbol, period) {
  period = period || 14;
  try {
    var resolved = resolveToTickerForRSI(symbol);
    var cached = _loadCachedResult(resolved.type, resolved.code, period);
    if (cached && _isValidResultArr(cached.rsi)) return [cached.rsi];

    if (resolved.type === "KR") return getKoreanRSI(resolved.code, period);
    if (resolved.type === "US") return getUSRSI(resolved.code, period);
    return [["종목 코드를 찾지 못했습니다"]];
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

function getCCI(symbol, period) {
  period = period || 14;
  try {
    var resolved = resolveToTickerForRSI(symbol);
    var cached = _loadCachedResult(resolved.type, resolved.code, period);
    if (cached && _isValidResultArr(cached.cci)) return [cached.cci];

    if (resolved.type === "KR") return getKoreanCCI(resolved.code, period);
    if (resolved.type === "US") return getUSCCI(resolved.code, period);
    return [["종목 코드를 찾지 못했습니다"]];
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 종목명/티커 → { type, code } 변환
// ──────────────────────────────────────────────
function resolveToTickerForRSI(input) {
  var trimmed = String(input).trim();
  if (/^\d{6}$/.test(trimmed)) return { type: "KR", code: trimmed };
  if (/^[A-Za-z][A-Za-z0-9.\-]*$/.test(trimmed)) return { type: "US", code: trimmed.toUpperCase() };
  var code = searchNaverStockCodeForRSI(trimmed);
  if (code) return { type: "KR", code: code };
  return { type: "UNKNOWN", code: trimmed };
}

function searchNaverStockCodeForRSI(name) {
  try {
    var url = "https://ac.stock.naver.com/ac?q=" + encodeURIComponent(name) + "&target=stock";
    var res = UrlFetchApp.fetch(url, { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT_RSI } });
    if (res.getResponseCode() !== 200) return null;
    var json = JSON.parse(res.getContentText());
    if (!json.items || json.items.length === 0) return null;
    var korItems = json.items.filter(function(i) { return i.nationCode === "KOR"; });
    if (korItems.length === 0) return null;
    var matched = korItems.find(function(i) { return i.name === name; }) || korItems[0];
    return matched.code;
  } catch (e) {
    return null;
  }
}

// ──────────────────────────────────────────────
// 미국 주식 RSI
// ──────────────────────────────────────────────
function getUSRSI(symbol, period) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=" + (period + 60) + "d&interval=1d";
    var options = { 'muteHttpExceptions': true, 'headers': { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', 'Cache-Control': 'no-cache' } };

    var text = _fetchAndCache(url, "US_" + symbol + "_" + period, options, false);
    if (!text) throw new Error("API 통신 오류");
    var data = JSON.parse(text);

    var prices = (data.chart && data.chart.result && data.chart.result[0] &&
                  data.chart.result[0].indicators && data.chart.result[0].indicators.quote &&
                  data.chart.result[0].indicators.quote[0].close) || [];

    var validPrices = prices.filter(function(p) { return p !== null && !isNaN(p) && isFinite(p); });
    if (validPrices.length < period + 1) return [["데이터 부족"]];

    var rsiValues = _calcRSI(validPrices, period);
    if (!rsiValues || rsiValues.length < 2) return [["데이터 부족"]];

    var latestPrice = prices[prices.length - 1];
    var prevPrice   = prices[prices.length - 2];

    var currentRSI = (latestPrice !== null && !isNaN(latestPrice))
      ? Number(rsiValues[rsiValues.length - 1].toFixed(2)) : "데이터 부족";
    var prevRSI = (prevPrice !== null && !isNaN(prevPrice))
      ? Number(rsiValues[rsiValues.length - 2].toFixed(2)) : "데이터 부족";

    var signal = Number(calculateEMA(rsiValues, 9).toFixed(2));
    return [[currentRSI, prevRSI, signal]];
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 RSI
// ──────────────────────────────────────────────
function getKoreanRSI(code, period) {
  try {
    var count = period * 10 + 50;
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=" + count + "&requestType=0";
    var options = { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT_RSI } };

    var text = _fetchAndCache(url, "KR_" + code + "_" + period, options, true);
    if (!text) return [["데이터 불러오기 실패"]];

    var itemRegex = /item data="([^"]+)"/g;
    var prices = [];
    var m;
    while ((m = itemRegex.exec(text)) !== null) {
      var parts = m[1].split("|");
      var close = parseFloat(parts[4]);
      prices.push(isNaN(close) ? null : close);
    }

    var validPrices = prices.filter(function(p) { return p !== null && !isNaN(p) && isFinite(p); });
    if (validPrices.length < period + 1) return [["데이터 부족"]];

    var rsiValues = _calcRSI(validPrices, period);
    if (!rsiValues || rsiValues.length < 2) return [["데이터 부족"]];

    var latestPrice = prices[prices.length - 1];
    var prevPrice   = prices[prices.length - 2];

    var currentRSI = (latestPrice !== null && !isNaN(latestPrice))
      ? Number(rsiValues[rsiValues.length - 1].toFixed(2)) : "데이터 부족";
    var prevRSI = (prevPrice !== null && !isNaN(prevPrice))
      ? Number(rsiValues[rsiValues.length - 2].toFixed(2)) : "데이터 부족";

    var signal = Number(calculateEMA(rsiValues, 9).toFixed(2));
    return [[currentRSI, prevRSI, signal]];
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 CCI
// ──────────────────────────────────────────────
function getUSCCI(symbol, period) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=" + (period + 60) + "d&interval=1d";
    var options = { 'muteHttpExceptions': true, 'headers': { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', 'Cache-Control': 'no-cache' } };

    var text = _fetchAndCache(url, "US_" + symbol + "_" + period, options, false);
    if (!text) throw new Error("API 통신 오류");
    var data = JSON.parse(text);

    var quote = data.chart && data.chart.result && data.chart.result[0] &&
                data.chart.result[0].indicators && data.chart.result[0].indicators.quote &&
                data.chart.result[0].indicators.quote[0];
    if (!quote) return [["데이터 오류: 주가 데이터를 찾을 수 없습니다."]];

    var high  = quote.high  || [];
    var low   = quote.low   || [];
    var close = quote.close || [];

    var validHigh  = high.filter(function(h) { return h !== null && !isNaN(h) && isFinite(h); });
    var validLow   = low.filter(function(l)  { return l !== null && !isNaN(l) && isFinite(l); });
    var validClose = close.filter(function(c) { return c !== null && !isNaN(c) && isFinite(c); });

    if (validHigh.length < period || validLow.length < period || validClose.length < period) {
      return [["데이터 부족"]];
    }

    return calcCCIFromHLC(high, low, close, validHigh, validLow, validClose, period);
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 CCI
// ──────────────────────────────────────────────
function getKoreanCCI(code, period) {
  try {
    var count = period * 10 + 50;
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=" + count + "&requestType=0";
    var options = { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT_CCI } };

    var text = _fetchAndCache(url, "KR_" + code + "_" + period, options, true);
    if (!text) return [["데이터 불러오기 실패"]];

    var itemRegex = /item data="([^"]+)"/g;
    var high = [], low = [], close = [];
    var m;
    while ((m = itemRegex.exec(text)) !== null) {
      var parts = m[1].split("|");
      var h = parseFloat(parts[2]);
      var l = parseFloat(parts[3]);
      var c = parseFloat(parts[4]);
      high.push(isNaN(h) ? null : h);
      low.push(isNaN(l)  ? null : l);
      close.push(isNaN(c) ? null : c);
    }

    if (high.length === 0) return [["데이터 부족"]];

    var validHigh  = high.filter(function(h) { return h !== null && !isNaN(h) && isFinite(h); });
    var validLow   = low.filter(function(l)  { return l !== null && !isNaN(l) && isFinite(l); });
    var validClose = close.filter(function(c) { return c !== null && !isNaN(c) && isFinite(c); });

    if (validHigh.length < period || validLow.length < period || validClose.length < period) {
      return [["데이터 부족"]];
    }

    return calcCCIFromHLC(high, low, close, validHigh, validLow, validClose, period);
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// RSI 계산 공통 로직
// ──────────────────────────────────────────────
function _calcRSI(validPrices, period) {
  var rsiValues = [];
  for (var i = period; i < validPrices.length; i++) {
    var gains = 0, losses = 0;
    for (var j = i - period + 1; j <= i; j++) {
      var change = validPrices[j] - validPrices[j - 1];
      if (change > 0) gains += change;
      if (change < 0) losses -= change;
    }
    var avgGain = gains / period;
    var avgLoss = losses / period;
    if (avgLoss === 0) {
      rsiValues.push(100);
    } else {
      var rs = avgGain / avgLoss;
      rsiValues.push(100 - (100 / (1 + rs)));
    }
  }
  return rsiValues;
}

// ──────────────────────────────────────────────
// 공통 CCI 계산 로직 & EMA
// ──────────────────────────────────────────────
function calcCCIFromHLC(high, low, close, validHigh, validLow, validClose, period) {
  var cciValues = [];
  for (var i = period - 1; i < validHigh.length; i++) {
    var recentHigh  = validHigh.slice(i - (period - 1), i + 1);
    var recentLow   = validLow.slice(i - (period - 1), i + 1);
    var recentClose = validClose.slice(i - (period - 1), i + 1);

    var tp = recentHigh.map(function(h, j) {
      return (h + recentLow[j] + recentClose[j]) / 3;
    });

    var sma = tp.reduce(function(a, b) { return a + b; }, 0) / period;
    var meanDev = tp.reduce(function(sum, price) { return sum + Math.abs(price - sma); }, 0) / period;
    cciValues.push(meanDev === 0 ? 0 : (tp[tp.length - 1] - sma) / (0.015 * meanDev));
  }

  if (cciValues.length < 2) return [["데이터 부족"]];

  var latestPrice = close[close.length - 1];
  var prevPrice   = close[close.length - 2];

  var currentCCI = (latestPrice !== null && !isNaN(latestPrice))
    ? Number(cciValues[cciValues.length - 1].toFixed(2)) : "데이터 부족";
  var prevCCI = (prevPrice !== null && !isNaN(prevPrice))
    ? Number(cciValues[cciValues.length - 2].toFixed(2)) : "데이터 부족";

  var signal = Number(calculateEMA(cciValues, 9).toFixed(2));
  return [[currentCCI, prevCCI, signal]];
}

function calculateEMA(prices, period) {
  if (!Array.isArray(prices) || prices.length < period) return 0;
  var multiplier = 2 / (period + 1);
  var ema = prices.slice(0, period).reduce(function(a, b) { return a + b; }, 0) / period;
  for (var i = period; i < prices.length; i++) {
    ema = (prices[i] - ema) * multiplier + ema;
  }
  return ema;
}

// ──────────────────────────────────────────────
// 캐시 강제 초기화
// ──────────────────────────────────────────────
function clearPriceCache() {
  var cache = CacheService.getScriptCache();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  if (!sheet) return;
  var rows = sheet.getRange(3, 1, sheet.getLastRow() - 2, 1).getValues();
  var scriptCacheKeys = [];
  var propKeys = [];

  for (var i = 0; i < rows.length; i++) {
    var symbol = String(rows[i][0]).trim();
    if (!symbol) continue;
    scriptCacheKeys.push("US_" + symbol + "_14");
    scriptCacheKeys.push("KR_" + symbol + "_14");
    propKeys.push(_propKey("US", symbol, 14));
    propKeys.push(_propKey("KR", symbol, 14));
  }

  cache.removeAll(scriptCacheKeys);
  var props = PropertiesService.getScriptProperties();
  propKeys.forEach(function(k) { props.deleteProperty(k); });
  Logger.log("ScriptCache + PropertiesService 캐시 삭제 완료");
}
