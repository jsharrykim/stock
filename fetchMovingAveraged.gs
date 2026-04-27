var USER_AGENT_MA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function fetchMovingAveraged(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanMA(resolved.code);
    } else if (resolved.type === "US") {
      return getUSMA(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다", "", "", "", ""]];
    }
  } catch (e) {
    return [["오류", "오류", "오류", "오류", "오류"]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 이동평균 (Yahoo Finance)
// ──────────────────────────────────────────────
function getUSMA(symbol) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=1y&interval=1d";
    var response = UrlFetchApp.fetch(url);
    var json = JSON.parse(response.getContentText());
    var prices = json.chart.result[0].indicators.quote[0].close;

    return calcMAFromPrices(prices, false);
  } catch (e) {
    if (e.toString().indexOf("429") !== -1) {
      Utilities.sleep(2000);
      return getUSMA(symbol);
    }
    return [["오류", "오류", "오류", "오류", "오류"]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 이동평균 (네이버 금융 차트 XML API)
// ──────────────────────────────────────────────
function getKoreanMA(code) {
  try {
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=250&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_MA }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")", "", "", "", ""]];
    }

    var xml = res.getContentText("EUC-KR");
    var itemRegex = /item data="([^"]+)"/g;
    var prices = [];
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      var parts = m[1].split("|");
      var close = parseFloat(parts[4]);
      prices.push(isNaN(close) ? null : close);
    }

    return calcMAFromPrices(prices, true);
  } catch (e) {
    return [["오류", "오류", "오류", "오류", "오류"]];
  }
}

// ──────────────────────────────────────────────
// 공통 이동평균 계산 로직
// ──────────────────────────────────────────────
function calcMAFromPrices(prices, isKR) {
  var ma5 = NaN;
  var ma20 = NaN;
  var ma60 = NaN;
  var ma144 = NaN;
  var ma200 = NaN;

  if (prices && prices.length >= 5) ma5 = calculateAverage(prices, 5);
  if (prices && prices.length >= 20) ma20 = calculateAverage(prices, 20);
  if (prices && prices.length >= 60) ma60 = calculateAverage(prices, 60);
  if (prices && prices.length >= 144) ma144 = calculateAverage(prices, 144);
  if (prices && prices.length >= 200) ma200 = calculateAverage(prices, 200);

  var fmt = isKR
    ? function(v) { return isNaN(v) ? "" : Math.round(v); }
    : function(v) { return isNaN(v) ? "" : parseFloat(v.toFixed(2)); };

  return [[fmt(ma5), fmt(ma20), fmt(ma60), fmt(ma144), fmt(ma200)]];
}

function calculateAverage(prices, days) {
  if (!prices || prices.length < days) {
    return NaN;
  }

  var lastPrices = prices.slice(-days);
  var sum = lastPrices.filter(function(p) {
    return p !== null;
  }).reduce(function(a, b) {
    return a + b;
  }, 0);
  var validCount = lastPrices.filter(function(p) {
    return p !== null;
  }).length;

  return validCount > 0 ? sum / validCount : NaN;
}
