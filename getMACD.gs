var USER_AGENT_MACD = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function getMACD(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanMACD(resolved.code);
    } else if (resolved.type === "US") {
      return getUSMACD(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다"]];
    }
  } catch (error) {
    return [["오류 발생: " + error.toString() + " (Symbol: " + symbol + ")"]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 MACD (Yahoo Finance)
// ──────────────────────────────────────────────
function getUSMACD(symbol) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=200d&interval=1d";
    var options = {
      muteHttpExceptions: true,
      headers: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Cache-Control": "no-cache"
      }
    };

    var response = UrlFetchApp.fetch(url, options);
    var data = JSON.parse(response.getContentText());
    var prices = data.chart.result[0].indicators.quote[0].close;
    var priceArray = prices.filter(function(price) {
      return price !== null && typeof price === "number";
    });

    if (priceArray.length < 26) {
      return [["데이터 부족: MACD 계산에 필요한 충분한 종가가 없습니다."]];
    }

    return calcMACDFromPrices(priceArray);
  } catch (error) {
    if (error.toString().indexOf("429") !== -1) {
      Utilities.sleep(2000);
      return getUSMACD(symbol);
    }
    return [["오류 발생: " + error.toString() + " (Symbol: " + symbol + ")"]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 MACD (네이버 금융 차트 XML API)
// ──────────────────────────────────────────────
function getKoreanMACD(code) {
  try {
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=200&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_MACD }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")"]];
    }

    var xml = res.getContentText("EUC-KR");
    var itemRegex = /item data="([^"]+)"/g;
    var prices = [];
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      var parts = m[1].split("|");
      var close = parseFloat(parts[4]);
      if (!isNaN(close)) prices.push(close);
    }

    if (prices.length < 26) {
      return [["데이터 부족: MACD 계산에 필요한 충분한 종가가 없습니다."]];
    }

    return calcMACDFromPrices(prices);
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 공통 MACD 계산 로직
// ──────────────────────────────────────────────
function calcMACDFromPrices(priceArray) {
  var shortEMA = calculateMACDEMA(priceArray, 12);
  var longEMA = calculateMACDEMA(priceArray, 26);
  var macdLine = shortEMA - longEMA;

  var allShortEMAs = calculateAllHistoricalMACDEMAs(priceArray, 12);
  var allLongEMAs = calculateAllHistoricalMACDEMAs(priceArray, 26);

  var macdValues = [];
  var startIndex = Math.max(12, 26) - 1;
  var i;

  for (i = startIndex; i < priceArray.length; i++) {
    if (allShortEMAs[i] !== undefined && allLongEMAs[i] !== undefined) {
      macdValues.push(allShortEMAs[i] - allLongEMAs[i]);
    }
  }

  if (macdValues.length < 9) {
    return [["데이터 부족: 시그널 라인 계산에 필요한 MACD 값이 충분하지 않습니다."]];
  }

  var signalLine = calculateMACDEMA(macdValues, 9);
  var histogram = macdLine - signalLine;

  var prevMacdLineForHist1 = macdValues[macdValues.length - 2];
  var prevSignalLineForHist1 = calculateMACDEMA(macdValues.slice(0, macdValues.length - 1), 9);
  var prevHistogram1 = prevMacdLineForHist1 - prevSignalLineForHist1;

  var prevMacdLineForHist2 = macdValues[macdValues.length - 3];
  var prevSignalLineForHist2 = calculateMACDEMA(macdValues.slice(0, macdValues.length - 2), 9);
  var prevHistogram2 = prevMacdLineForHist2 - prevSignalLineForHist2;

  var prevShortEMA = calculateMACDEMA(priceArray.slice(0, -1), 12);
  var prevLongEMA = calculateMACDEMA(priceArray.slice(0, -1), 26);
  var prevMACD = prevShortEMA - prevLongEMA;

  return [[
    Number(macdLine.toFixed(2)),
    Number(prevMACD.toFixed(2)),
    Number(signalLine.toFixed(2)),
    Number(histogram.toFixed(2)),
    Number(prevHistogram1.toFixed(2)),
    Number(prevHistogram2.toFixed(2))
  ]];
}

// ──────────────────────────────────────────────
// EMA 헬퍼 함수
// ──────────────────────────────────────────────
function calculateMACDEMA(prices, period) {
  if (!Array.isArray(prices) || prices.length < period) {
    return 0;
  }

  var multiplier = 2 / (period + 1);
  var ema = prices.slice(0, period).reduce(function(a, b) {
    return a + b;
  }, 0) / period;
  var i;

  for (i = period; i < prices.length; i++) {
    ema = (prices[i] - ema) * multiplier + ema;
  }

  return ema;
}

function calculateAllHistoricalMACDEMAs(prices, period) {
  var emas = [];
  if (!Array.isArray(prices) || prices.length < period) {
    return emas;
  }

  var currentEma = prices.slice(0, period).reduce(function(a, b) {
    return a + b;
  }, 0) / period;
  var i;

  for (i = 0; i < period - 1; i++) {
    emas.push(undefined);
  }

  emas.push(currentEma);

  var multiplier = 2 / (period + 1);
  for (i = period; i < prices.length; i++) {
    currentEma = (prices[i] - currentEma) * multiplier + currentEma;
    emas.push(currentEma);
  }

  return emas;
}
