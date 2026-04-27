var USER_AGENT_ADX = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function getADX(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanADX(resolved.code);
    } else if (resolved.type === "US") {
      return getUSADX(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다"]];
    }
  } catch (error) {
    return [["오류 발생: " + error.toString() + " (Symbol: " + symbol + ")"]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 ADX (Yahoo Finance)
// ──────────────────────────────────────────────
function getUSADX(symbol) {
  try {
    var period = 14;
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

    if (!data || !data.chart || !data.chart.result || !data.chart.result[0] ||
        !data.chart.result[0].indicators || !data.chart.result[0].indicators.quote ||
        !data.chart.result[0].indicators.quote[0]) {
      return [["데이터 구조 오류: Yahoo Finance API 응답이 예상과 다릅니다. (Symbol: " + symbol + ")"]];
    }

    var highPricesRaw = data.chart.result[0].indicators.quote[0].high;
    var lowPricesRaw = data.chart.result[0].indicators.quote[0].low;
    var closePricesRaw = data.chart.result[0].indicators.quote[0].close;

    var pricesData = [];
    var i;
    for (i = 0; i < closePricesRaw.length; i++) {
      if (highPricesRaw[i] !== null && typeof highPricesRaw[i] === "number" &&
          lowPricesRaw[i] !== null && typeof lowPricesRaw[i] === "number" &&
          closePricesRaw[i] !== null && typeof closePricesRaw[i] === "number") {
        pricesData.push({
          high: highPricesRaw[i],
          low: lowPricesRaw[i],
          close: closePricesRaw[i]
        });
      }
    }

    return calcADXFromPricesData(pricesData, period, symbol);
  } catch (error) {
    if (error.toString().indexOf("429") !== -1) {
      Utilities.sleep(2000);
      return getUSADX(symbol);
    }
    return [["오류 발생: " + error.toString() + " (Symbol: " + symbol + ")"]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 ADX (네이버 금융 차트 XML API)
// ──────────────────────────────────────────────
function getKoreanADX(code) {
  try {
    var period = 14;
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=200&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_ADX }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")"]];
    }

    var xml = res.getContentText("EUC-KR");
    var itemRegex = /item data="([^"]+)"/g;
    var pricesData = [];
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      var parts = m[1].split("|");
      var h = parseFloat(parts[2]);
      var l = parseFloat(parts[3]);
      var c = parseFloat(parts[4]);
      if (!isNaN(h) && !isNaN(l) && !isNaN(c)) {
        pricesData.push({ high: h, low: l, close: c });
      }
    }

    return calcADXFromPricesData(pricesData, period, code);
  } catch (error) {
    return [["오류 발생: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 공통 ADX 계산 로직
// ──────────────────────────────────────────────
function calcADXFromPricesData(pricesData, period, symbol) {
  if (pricesData.length < 2 * period + 1) {
    return [["데이터 부족: ADX 계산에 필요한 충분한 데이터가 없습니다. (최소 " + (2 * period + 1) + "일 필요)"]];
  }

  var trValues = [];
  var plusDMValues = [];
  var minusDMValues = [];
  var i;

  for (i = 1; i < pricesData.length; i++) {
    var currentHigh = pricesData[i].high;
    var currentLow = pricesData[i].low;
    var prevClose = pricesData[i - 1].close;
    var prevHigh = pricesData[i - 1].high;
    var prevLow = pricesData[i - 1].low;

    var tr = Math.max(
      currentHigh - currentLow,
      Math.abs(currentHigh - prevClose),
      Math.abs(currentLow - prevClose)
    );
    trValues.push(tr);

    var plusDMRaw = currentHigh - prevHigh;
    var minusDMRaw = prevLow - currentLow;
    var currentPlusDM = 0;
    var currentMinusDM = 0;

    if (plusDMRaw > minusDMRaw && plusDMRaw > 0) {
      currentPlusDM = plusDMRaw;
    } else if (minusDMRaw > plusDMRaw && minusDMRaw > 0) {
      currentMinusDM = minusDMRaw;
    }

    plusDMValues.push(currentPlusDM);
    minusDMValues.push(currentMinusDM);
  }

  var smoothedTR = calculateAllHistoricalADXEMAs(trValues, period);
  var smoothedPlusDM = calculateAllHistoricalADXEMAs(plusDMValues, period);
  var smoothedMinusDM = calculateAllHistoricalADXEMAs(minusDMValues, period);

  var historicalPlusDI = [];
  var historicalMinusDI = [];
  var historicalDX = [];
  var firstValidEmaIndex = period - 1;

  for (i = firstValidEmaIndex; i < smoothedTR.length; i++) {
    var trSmooth = smoothedTR[i];
    var plusDMSmooth = smoothedPlusDM[i];
    var minusDMSmooth = smoothedMinusDM[i];
    var currentPlusDI = 0;
    var currentMinusDI = 0;

    if (trSmooth > 0) {
      currentPlusDI = (plusDMSmooth / trSmooth) * 100;
      currentMinusDI = (minusDMSmooth / trSmooth) * 100;
    }

    historicalPlusDI.push(currentPlusDI);
    historicalMinusDI.push(currentMinusDI);

    var sumDI = currentPlusDI + currentMinusDI;
    var currentDX = 0;
    if (sumDI > 0) {
      currentDX = (Math.abs(currentPlusDI - currentMinusDI) / sumDI) * 100;
    }
    historicalDX.push(currentDX);
  }

  if (historicalDX.length < period) {
    return [["데이터 부족: ADX 계산에 필요한 충분한 DX 값이 없습니다."]];
  }

  var historicalADX = calculateAllHistoricalADXEMAs(historicalDX, period);
  var currentPlusDIValue = historicalPlusDI[historicalPlusDI.length - 1];
  var currentMinusDIValue = historicalMinusDI[historicalMinusDI.length - 1];
  var currentADX = historicalADX[historicalADX.length - 1];

  if (historicalADX.length < 2) {
    return [["데이터 부족: ADX (D-1) 계산에 필요한 이전 ADX 값이 없습니다."]];
  }
  var prevADX1 = historicalADX[historicalADX.length - 2];

  if (historicalADX.length < 3) {
    return [["데이터 부족: ADX (D-2) 계산에 필요한 이전 ADX 값이 없습니다."]];
  }
  var prevADX2 = historicalADX[historicalADX.length - 3];

  var adxSlope = currentADX - prevADX1;

  return [[
    Number(currentPlusDIValue.toFixed(2)),
    Number(currentMinusDIValue.toFixed(2)),
    Number(currentADX.toFixed(2)),
    Number(prevADX1.toFixed(2)),
    Number(prevADX2.toFixed(2)),
    Number(adxSlope.toFixed(2))
  ]];
}

function calculateAllHistoricalADXEMAs(prices, period) {
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
