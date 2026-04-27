var USER_AGENT_BB = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function getBollingerBand(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanBollingerBand(resolved.code);
    } else if (resolved.type === "US") {
      return getUSBollingerBand(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다", "", "", "", "", "", ""]];
    }
  } catch (error) {
    return [["오류", "오류", "오류", "오류", "오류", "오류", "오류"]];
  }
}

function getUSBollingerBand(symbol) {
  var responseText = "";
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=1y&interval=1d";
    var options = {
      muteHttpExceptions: true,
      headers: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Cache-Control": "no-cache"
      }
    };

    var response = UrlFetchApp.fetch(url, options);
    responseText = response.getContentText();
    var data = JSON.parse(responseText);

    var result = data.chart.result;
    if (!result || result.length === 0 || !result[0].indicators || !result[0].indicators.quote || !result[0].indicators.quote[0]) {
      return [["데이터 없음", "데이터 없음", "데이터 없음", "데이터 없음", "데이터 없음", "데이터 없음", "데이터 없음"]];
    }

    var closePrices = result[0].indicators.quote[0].close;
    var highPrices = result[0].indicators.quote[0].high;
    var lowPrices = result[0].indicators.quote[0].low;
    var validData = [];
    var i;

    for (i = 0; i < closePrices.length; i++) {
      if (closePrices[i] !== null && highPrices[i] !== null && lowPrices[i] !== null) {
        validData.push({
          close: closePrices[i],
          high: highPrices[i],
          low: lowPrices[i]
        });
      }
    }

    return calcBollingerBandFromData(validData);
  } catch (error) {
    var errorMessage = "오류";
    if (error.toString().indexOf("SyntaxError: Unexpected token") !== -1 && responseText && responseText.indexOf("upstream c") !== -1) {
      errorMessage = "서버 오류";
    }
    return [[errorMessage, errorMessage, errorMessage, errorMessage, errorMessage, errorMessage, errorMessage]];
  }
}

function getKoreanBollingerBand(code) {
  try {
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=300&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_BB }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")", "", "", "", "", "", ""]];
    }

    var xml = res.getContentText("EUC-KR");
    var itemRegex = /item data="([^"]+)"/g;
    var validData = [];
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      var parts = m[1].split("|");
      var high = parseFloat(parts[2]);
      var low = parseFloat(parts[3]);
      var close = parseFloat(parts[4]);
      if (!isNaN(high) && !isNaN(low) && !isNaN(close)) {
        validData.push({ close: close, high: high, low: low });
      }
    }

    return calcBollingerBandFromData(validData);
  } catch (error) {
    return [["오류", "오류", "오류", "오류", "오류", "오류", "오류"]];
  }
}

function calcBollingerBandFromData(validData) {
  var period = 20;
  var stdDevMultiplier = 2;
  var avgBandwidthPeriod = 60;

  var bollingerBandD = NaN;
  var bollingerBandPeakD = NaN;
  var bandwidthD = NaN;
  var bollingerBandLowD = NaN;
  var bollingerBandPeakDMinus1 = NaN;
  var bandwidthDMinus1 = NaN;
  var avgBollingerBandwidth60Day = NaN;

  function calculateBands(pricesSlice) {
    if (pricesSlice.length < period) {
      return {
        sma: NaN,
        stdDev: NaN,
        upperBand: NaN,
        lowerBand: NaN,
        bandwidth: NaN
      };
    }

    var closingPricesOnly = pricesSlice.map(function(d) {
      return d.close;
    });
    var sma = closingPricesOnly.reduce(function(a, b) {
      return a + b;
    }, 0) / period;
    var stdDev = Math.sqrt(
      closingPricesOnly.reduce(function(sum, price) {
        return sum + Math.pow(price - sma, 2);
      }, 0) / period
    );
    var upperBand = sma + (stdDevMultiplier * stdDev);
    var lowerBand = sma - (stdDevMultiplier * stdDev);
    var bandwidth = NaN;

    if (sma !== 0) {
      bandwidth = ((upperBand - lowerBand) / sma) * 100;
    }

    return {
      sma: sma,
      stdDev: stdDev,
      upperBand: upperBand,
      lowerBand: lowerBand,
      bandwidth: bandwidth
    };
  }

  function getPercentB(price, lowerBand, upperBand) {
    if ((upperBand - lowerBand) === 0) {
      return price > upperBand ? 100 : price < lowerBand ? 0 : 50;
    }
    return ((price - lowerBand) / (upperBand - lowerBand)) * 100;
  }

  if (validData.length >= period) {
    var sliceD = validData.slice(-period);
    var lastDayDataD = validData[validData.length - 1];
    var bandD = calculateBands(sliceD);

    if (!isNaN(bandD.upperBand) && !isNaN(bandD.lowerBand)) {
      bollingerBandD = getPercentB(lastDayDataD.close, bandD.lowerBand, bandD.upperBand);
      bollingerBandPeakD = getPercentB(lastDayDataD.high, bandD.lowerBand, bandD.upperBand);
      bollingerBandLowD = getPercentB(lastDayDataD.low, bandD.lowerBand, bandD.upperBand);
      bandwidthD = bandD.bandwidth;
    }
  }

  if (validData.length >= period + 1) {
    var sliceDMinus1 = validData.slice(-period - 1, -1);
    var lastDayDataDMinus1 = validData[validData.length - 2];
    var bandDMinus1 = calculateBands(sliceDMinus1);

    if (!isNaN(bandDMinus1.upperBand) && !isNaN(bandDMinus1.lowerBand)) {
      bollingerBandPeakDMinus1 = getPercentB(lastDayDataDMinus1.high, bandDMinus1.lowerBand, bandDMinus1.upperBand);
      bandwidthDMinus1 = bandDMinus1.bandwidth;
    }
  }

  if (validData.length >= (period - 1) + avgBandwidthPeriod) {
    var allBandwidths = [];
    var i;

    for (i = validData.length - 1; i >= validData.length - avgBandwidthPeriod; i--) {
      if (i - (period - 1) >= 0) {
        var sliceForBandwidth = validData.slice(i - (period - 1), i + 1);
        var bandForBandwidth = calculateBands(sliceForBandwidth);
        if (!isNaN(bandForBandwidth.bandwidth)) {
          allBandwidths.push(bandForBandwidth.bandwidth);
        }
      }
    }

    if (allBandwidths.length > 0) {
      avgBollingerBandwidth60Day = allBandwidths.reduce(function(a, b) {
        return a + b;
      }, 0) / allBandwidths.length;
    }
  }

  return [[
    Number(!isNaN(bollingerBandD) ? bollingerBandD.toFixed(2) : NaN),
    Number(!isNaN(bollingerBandLowD) ? bollingerBandLowD.toFixed(2) : NaN),
    Number(!isNaN(bollingerBandPeakD) ? bollingerBandPeakD.toFixed(2) : NaN),
    Number(!isNaN(bollingerBandPeakDMinus1) ? bollingerBandPeakDMinus1.toFixed(2) : NaN),
    Number(!isNaN(bandwidthD) ? bandwidthD.toFixed(2) : NaN),
    Number(!isNaN(bandwidthDMinus1) ? bandwidthDMinus1.toFixed(2) : NaN),
    Number(!isNaN(avgBollingerBandwidth60Day) ? avgBollingerBandwidth60Day.toFixed(2) : NaN)
  ]];
}
