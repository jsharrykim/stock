var USER_AGENT_CANDLE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function getCandle(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanCandle(resolved.code);
    } else if (resolved.type === "US") {
      return getUSCandle(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다"]];
    }
  } catch (error) {
    return [["Error: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 캔들 (Yahoo Finance)
// ──────────────────────────────────────────────
function getUSCandle(symbol) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?interval=1d&range=1d";
    var response = UrlFetchApp.fetch(url);
    var data = JSON.parse(response.getContentText());
    var quote = data.chart.result[0].indicators.quote[0];

    return [[
      quote.open[0],
      quote.high[0],
      quote.low[0],
      quote.close[0],
      quote.volume[0]
    ]];
  } catch (error) {
    return [["Error: " + error.toString()]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 캔들 (네이버 금융 차트 XML API)
// ──────────────────────────────────────────────
function getKoreanCandle(code) {
  try {
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=2&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_CANDLE }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")"]];
    }

    var xml = res.getContentText("EUC-KR");
    var items = [];
    var itemRegex = /item data="([^"]+)"/g;
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      items.push(m[1].split("|"));
    }

    if (items.length === 0) {
      return [["데이터 부족"]];
    }

    var latest = items[items.length - 1];
    var open = parseFloat(latest[1]);
    var high = parseFloat(latest[2]);
    var low = parseFloat(latest[3]);
    var close = parseFloat(latest[4]);
    var volume = parseFloat(latest[5]);

    return [[open, high, low, close, volume]];
  } catch (error) {
    return [["Error: " + error.toString()]];
  }
}
