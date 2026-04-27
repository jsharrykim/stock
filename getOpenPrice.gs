function getOpenPrice(symbol) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=1d&interval=1d";
    var options = {
      muteHttpExceptions: true,
      headers: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
      }
    };

    var response = UrlFetchApp.fetch(url, options);
    var data = JSON.parse(response.getContentText());
    var quote = data.chart.result[0].indicators.quote[0];
    var openPrices = quote.open;

    return openPrices[openPrices.length - 1];
  } catch (e) {
    Logger.log("시가 가져오기 실패 (" + symbol + "): " + e.toString());
    return null;
  }
}
