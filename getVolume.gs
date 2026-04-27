var USER_AGENT_VOL = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

function getVolume(symbol) {
  try {
    var resolved = resolveToTickerForRSI(symbol);

    if (resolved.type === "KR") {
      return getKoreanVolume(resolved.code);
    } else if (resolved.type === "US") {
      return getUSVolume(resolved.code);
    } else {
      return [["종목 코드를 찾지 못했습니다", "", "", ""]];
    }
  } catch (error) {
    return [["오류 발생: " + error.toString(), "", "", ""]];
  }
}

// ──────────────────────────────────────────────
// 미국 주식 거래량 (Yahoo Finance)
// ──────────────────────────────────────────────
function getUSVolume(symbol) {
  try {
    var url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?range=30d&interval=1d";
    var response = UrlFetchApp.fetch(url);
    var data = JSON.parse(response.getContentText());
    var volumes = (((data || {}).chart || {}).result || [])[0];

    volumes = volumes && volumes.indicators && volumes.indicators.quote &&
      volumes.indicators.quote[0] && volumes.indicators.quote[0].volume || [];

    return calcVolumeFromArray(volumes);
  } catch (error) {
    return [["오류 발생: " + error.toString(), "", "", ""]];
  }
}

// ──────────────────────────────────────────────
// 한국 주식 거래량 (네이버 금융 차트 XML API)
// ──────────────────────────────────────────────
function getKoreanVolume(code) {
  try {
    var url = "https://fchart.stock.naver.com/sise.nhn?symbol=" + code + "&timeframe=day&count=30&requestType=0";
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT_VOL }
    });

    if (res.getResponseCode() !== 200) {
      return [["데이터 불러오기 실패 (HTTP " + res.getResponseCode() + ")", "", "", ""]];
    }

    var xml = res.getContentText("EUC-KR");
    var itemRegex = /item data="([^"]+)"/g;
    var volumes = [];
    var m;

    while ((m = itemRegex.exec(xml)) !== null) {
      var parts = m[1].split("|");
      var vol = parseFloat(parts[5]);
      volumes.push(isNaN(vol) ? null : vol);
    }

    return calcVolumeFromArray(volumes);
  } catch (error) {
    return [["오류 발생: " + error.toString(), "", "", ""]];
  }
}

// ──────────────────────────────────────────────
// 공통 거래량 계산 로직
// ──────────────────────────────────────────────
function calcVolumeFromArray(volumes) {
  if (volumes.length < 20) {
    return [["데이터 부족", "데이터 부족", "데이터 부족", "데이터 부족"]];
  }

  var validVolumes = volumes.map(function(v) {
    return (v && v > 0) ? v : null;
  });

  var latestVolume = volumes[volumes.length - 1];
  var volumeRatio = "데이터 부족";
  if (latestVolume && latestVolume > 0) {
    var last5 = validVolumes.slice(-5).filter(function(v) {
      return v !== null;
    });
    if (last5.length > 0) {
      var avg5 = last5.reduce(function(a, b) {
        return a + b;
      }, 0) / last5.length;
      volumeRatio = Number((latestVolume / avg5).toFixed(2));
    }
  }

  var prevDayVolume = volumes[volumes.length - 2];
  var prevVolumeRatio = "데이터 부족";
  if (prevDayVolume && prevDayVolume > 0) {
    var prev5 = validVolumes.slice(-6, -1).filter(function(v) {
      return v !== null;
    });
    if (prev5.length > 0) {
      var avgPrev5 = prev5.reduce(function(a, b) {
        return a + b;
      }, 0) / prev5.length;
      prevVolumeRatio = Number((prevDayVolume / avgPrev5).toFixed(2));
    }
  }

  var avgVolume20DayPercent = "데이터 부족";
  if (latestVolume && latestVolume > 0) {
    var last20 = validVolumes.slice(-20).filter(function(v) {
      return v !== null;
    });
    if (last20.length > 0) {
      var avg20 = last20.reduce(function(a, b) {
        return a + b;
      }, 0) / last20.length;
      avgVolume20DayPercent = Number((latestVolume / avg20).toFixed(2));
    }
  }

  return [[
    volumeRatio,
    prevVolumeRatio,
    avgVolume20DayPercent,
    (latestVolume && latestVolume > 0) ? latestVolume : "데이터 부족"
  ]];
}
