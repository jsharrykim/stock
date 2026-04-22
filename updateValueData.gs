var USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";
var FINVIZ_RATE_LIMITED = "__RATE_LIMITED__";

// Finviz에서 재무 데이터를 제공하지 않는 ETF/ETP 목록 (요청 생략)
var US_NO_DATA_TICKERS = new Set(["SOXL", "TSLL", "CONL", "ETHU", "SOLT", "HOOG"]);

function updateValueData() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("가치분석");
  var tickers = sheet.getRange("A3:A").getValues().flat().filter(String);
  var startColumn = 8;
  var dataRangeLength = 19;
  var consecutiveBlocks = 0;

  // 기존 데이터 일괄 읽기 (실패 시 덮어쓰기 방지용)
  var existingData = sheet.getRange(3, startColumn, tickers.length, dataRangeLength).getValues();

  // 실패 시 기존 데이터가 있으면 유지, 없으면 "-" 기록
  function writeOnFailure(index, ticker) {
    var row = existingData[index];
    var hasData = row && row.some(v => v !== "" && v !== "-");
    if (hasData) {
      Logger.log(`실패: ${ticker} → 기존 데이터 유지`);
    } else {
      Logger.log(`실패: ${ticker} → 기존 데이터 없음, '-' 기록`);
      sheet.getRange(index + 3, startColumn, 1, dataRangeLength).setValues([Array(dataRangeLength).fill("-")]);
    }
  }

  Logger.log("총 종목 개수: " + tickers.length);
  tickers.forEach((ticker, index) => {
    var resolved = resolveToTicker(ticker);
    var stockData;

    if (resolved.type === "KR") {
      Logger.log(`데이터 가져오는 중: ${ticker} (${resolved.code}) | 네이버 금융`);
      stockData = fetchKoreanStockData(resolved.code);

      if (stockData && stockData.length > 0) {
        Logger.log(`추출된 데이터: ${ticker} → ${stockData}`);
        sheet.getRange(index + 3, startColumn, 1, dataRangeLength).setValues([stockData]);
      } else {
        writeOnFailure(index, ticker);
      }
      Utilities.sleep(500);

    } else if (resolved.type === "US") {
      if (US_NO_DATA_TICKERS.has(resolved.code)) {
        Logger.log(`[스킵] ${ticker} → ETF/ETP, Finviz 요청 생략`);
        writeOnFailure(index, ticker);
      } else {
        var url = `https://finviz.com/quote.ashx?t=${resolved.code}&p=d`;
        Logger.log(`데이터 가져오는 중: ${ticker} | URL: ${url}`);
        stockData = fetchStockDataForValueAnalysis(url);

        if (stockData === FINVIZ_RATE_LIMITED) {
          // 429 차단 → 6초 대기 후 같은 종목 1회 재시도
          Logger.log(`[429 차단] ${ticker} → 6초 대기 후 재시도`);
          Utilities.sleep(6000);
          stockData = fetchStockDataForValueAnalysis(url);

          if (stockData && stockData !== FINVIZ_RATE_LIMITED && stockData.length > 0) {
            Logger.log(`[재시도 성공] ${ticker} → ${stockData}`);
            sheet.getRange(index + 3, startColumn, 1, dataRangeLength).setValues([stockData]);
            consecutiveBlocks = 0;
          } else {
            Logger.log(`[재시도 실패] ${ticker}`);
            consecutiveBlocks++;
            writeOnFailure(index, ticker);
            if (consecutiveBlocks >= 3) {
              Logger.log(`[연속 차단 ${consecutiveBlocks}회 감지] 12초 추가 대기 후 재개`);
              Utilities.sleep(12000);
              consecutiveBlocks = 0;
            }
          }
        } else if (stockData && stockData.length > 0) {
          Logger.log(`추출된 데이터: ${ticker} → ${stockData}`);
          sheet.getRange(index + 3, startColumn, 1, dataRangeLength).setValues([stockData]);
          consecutiveBlocks = 0;
        } else {
          writeOnFailure(index, ticker);
          consecutiveBlocks = 0;
        }
        Utilities.sleep(2000);
      }

    } else {
      Logger.log(`실패: "${ticker}" → 종목 코드를 찾지 못했습니다.`);
      writeOnFailure(index, ticker);
    }
  });
  Logger.log("업데이트 완료!");
}

// ──────────────────────────────────────────────
// 종목명/티커 → { type, code, name } 변환
// ──────────────────────────────────────────────

function resolveToTicker(input) {
  var trimmed = String(input).trim();
  if (/^\d{6}$/.test(trimmed)) return { type: "KR", code: trimmed, name: null };
  if (/^[A-Za-z][A-Za-z0-9.\-]*$/.test(trimmed)) return { type: "US", code: trimmed.toUpperCase(), name: trimmed.toUpperCase() };
  var result = searchNaverStockCode(trimmed);
  if (result) return { type: "KR", code: result.code, name: result.name };
  Logger.log(`[WARN] "${trimmed}" 종목 코드를 찾지 못했습니다.`);
  return { type: "UNKNOWN", code: trimmed, name: null };
}

function searchNaverStockCode(name) {
  try {
    var url = `https://ac.stock.naver.com/ac?q=${encodeURIComponent(name)}&target=stock`;
    var res = UrlFetchApp.fetch(url, { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT } });
    if (res.getResponseCode() !== 200) return null;
    var json = JSON.parse(res.getContentText());
    if (!json.items || json.items.length === 0) { Logger.log(`[KR Search] 검색 결과 없음: "${name}"`); return null; }
    var korItems = json.items.filter(i => i.nationCode === "KOR");
    if (korItems.length === 0) return null;
    var matched = korItems.find(i => i.name === name) || korItems[0];
    Logger.log(`[KR Search] "${name}" → ${matched.name} (${matched.code})`);
    return { code: matched.code, name: matched.name };
  } catch (e) { Logger.log(`[KR Search] 오류: ${e}`); return null; }
}

// ──────────────────────────────────────────────
// 한국 주식 데이터 수집 (네이버 금융)
// ──────────────────────────────────────────────

function fetchKoreanStockData(ticker) {
  try {
    var url = `https://finance.naver.com/item/main.naver?code=${ticker}`;
    var res = UrlFetchApp.fetch(url, { muteHttpExceptions: true, headers: { "User-Agent": USER_AGENT } });
    if (res.getResponseCode() !== 200) return null;
    var html = res.getContentText("UTF-8");

    var tableMatch = html.match(/기업실적분석([\s\S]*?)동종업종비교/);
    if (!tableMatch) { Logger.log(`[KR] ${ticker} 기업실적분석 없음`); return null; }
    var tableHtml = tableMatch[1];

    var salesData = extractRowValues(tableHtml, "매출액");
    var operData  = extractRowValues(tableHtml, "영업이익");
    var epsData   = extractRowValues(tableHtml, "EPS\\(원\\)");
    var roeData   = extractRowValues(tableHtml, "ROE\\(지배주주\\)");
    var debtData  = extractRowValues(tableHtml, "부채비율");
    var quickData = extractRowValues(tableHtml, "당좌비율");

    var isNum = v => typeof v === "number";
    var qtSales   = [salesData[4], salesData[5], salesData[6], salesData[7], salesData[8]].filter(isNum);
    var annSales  = [salesData[0], salesData[1], salesData[2]].filter(isNum);
    var qtOper    = [operData[4], operData[5], operData[6], operData[7], operData[8]].filter(isNum);
    var qtEps     = [epsData[4], epsData[5], epsData[6], epsData[7], epsData[8]].filter(isNum);
    var annEps    = [epsData[0], epsData[1], epsData[2]].filter(isNum);
    var epsNextRaw = (epsData.length > 3 && typeof epsData[3] === "number") ? epsData[3] : null;
    var qtRoe     = [roeData[4], roeData[5], roeData[6], roeData[7], roeData[8]].filter(isNum);
    var annRoe    = [roeData[0], roeData[1], roeData[2]].filter(isNum);
    var qtDebt    = [debtData[4], debtData[5], debtData[6], debtData[7], debtData[8]].filter(isNum);
    var qtQuick   = [quickData[4], quickData[5], quickData[6], quickData[7], quickData[8]].filter(isNum);

    var per = "-";
    var pbr = "-";
    var perMatch = html.match(/<em id="_per">([\d,.]+)<\/em>/);
    var pbrMatch = html.match(/<em id="_pbr">([\d,.]+)<\/em>/);
    if (perMatch) {
      var perVal = parseFloat(perMatch[1].replace(/,/g, ""));
      per = (isNaN(perVal) || perVal < 0) ? "-" : perVal.toFixed(2);
    }
    if (pbrMatch) {
      var pbrVal = parseFloat(pbrMatch[1].replace(/,/g, ""));
      pbr = (isNaN(pbrVal) || pbrVal < 0) ? "-" : pbrVal.toFixed(2);
    }

    var marketCap = "-";
    var mcBillion = 0;
    var mcMatch = html.match(/<em id="_market_sum">([\s\S]*?)<\/em>억원/);
    if (mcMatch) {
      var mcRaw = mcMatch[1].replace(/<[^>]+>/g, "").replace(/\s/g, "");
      var joMatch  = mcRaw.match(/([\d,]+)조([\d,]+)?/);
      var eokMatch = mcRaw.match(/^([\d,]+)$/);
      if (joMatch) {
        var jo  = parseFloat(joMatch[1].replace(/,/g, "")) * 10000;
        var eok = joMatch[2] ? parseFloat(joMatch[2].replace(/,/g, "")) : 0;
        mcBillion = jo + eok;
      } else if (eokMatch) {
        mcBillion = parseFloat(eokMatch[1].replace(/,/g, ""));
      }
      marketCap = formatBillionWon(mcBillion);
    }

    var sharesOutstanding = "-";
    var sharesMatch = html.match(/상장주식수<\/th>\s*<td[^>]*><em>([\d,]+)<\/em><\/td>/);
    if (sharesMatch) sharesOutstanding = sharesMatch[1];

    var salesTTM = "-";
    var salesTTMBillion = 0;
    if (qtSales.length >= 4) {
      salesTTMBillion = qtSales.slice(-4).reduce((a, b) => a + b, 0);
      salesTTM = formatBillionWon(salesTTMBillion);
    } else if (annSales.length > 0) {
      salesTTMBillion = annSales[annSales.length - 1];
      salesTTM = formatBillionWon(salesTTMBillion);
    }

    var salesQQ = "-";
    if (qtSales.length >= 2) {
      var qp = qtSales[qtSales.length - 2], qc = qtSales[qtSales.length - 1];
      if (qp && qp !== 0) salesQQ = pctNoPlus(qc, qp);
    }

    var salesYY = "-";
    if (qtSales.length >= 5) {
      salesYY = pctNoPlus(qtSales[qtSales.length - 1], qtSales[qtSales.length - 5]);
    } else if (annSales.length >= 2) {
      salesYY = pctNoPlus(annSales[annSales.length - 1], annSales[annSales.length - 2]);
    }

    var salesPast35Y = "-";
    if (annSales.length >= 2) {
      var a0 = annSales[annSales.length - 2], a1 = annSales[annSales.length - 1];
      if (a0 && a0 !== 0) salesPast35Y = pctNoPlus(a1, a0) + " / -";
    }

    var currentRatio = qtQuick.length > 0 ? qtQuick[qtQuick.length - 1].toFixed(2) + "%" : "-";
    var de = qtDebt.length > 0 ? qtDebt[qtDebt.length - 1].toFixed(2) + "%" : "-";
    var pfcf = "-";

    var ps = "-";
    if (mcBillion > 0 && salesTTMBillion > 0) {
      ps = (mcBillion / salesTTMBillion).toFixed(2);
    }

    var roe = "-";
    if (qtRoe.length > 0) {
      roe = qtRoe[qtRoe.length - 1].toFixed(2) + "%";
    } else if (annRoe.length > 0) {
      roe = annRoe[annRoe.length - 1].toFixed(2) + "%";
    }

    var epsTTMValue = null;
    var epsTTM = "-";
    if (qtEps.length >= 4) {
      epsTTMValue = qtEps.slice(-4).reduce((a, b) => a + b, 0);
      epsTTM = "₩" + Math.round(epsTTMValue).toLocaleString("ko-KR");
    } else if (annEps.length > 0) {
      epsTTMValue = annEps[annEps.length - 1];
      epsTTM = "₩" + Math.round(epsTTMValue).toLocaleString("ko-KR");
    }

    var epsNextY = (epsNextRaw !== null && epsNextRaw !== undefined)
      ? "₩" + Math.round(epsNextRaw).toLocaleString("ko-KR")
      : "-";

    var epsQQ = "-";
    if (qtEps.length >= 2) {
      var ep = qtEps[qtEps.length - 2], ec = qtEps[qtEps.length - 1];
      if (ep && ep !== 0) epsQQ = pctNoPlus(ec, ep);
    }

    var peg = "-";
    if (per !== "-" && qtEps.length >= 5) {
      var epsYoY = qtEps[qtEps.length - 5] !== 0
        ? ((qtEps[qtEps.length - 1] - qtEps[qtEps.length - 5]) / Math.abs(qtEps[qtEps.length - 5])) * 100
        : null;
      if (epsYoY !== null && epsYoY > 0) {
        var pegVal = parseFloat(per) / epsYoY;
        peg = (pegVal < 0) ? "-" : pegVal.toFixed(2);
      }
    }

    var operMargin = "-";
    if (qtOper.length >= 4 && salesTTMBillion > 0) {
      var operTTM = qtOper.slice(-4).reduce((a, b) => a + b, 0);
      operMargin = ((operTTM / salesTTMBillion) * 100).toFixed(2) + "%";
    } else if (qtOper.length > 0 && qtSales.length > 0) {
      var latestOper  = qtOper[qtOper.length - 1];
      var latestSales = qtSales[qtSales.length - 1];
      if (latestSales > 0) operMargin = ((latestOper / latestSales) * 100).toFixed(2) + "%";
    }

    var grossMargin = "-";

    return [
      marketCap, salesTTM, salesQQ, salesYY, salesPast35Y,
      currentRatio, de, pfcf, ps, per,
      pbr, roe, peg, sharesOutstanding, grossMargin,
      operMargin, epsTTM, epsNextY, epsQQ
    ];

  } catch (e) {
    Logger.log(`오류 발생: ${ticker} | 오류 내용: ${e.toString()}`);
    return null;
  }
}

// ──────────────────────────────────────────────
// 파싱 헬퍼
// ──────────────────────────────────────────────

function extractRowValues(tableHtml, rowLabel) {
  var thRegex = new RegExp(
    "<th[^>]*>(?:(?!<\\/th>)[\\s\\S])*?" + rowLabel + "(?:(?!<\\/th>)[\\s\\S])*?<\\/th>([\\s\\S]*?)(?=<th|<\\/tbody)"
  );
  var rowMatch = tableHtml.match(thRegex);
  if (!rowMatch) return [];

  var rowContent = rowMatch[1];
  var tdRegex = /<td[^>]*>([\s\S]*?)<\/td>/g;
  var nums = [];
  var m;
  while ((m = tdRegex.exec(rowContent)) !== null) {
    var clean = m[1].replace(/<[^>]+>/g, "").trim().replace(/,/g, "");
    var numMatch = clean.match(/^-?[\d]+\.?[\d]*$/);
    if (numMatch) {
      nums.push(parseFloat(clean));
    } else {
      nums.push(null);
    }
  }
  return nums;
}

function pct(curr, prev) {
  var r = ((curr - prev) / Math.abs(prev) * 100);
  return (r >= 0 ? "+" : "") + r.toFixed(2) + "%";
}

function pctNoPlus(curr, prev) {
  var r = ((curr - prev) / Math.abs(prev) * 100);
  return r.toFixed(2) + "%";
}

function formatBillionWon(value) {
  if (value === null || value === undefined || isNaN(value)) return "-";
  var v = Math.round(value);
  if (v >= 10000) {
    var jo  = Math.floor(v / 10000);
    var eok = v % 10000;
    return eok > 0
      ? `${jo.toLocaleString("ko-KR")}조 ${eok.toLocaleString("ko-KR")}억`
      : `${jo.toLocaleString("ko-KR")}조`;
  }
  return `${v.toLocaleString("ko-KR")}억`;
}

// ──────────────────────────────────────────────
// 미국 주식 데이터 수집 (finviz)
// ──────────────────────────────────────────────

function fetchStockDataForValueAnalysis(url) {
  try {
    var response = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/"
      }
    });
    var responseCode = response.getResponseCode();

    if (responseCode === 403 || responseCode === 429) {
      Logger.log(`경고: Finviz에서 요청을 차단했습니다. ${url}. 응답 코드: ${responseCode}`);
      return FINVIZ_RATE_LIMITED;
    }

    var html = response.getContentText();

    var td = '<\\/div><\\/td><td[^>]*><div[^>]*><b>([\\s\\S]*?)<\\/b>';
    var regexPatterns = {
      "MarketCap":         new RegExp("Market Cap" + td),
      "Sales":             new RegExp("(?<![\\w\\/])Sales" + td),
      "SalesQQ":           new RegExp("Sales Q\\/Q" + td),
      "SalesYY":           new RegExp("Sales Y\\/Y TTM" + td),
      "SalesPast35Y":      new RegExp("Sales past 3\\/5Y" + td),
      "CurrentRatio":      new RegExp("Current Ratio" + td),
      "DE":                new RegExp("Debt\\/Eq" + td),
      "PFCF":              new RegExp("P\\/FCF" + td),
      "PS":                new RegExp("P\\/S" + td),
      "PER":               new RegExp("P\\/E" + td),
      "PBR":               new RegExp("P\\/B" + td),
      "ROE":               new RegExp("ROE" + td),
      "PEG":               new RegExp("PEG" + td),
      "SharesOutstanding": new RegExp("Shs Outstand" + td),
      "GrossMargin":       new RegExp("Gross Margin" + td),
      "OperMargin":        new RegExp("Oper\\. Margin" + td),
      "EPS_TTM":           new RegExp("EPS \\(ttm\\)" + td),
      "EPS_Next_Y":        new RegExp("EPS next Y" + td),
      "EPS_QQ":            new RegExp("EPS Q\\/Q" + td)
    };

    var keysToExtract = [
      "MarketCap", "Sales", "SalesQQ", "SalesYY", "SalesPast35Y", "CurrentRatio",
      "DE", "PFCF", "PS", "PER", "PBR", "ROE", "PEG", "SharesOutstanding",
      "GrossMargin", "OperMargin", "EPS_TTM", "EPS_Next_Y", "EPS_QQ"
    ];

    var extractedData = [];
    var tempExtracted = {};
    keysToExtract.forEach(key => {
      var match = html.match(regexPatterns[key]);
      if (match) {
        let value = match[1].replace(/<.*?>/g, "").trim();
        tempExtracted[key] = value;
      } else {
        tempExtracted[key] = "-";
        Logger.log(`'${key}'에 대한 매칭을 찾지 못했습니다.`);
      }
    });

    keysToExtract.forEach(key => {
      let value = tempExtracted[key];
      if (key === "Sales" || key === "SharesOutstanding" || key === "MarketCap") {
        var numMatch = value.match(/([\d\.]+)(B|M|K)?/);
        if (numMatch) {
          var num = parseFloat(numMatch[1]);
          var unit = numMatch[2] || "";
          var actualValue;
          if (unit === "B") actualValue = num * 1000000000;
          else if (unit === "M") actualValue = num * 1000000;
          else if (unit === "K") actualValue = num * 1000;
          else actualValue = num;

          var formattedNum = actualValue.toLocaleString('en-US', { maximumFractionDigits: 0 });
          if (key === "Sales" || key === "MarketCap") {
            extractedData.push(`$${formattedNum} (${value})`);
          } else {
            extractedData.push(`${formattedNum} (${value})`);
          }
        } else {
          extractedData.push(value);
        }
      } else if (key === "SalesPast35Y") {
        var past35YMatch = value.match(/([-\d\.]+%) ([-?\d\.]+%)/);
        if (past35YMatch && past35YMatch.length === 3) {
          extractedData.push(`${past35YMatch[1]} / ${past35YMatch[2]}`);
        } else {
          extractedData.push(value);
        }
      } else {
        extractedData.push(value);
      }
    });
    return extractedData;
  } catch (e) {
    Logger.log(`오류 발생: ${url} | 오류 내용: ${e.toString()}`);
    return null;
  }
}

// ──────────────────────────────────────────────
// 유틸리티
// ──────────────────────────────────────────────

function parseValueWithUnit(value) {
  if (!value) return "-";
  var numMatch = value.match(/([\d\.]+)(B|M|K)?/);
  if (!numMatch) return "-";

  var num = parseFloat(numMatch[1]);
  var unit = numMatch[2] || "";

  if (unit === "B") return (num * 1000000000).toFixed(0);
  if (unit === "M") return (num * 1000000).toFixed(0);
  if (unit === "K") return (num * 1000).toFixed(0);

  return num.toString();
}

function deleteExistingTriggers() {
  var triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => {
    if (trigger.getHandlerFunction() === "updateValueData") {
      ScriptApp.deleteTrigger(trigger);
      Logger.log("기존 트리거 삭제됨: " + trigger.getHandlerFunction());
    }
  });
}

// ──────────────────────────────────────────────
// 테스트 함수
// ──────────────────────────────────────────────

function testKoreanStock() {
  var ticker = "042700";
  var result = fetchKoreanStockData(ticker);
  if (result) {
    var labels = [
      "시가총액","매출(TTM)","Sales Q/Q","Sales Y/Y","Sales Past 3/5Y",
      "유동비율(당좌)","부채비율(D/E)","P/FCF","P/S","PER",
      "PBR","ROE","PEG","발행주식수","매출총이익률",
      "영업이익률(TTM)","EPS(TTM)","EPS(Next Y)","EPS Q/Q"
    ];
    labels.forEach((l, i) => Logger.log(`  [${i+1}] ${l}: ${result[i]}`));
  }
}
