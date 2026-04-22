var USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36";

var EARNINGS_CACHE_PREFIX = "EARNINGS_CACHE_V1_";
var URLFETCH_QUOTA_EXCEEDED = "__URLFETCH_QUOTA_EXCEEDED__";

var EARNINGS_CACHE_TTL_MS = {
  confirmed: 24 * 60 * 60 * 1000,
  undecided: 12 * 60 * 60 * 1000,
  noData: 6 * 60 * 60 * 1000
};

function _buildEarningsCacheKey_(type, code) {
  return EARNINGS_CACHE_PREFIX + type + "_" + code;
}

function _isUrlFetchQuotaError_(e) {
  var msg = String(e || "");
  return /urlfetch/i.test(msg) && (/too many times|quota/i.test(msg) || /너무 많이 호출/.test(msg));
}

function _getEarningsCacheTTL_(rawDate) {
  if (!rawDate) return 0;
  if (rawDate === "미정") return EARNINGS_CACHE_TTL_MS.undecided;
  if (rawDate === "데이터 없음") return EARNINGS_CACHE_TTL_MS.noData;
  if (rawDate === "오류" || rawDate === URLFETCH_QUOTA_EXCEEDED) return 0;
  return EARNINGS_CACHE_TTL_MS.confirmed;
}

function _loadEarningsCache_(type, code) {
  try {
    var raw = PropertiesService.getScriptProperties().getProperty(_buildEarningsCacheKey_(type, code));
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    console.log("[실적 캐시 파싱 오류] " + type + " " + code + ": " + e.toString());
    return null;
  }
}

function _loadFreshEarningsCache_(type, code) {
  var cache = _loadEarningsCache_(type, code);
  if (!cache || !cache.rawDate || !cache.savedAt || !cache.ttlMs) return null;
  if ((Date.now() - cache.savedAt) > cache.ttlMs) return null;
  return cache;
}

function _saveEarningsCache_(type, code, rawDate) {
  var ttlMs = _getEarningsCacheTTL_(rawDate);
  if (ttlMs <= 0) return;
  try {
    PropertiesService.getScriptProperties().setProperty(
      _buildEarningsCacheKey_(type, code),
      JSON.stringify({
        rawDate: rawDate,
        savedAt: Date.now(),
        ttlMs: ttlMs
      })
    );
  } catch (e) {
    console.log("[실적 캐시 저장 오류] " + type + " " + code + ": " + e.toString());
  }
}

function _formatExistingCellValue_(v) {
  if (v === null || v === undefined || v === "") return "-";
  return String(v).trim() || "-";
}

function _getFallbackDisplayValue_(type, cache, existingValue, koreaToday) {
  if (cache && cache.rawDate) {
    if (type === "KR" && cache.rawDate !== "미정" && cache.rawDate !== "데이터 없음" && cache.rawDate !== "오류" && cache.rawDate !== URLFETCH_QUOTA_EXCEEDED) {
      return processKoreanEarningsDate(cache.rawDate, koreaToday);
    }
    if (type === "US" && cache.rawDate !== "데이터 없음" && cache.rawDate !== "오류" && cache.rawDate !== "티커 없음" && cache.rawDate !== URLFETCH_QUOTA_EXCEEDED) {
      return processAndFormatDate(cache.rawDate, koreaToday);
    }
    if (cache.rawDate === "미정" || cache.rawDate === "데이터 없음") {
      return "-";
    }
  }
  return existingValue || "-";
}

function _parseUSEarningsDateFromHtml_(html) {
  if (!html) return "데이터 없음";

  var match1 = html.match(/Earnings Date<\/td><td[^>]*><span class="snapshot-td2">([A-Z][a-z]{2} \d{1,2} (?:BMO|AMC))<\/span>/);
  if (match1) return match1[1];

  var earningsIdx = html.indexOf("Earnings");
  if (earningsIdx > -1) {
    var section = html.substring(earningsIdx, earningsIdx + 400);
    var match2 = section.match(/[A-Z][a-z]{2} \d{1,2} (?:BMO|AMC)/);
    if (match2) return match2[0];
  }

  var match3 = html.match(/[A-Z][a-z]{2} \d{1,2} (?:BMO|AMC)/);
  return match3 ? match3[0] : "데이터 없음";
}

// ──────────────────────────────────────────────
// 진단 함수: 실제 HTML 구조 확인용 (파싱 실패 시 실행)
// DIAGNOSE_CODE 값을 바꿔가며 실행 → 로그에서 HTML 구조 확인
// ──────────────────────────────────────────────
var DIAGNOSE_CODE = "000660";

function diagnoseKrEarnings() {
  console.log("===== 진단 시작: " + DIAGNOSE_CODE + " =====");

  // ── FnGuide ──
  var fnUrl = "https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A" + DIAGNOSE_CODE + "&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701";
  try {
    var fnRes = UrlFetchApp.fetch(fnUrl, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT, "Referer": "https://comp.fnguide.com", "Accept-Language": "ko-KR,ko;q=0.9" }
    });
    console.log("[FnGuide] HTTP " + fnRes.getResponseCode() + " / 본문 길이: " + fnRes.getContentText("UTF-8").length);
    var fnHtml = fnRes.getContentText("UTF-8");

    // 키워드별 존재 여부
    ["잠정실적발표일", "잠정실적발표예정일", "실적발표일", "실적발표", "미정"].forEach(function(kw) {
      var idx = fnHtml.indexOf(kw);
      console.log("  키워드 '" + kw + "': " + (idx > -1 ? "발견 (위치 " + idx + ")" : "없음"));
      if (idx > -1) console.log("    → 전후 200자: " + fnHtml.substring(Math.max(0, idx - 50), idx + 200).replace(/\s+/g, " "));
    });

    // 앞 1000자 출력 (페이지가 정상인지 확인)
    console.log("[FnGuide 앞 1000자]\n" + fnHtml.substring(0, 1000).replace(/\s+/g, " "));
  } catch (e) {
    console.log("[FnGuide 오류] " + e.toString());
  }

  Utilities.sleep(1000);

  // ── 네이버 메인 ──
  var naverUrl = "https://finance.naver.com/item/main.naver?code=" + DIAGNOSE_CODE;
  try {
    var naverRes = UrlFetchApp.fetch(naverUrl, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT }
    });
    console.log("[네이버 메인] HTTP " + naverRes.getResponseCode() + " / 본문 길이: " + naverRes.getContentText("UTF-8").length);
    var naverHtml = naverRes.getContentText("UTF-8");

    ["실적발표", "잠정실적", "발표예정", "공시예정"].forEach(function(kw) {
      var idx = naverHtml.indexOf(kw);
      console.log("  키워드 '" + kw + "': " + (idx > -1 ? "발견 (위치 " + idx + ")" : "없음"));
      if (idx > -1) console.log("    → 전후 200자: " + naverHtml.substring(Math.max(0, idx - 50), idx + 200).replace(/\s+/g, " "));
    });

    // 날짜 패턴 직접 탐색
    var allDates = naverHtml.match(/\d{4}\.\d{2}\.\d{2}/g);
    console.log("  [네이버 메인] 전체 날짜 패턴 발견: " + (allDates ? allDates.join(", ") : "없음"));
  } catch (e) {
    console.log("[네이버 메인 오류] " + e.toString());
  }

  Utilities.sleep(1000);

  // ── 네이버 기업정보 ──
  var coUrl = "https://finance.naver.com/item/coinfo.naver?code=" + DIAGNOSE_CODE;
  try {
    var coRes = UrlFetchApp.fetch(coUrl, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT }
    });
    console.log("[네이버 기업정보] HTTP " + coRes.getResponseCode() + " / 본문 길이: " + coRes.getContentText("UTF-8").length);
    var coHtml = coRes.getContentText("UTF-8");

    ["실적발표", "잠정실적", "발표예정", "공시예정"].forEach(function(kw) {
      var idx = coHtml.indexOf(kw);
      console.log("  키워드 '" + kw + "': " + (idx > -1 ? "발견 (위치 " + idx + ")" : "없음"));
      if (idx > -1) console.log("    → 전후 200자: " + coHtml.substring(Math.max(0, idx - 50), idx + 200).replace(/\s+/g, " "));
    });

    var allDates2 = coHtml.match(/\d{4}\.\d{2}\.\d{2}/g);
    console.log("  [네이버 기업정보] 전체 날짜 패턴 발견: " + (allDates2 ? allDates2.join(", ") : "없음"));
  } catch (e) {
    console.log("[네이버 기업정보 오류] " + e.toString());
  }

  console.log("===== 진단 종료 =====");
}

function updateEarningsDates() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  const tickers = sheet.getRange("A3:A").getValues().flat().filter(String);
  const headerRow = sheet.getRange("1:1").getValues()[0];
  const targetColumnIndex = headerRow.indexOf("실적발표일 (한국 시간 기준)") + 1;

  if (targetColumnIndex === 0) {
    console.log("오류: '실적발표일 (한국 시간 기준)' 헤더를 찾을 수 없습니다.");
    return;
  }

  console.log("--- 실적 발표일 업데이트 실행 시작 ---");
  console.log("총 종목 개수: " + tickers.length);

  const koreaTimeZone = "Asia/Seoul";
  const koreaToday = Utilities.formatDate(new Date(), koreaTimeZone, "yyyy-MM-dd");
  console.log(`[한국 현재 날짜] ${koreaToday}`);

  const existingValues = sheet.getRange(3, targetColumnIndex, tickers.length, 1).getDisplayValues();
  const results = existingValues.map(function(row) {
    return [_formatExistingCellValue_(row[0])];
  });
  const ETF_LIST = ["SOXL", "TSLL", "TQQQ", "SQQQ", "UVXY", "UPRO", "SPXL", "FNGU"];

  let quotaExceeded = false;

  for (let index = 0; index < tickers.length; index++) {
    if (quotaExceeded) break;

    const ticker = tickers[index];
    let formattedDate = "-";
    const resolved = resolveToTickerForRSI(ticker);
    const existingValue = _formatExistingCellValue_(existingValues[index][0]);

    if (resolved.type === "KR") {
      const freshCache = _loadFreshEarningsCache_("KR", resolved.code);
      if (freshCache) {
        formattedDate = _getFallbackDisplayValue_("KR", freshCache, existingValue, koreaToday);
        results[index] = [formattedDate];
        console.log(`[KR 캐시 사용] ${ticker} (${resolved.code}): ${freshCache.rawDate} → ${formattedDate}`);
        continue;
      }

      const rawDate = fetchKoreanEarningsDate(resolved.code);
      if (rawDate === URLFETCH_QUOTA_EXCEEDED) {
        quotaExceeded = true;
        const anyCache = _loadEarningsCache_("KR", resolved.code);
        formattedDate = _getFallbackDisplayValue_("KR", anyCache, existingValue, koreaToday);
        results[index] = [formattedDate];
        console.log(`[KR 쿼터 초과] ${ticker} (${resolved.code}) → 기존값/캐시 유지: ${formattedDate}`);
        break;
      }

      if (rawDate && rawDate !== "미정" && rawDate !== "데이터 없음" && rawDate !== "오류") {
        formattedDate = processKoreanEarningsDate(rawDate, koreaToday);
        _saveEarningsCache_("KR", resolved.code, rawDate);
        console.log(`[KR 성공] ${ticker} (${resolved.code}): ${rawDate} → ${formattedDate}`);
      } else if (rawDate === "미정" || rawDate === "데이터 없음") {
        formattedDate = "-";
        _saveEarningsCache_("KR", resolved.code, rawDate);
        console.log(`[KR ${rawDate}] ${ticker} (${resolved.code})`);
      } else {
        const anyCache = _loadEarningsCache_("KR", resolved.code);
        formattedDate = _getFallbackDisplayValue_("KR", anyCache, existingValue, koreaToday);
        console.log(`[KR 실패 → 기존값/캐시 유지] ${ticker} (${resolved.code}): ${formattedDate}`);
      }
      results[index] = [formattedDate];
      Utilities.sleep(1500);

    } else if (resolved.type === "US") {
      if (ETF_LIST.includes(resolved.code)) {
        console.log(`[US ETF] ${ticker}: 실적발표일 없음 (ETF 스킵)`);
        results[index] = ["-"];
        Utilities.sleep(300);
        continue;
      }

      const freshCache = _loadFreshEarningsCache_("US", resolved.code);
      if (freshCache) {
        formattedDate = _getFallbackDisplayValue_("US", freshCache, existingValue, koreaToday);
        results[index] = [formattedDate];
        console.log(`[US 캐시 사용] ${ticker}: ${freshCache.rawDate} → ${formattedDate}`);
        continue;
      }

      const rawDate = fetchEarningsDate(resolved.code);
      if (rawDate === URLFETCH_QUOTA_EXCEEDED) {
        quotaExceeded = true;
        const anyCache = _loadEarningsCache_("US", resolved.code);
        formattedDate = _getFallbackDisplayValue_("US", anyCache, existingValue, koreaToday);
        results[index] = [formattedDate];
        console.log(`[US 쿼터 초과] ${ticker} → 기존값/캐시 유지: ${formattedDate}`);
        break;
      }

      if (rawDate !== "오류" && rawDate !== "데이터 없음" && rawDate !== "티커 없음") {
        formattedDate = processAndFormatDate(rawDate, koreaToday);
        _saveEarningsCache_("US", resolved.code, rawDate);
        console.log(`[US 성공] ${ticker}: ${rawDate} → ${formattedDate}`);
        results[index] = [formattedDate];
      } else if (rawDate === "데이터 없음") {
        _saveEarningsCache_("US", resolved.code, rawDate);
        results[index] = ["-"];
        console.log(`[US 데이터 없음] ${ticker}`);
      } else {
        const anyCache = _loadEarningsCache_("US", resolved.code);
        formattedDate = _getFallbackDisplayValue_("US", anyCache, existingValue, koreaToday);
        results[index] = [formattedDate];
        console.log(`[US 실패 → 기존값/캐시 유지] ${ticker}: ${formattedDate}`);
      }
      Utilities.sleep(1000);

    } else {
      console.log(`[UNKNOWN] "${ticker}" 종목 코드를 찾지 못했습니다.`);
      results[index] = [formattedDate];
    }
  }

  sheet.getRange(3, targetColumnIndex, results.length, 1).setValues(results);
  if (quotaExceeded) {
    console.log("--- 실적 발표일 업데이트 중단: URLFetch 일일 쿼터 초과 ---");
    return;
  }
  console.log("--- 실적 발표일 업데이트 완료 ---");
}

// ──────────────────────────────────────────────
// 미국 주식 실적발표일 (Finviz)
// ──────────────────────────────────────────────

function fetchEarningsDate(ticker) {
  if (!ticker) return "티커 없음";
  const url = `https://finviz.com/quote.ashx?t=${ticker}&p=d`;
  try {
    const response = UrlFetchApp.fetch(url, {
      headers: { "User-Agent": USER_AGENT },
      muteHttpExceptions: true
    });
    if (response.getResponseCode() !== 200) {
      console.log(`[US HTTP ${response.getResponseCode()}] ${ticker}`);
      return "오류";
    }
    return _parseUSEarningsDateFromHtml_(response.getContentText());
  } catch (e) {
    if (_isUrlFetchQuotaError_(e)) {
      console.log(`[US URLFetch 쿼터 초과] ${ticker}: ${e.toString()}`);
      return URLFETCH_QUOTA_EXCEEDED;
    }
    return "오류";
  }
}

function fetchEarningsDateFallback(ticker) {
  return fetchEarningsDate(ticker);
}

function processAndFormatDate(rawDate, koreaTodayString) {
  const monthMap = { "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11 };
  const parts = rawDate.split(" ");
  const month = parts[0];
  const day = parseInt(parts[1], 10);
  const type = parts[2];
  const todayParts = koreaTodayString.split("-");
  const currentYear = parseInt(todayParts[0]);
  const todayMonth = parseInt(todayParts[1]) - 1;
  const todayDay = parseInt(todayParts[2]);
  const koreaTodayDate = new Date(currentYear, todayMonth, todayDay, 0, 0, 0, 0);
  let usDate = new Date(currentYear, monthMap[month], day);
  let koreaEarningsDateTime;
  if (type === "BMO") {
    const usDateTime = new Date(Date.UTC(usDate.getFullYear(), usDate.getMonth(), usDate.getDate(), 13, 0, 0));
    koreaEarningsDateTime = new Date(usDateTime.getTime() + (9 * 60 * 60 * 1000));
  } else {
    const usDateTime = new Date(Date.UTC(usDate.getFullYear(), usDate.getMonth(), usDate.getDate(), 22, 0, 0));
    koreaEarningsDateTime = new Date(usDateTime.getTime() + (9 * 60 * 60 * 1000));
  }
  let koreaEarningsDate = new Date(koreaEarningsDateTime.getFullYear(), koreaEarningsDateTime.getMonth(), koreaEarningsDateTime.getDate(), 0, 0, 0, 0);
  let diffDays = Math.round((koreaEarningsDate.getTime() - koreaTodayDate.getTime()) / (1000 * 60 * 60 * 24));
  if (diffDays > 120) {
    koreaEarningsDate.setFullYear(currentYear - 1);
  } else if (diffDays < -240) {
    koreaEarningsDate.setFullYear(currentYear + 1);
  }
  const finalDiffDays = Math.round((koreaEarningsDate.getTime() - koreaTodayDate.getTime()) / (1000 * 60 * 60 * 24));
  const diffSign = finalDiffDays < 0 ? "+" : "-";
  const absDiff = Math.abs(finalDiffDays);
  const dDayLabel = finalDiffDays === 0 ? "D-0" : `D${diffSign}${absDiff}`;
  const year = koreaEarningsDate.getFullYear();
  const monthFormatted = (koreaEarningsDate.getMonth() + 1).toString().padStart(2, '0');
  const dayFormatted = koreaEarningsDate.getDate().toString().padStart(2, '0');
  return `${year}-${monthFormatted}-${dayFormatted} (${dDayLabel})`;
}

// ──────────────────────────────────────────────
// 한국 주식 실적발표일 (FnGuide 단일 소스)
// 네이버는 JS 렌더링으로 정적 HTML에 실적발표일 없음 — 사용 불가
// ──────────────────────────────────────────────

function fetchKoreanEarningsDate(code) {
  var result = _krFetchFnGuide_(code);
  if (result === URLFETCH_QUOTA_EXCEEDED) {
    console.log("[KR FnGuide 쿼터 초과] " + code);
    return result;
  }
  if (result && result !== "데이터 없음" && result !== "오류") {
    console.log("[KR FnGuide 성공] " + code + ": " + result);
    return result;
  }
  console.log("[KR FnGuide 실패] " + code + ": " + result);
  return result === "미정" ? "미정" : "데이터 없음";
}

function _krFetchFnGuide_(code) {
  var url = "https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A" + code + "&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701";
  try {
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: {
        "User-Agent": USER_AGENT,
        "Referer": "https://comp.fnguide.com",
        "Accept-Language": "ko-KR,ko;q=0.9"
      }
    });
    if (res.getResponseCode() !== 200) {
      console.log("[KR FnGuide HTTP " + res.getResponseCode() + "] " + code);
      return "오류";
    }
    var html = res.getContentText("UTF-8");

    // 진단 결과: 헤더(<th>)에 키워드, 실제 값은 이후 ~326자 뒤 <td class="c clf">에 위치
    // 구조: <th>잠정실적발표예정일</th> ... <tbody><tr><td class="c clf">미정 or 날짜</td>
    var keywords = ["잠정실적발표예정일", "잠정실적발표일", "실적발표예정일", "실적발표일"];
    for (var i = 0; i < keywords.length; i++) {
      var idx = html.indexOf(keywords[i]);
      if (idx < 0) continue;

      // 키워드 이후 600자 탐색 (실제 값이 ~326자 뒤에 있으므로 넉넉하게)
      var section = html.substring(idx, idx + 600);

      // 1순위: <td class="c clf"> 셀의 값 추출 (날짜 or 미정)
      var tdMatch = section.match(/<td[^>]*class="[^"]*clf[^"]*"[^>]*>\s*([\d\/\.\-]+|미정)\s*<\/td>/);
      if (tdMatch) {
        var val = tdMatch[1].trim();
        if (val === "미정") return "미정";
        var dateMatch = val.match(/(\d{4})[\/\.\-](\d{2})[\/\.\-](\d{2})/);
        if (dateMatch) return dateMatch[1] + "/" + dateMatch[2] + "/" + dateMatch[3];
      }

      // 2순위: 섹션 내 날짜 패턴 직접 탐색
      var anyDate = section.match(/(\d{4})[\/\.\-](\d{2})[\/\.\-](\d{2})/);
      if (anyDate) return anyDate[1] + "/" + anyDate[2] + "/" + anyDate[3];

      // 3순위: 미정 텍스트 확인
      if (section.indexOf("미정") > -1) return "미정";
    }
    return "데이터 없음";
  } catch (e) {
    if (_isUrlFetchQuotaError_(e)) {
      console.log("[KR FnGuide 쿼터 초과] " + code + ": " + e.toString());
      return URLFETCH_QUOTA_EXCEEDED;
    }
    console.log("[KR FnGuide 오류] " + code + ": " + e.toString());
    return "오류";
  }
}

function _krFetchNaverMain_(code) {
  try {
    var url = "https://finance.naver.com/item/main.naver?code=" + code;
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT }
    });
    if (res.getResponseCode() !== 200) return "오류";
    return _krExtractDateFromHtml_(res.getContentText("UTF-8"), code, "네이버메인");
  } catch (e) {
    console.log("[KR 네이버 메인 오류] " + code + ": " + e.toString());
    return "오류";
  }
}

function _krFetchNaverCoInfo_(code) {
  try {
    var url = "https://finance.naver.com/item/coinfo.naver?code=" + code;
    var res = UrlFetchApp.fetch(url, {
      muteHttpExceptions: true,
      headers: { "User-Agent": USER_AGENT }
    });
    if (res.getResponseCode() !== 200) return "오류";
    return _krExtractDateFromHtml_(res.getContentText("UTF-8"), code, "네이버기업정보");
  } catch (e) {
    console.log("[KR 네이버 기업정보 오류] " + code + ": " + e.toString());
    return "오류";
  }
}

function _krExtractDateFromHtml_(html, code, sourceName) {
  var keywords = ["실적발표", "잠정실적", "발표예정", "공시예정"];
  for (var i = 0; i < keywords.length; i++) {
    var searchFrom = 0;
    while (true) {
      var idx = html.indexOf(keywords[i], searchFrom);
      if (idx < 0) break;
      var section = html.substring(idx, idx + 600);
      var dateMatch = section.match(/(\d{4})\.(\d{2})\.(\d{2})/);
      if (dateMatch) {
        var result = dateMatch[1] + "/" + dateMatch[2] + "/" + dateMatch[3];
        console.log("[KR " + sourceName + " 파싱 성공] " + code + ": " + result + " (키워드: " + keywords[i] + ")");
        return result;
      }
      searchFrom = idx + keywords[i].length;
    }
  }
  console.log("[KR " + sourceName + " 파싱 실패] " + code + ": 날짜 패턴 없음");
  return "데이터 없음";
}

function processKoreanEarningsDate(rawDate, koreaTodayString) {
  try {
    // "/", "-", "." 모두 처리 (기존에 "-" 형식 입력 시 파싱 버그 수정)
    var normalized = rawDate.replace(/[\.\-]/g, "/");
    var parts = normalized.split("/");
    var year  = parseInt(parts[0]);
    var month = parseInt(parts[1]) - 1;
    var day   = parseInt(parts[2]);

    var earningsDate = new Date(year, month, day);
    var todayParts   = koreaTodayString.split("-");
    var todayDate    = new Date(parseInt(todayParts[0]), parseInt(todayParts[1]) - 1, parseInt(todayParts[2]));

    var diffDays  = Math.round((earningsDate.getTime() - todayDate.getTime()) / (1000 * 60 * 60 * 24));
    var diffSign  = diffDays < 0 ? "+" : "-";
    var absDiff   = Math.abs(diffDays);
    var dDayLabel = diffDays === 0 ? "D-0" : "D" + diffSign + absDiff;

    var monthFormatted = (earningsDate.getMonth() + 1).toString().padStart(2, "0");
    var dayFormatted   = earningsDate.getDate().toString().padStart(2, "0");

    return year + "-" + monthFormatted + "-" + dayFormatted + " (" + dDayLabel + ")";
  } catch (e) {
    console.log("[KR 날짜 포맷 오류] rawDate=" + rawDate + ": " + e.toString());
    return rawDate;
  }
}
