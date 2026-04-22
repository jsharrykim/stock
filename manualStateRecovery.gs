/**
 * 수동 상태 복구용 스크립트
 *
 * 1) diagnoseCurrentSystemHoldings_20260422()
 *    - 기술분석 시트 + Script Properties 기준으로
 *      "시스템이 지금 무엇을 보유로 보고 있는지" 진단 시트를 생성합니다.
 *
 * 2) applySystemStateFromTradingLogSnapshot_20260422()
 *    - 사용자가 제공한 2026-04-22 기준 트레이딩로그 스냅샷을 source of truth 로 보고
 *      기술분석 시트와 Script Properties 상태를 그 스냅샷에 맞춰 정합화합니다.
 *
 * 주의:
 * - 이 스크립트는 "트레이딩로그 이미지가 가장 정확하다"는 전제에서 작성되었습니다.
 * - 멀티 슬롯(SLOT_*) 상태는 이미지에 없으므로 전부 제거합니다.
 */

function diagnoseCurrentSystemHoldings_20260422() {
  const sheet = _manualGetTechSheet_();
  const C = _manualCols_();
  const props = PropertiesService.getScriptProperties().getProperties();
  const lastRow = sheet.getLastRow();
  const lastCol = Math.max(sheet.getLastColumn(), C.entryStrategy + 1);
  const rows = lastRow >= 3 ? sheet.getRange(3, 1, lastRow - 2, lastCol).getValues() : [];

  const output = [[
    "행",
    "종목코드",
    "종목명",
    "표시명",
    "시트 의견",
    "시트 진입가",
    "시트 진입일",
    "시트 진입전략",
    "ENTRY 가격",
    "ENTRY 날짜",
    "ENTRY 전략",
    "SELL 값",
    "활성 SLOT",
    "시스템상 보유",
    "불일치/메모"
  ]];

  const summary = {
    primaryHoldings: [],
    watchHoldings: [],
    buyHoldings: [],
    slotHoldings: [],
    mismatches: [],
    propsOnlyEntries: []
  };

  const seenStocks = {};

  rows.forEach((row, index) => {
    const stockName = String(row[C.stockName] || "").trim();
    if (!stockName) return;

    const stockLabel = String(row[C.stockLabel] || "").trim();
    const displayName = _manualDisplayName_(stockName, stockLabel);
    const opinion = String(row[C.opinion] || "").trim();
    const sheetEntryPrice = _manualToNum_(row[C.entryPrice]);
    const sheetEntryDate = _manualDateString_(row[C.entryDate]);
    const sheetEntryStrategy = String(row[C.entryStrategy] || "").trim();
    const entry = _manualLoadEntry_(stockName, props);
    const sellVal = props["SELL_" + stockName] || "";
    const activeSlots = _manualLoadActiveSlots_(stockName, props);
    const sheetHasEntry = !!sheetEntryPrice || !!sheetEntryDate;
    const systemHolding = entry.price > 0 || activeSlots.length > 0;

    seenStocks[stockName] = true;

    if (entry.price > 0) {
      summary.primaryHoldings.push(`${displayName} | ${entry.strategyType} | ${_manualFmtPrice_(entry.price, stockName)} | ${entry.dateString}`);
      if (opinion === "관망") summary.watchHoldings.push(displayName);
      if (opinion === "매수") summary.buyHoldings.push(displayName);
    }
    if (activeSlots.length > 0) {
      summary.slotHoldings.push(`${displayName} | ${activeSlots.join(",")}`);
    }

    const notes = [];
    if (opinion === "매수" && entry.price <= 0) notes.push("매수인데 ENTRY_ 없음");
    if (opinion === "매도" && entry.price > 0) notes.push("매도인데 ENTRY_ 존재");
    if (sheetHasEntry && entry.price <= 0) notes.push("시트에는 보유 흔적 있으나 ENTRY_ 없음");
    if (!sheetHasEntry && entry.price > 0) notes.push("시트에는 빈칸인데 ENTRY_ 존재");
    if ((opinion === "관망" || opinion === "매수") && sellVal && entry.price <= 0) notes.push("SELL_ 잔존");
    if (activeSlots.length > 0) notes.push("멀티 슬롯 활성");

    if (notes.length > 0) {
      summary.mismatches.push(`${displayName}: ${notes.join(" / ")}`);
    }

    output.push([
      index + 3,
      stockName,
      stockLabel,
      displayName,
      opinion,
      sheetEntryPrice || "",
      sheetEntryDate,
      sheetEntryStrategy,
      entry.price || "",
      entry.dateString,
      entry.strategyType || "",
      sellVal,
      activeSlots.join(", "),
      systemHolding ? "Y" : "",
      notes.join(" / ")
    ]);
  });

  Object.keys(props)
    .filter(function(key) { return key.indexOf("ENTRY_") === 0; })
    .sort()
    .forEach(function(key) {
      const stockName = key.substring("ENTRY_".length);
      if (seenStocks[stockName]) return;
      const entry = _manualLoadEntry_(stockName, props);
      if (entry.price <= 0) return;
      const note = "기술분석 시트에는 없는데 ENTRY_만 존재";
      summary.propsOnlyEntries.push(`${stockName} | ${entry.strategyType} | ${_manualFmtPrice_(entry.price, stockName)} | ${entry.dateString}`);
      summary.mismatches.push(`${stockName}: ${note}`);
      output.push([
        "",
        stockName,
        "",
        stockName,
        "",
        "",
        "",
        "",
        entry.price,
        entry.dateString,
        entry.strategyType || "",
        props["SELL_" + stockName] || "",
        _manualLoadActiveSlots_(stockName, props).join(", "),
        "Y",
        note
      ]);
    });

  const diagSheetName = "시스템보유진단";
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const diagSheet = ss.getSheetByName(diagSheetName) || ss.insertSheet(diagSheetName);
  diagSheet.clearContents();
  diagSheet.clearFormats();
  diagSheet.getRange(1, 1, output.length, output[0].length).setValues(output);
  diagSheet.setFrozenRows(1);
  diagSheet.autoResizeColumns(1, output[0].length);

  console.log("========== 시스템 보유 진단 시작 ==========");
  console.log("[PRIMARY 보유]");
  if (summary.primaryHoldings.length === 0) console.log("없음");
  summary.primaryHoldings.forEach(function(line) { console.log(" - " + line); });

  console.log("[관망 보유]");
  if (summary.watchHoldings.length === 0) console.log("없음");
  summary.watchHoldings.forEach(function(line) { console.log(" - " + line); });

  console.log("[매수 보유]");
  if (summary.buyHoldings.length === 0) console.log("없음");
  summary.buyHoldings.forEach(function(line) { console.log(" - " + line); });

  console.log("[활성 SLOT]");
  if (summary.slotHoldings.length === 0) console.log("없음");
  summary.slotHoldings.forEach(function(line) { console.log(" - " + line); });

  console.log("[불일치]");
  if (summary.mismatches.length === 0) console.log("없음");
  summary.mismatches.forEach(function(line) { console.log(" - " + line); });

  console.log(`[완료] 진단 시트 생성: ${diagSheetName}`);
  console.log("========== 시스템 보유 진단 종료 ==========");
  return summary;
}

function applySystemStateFromTradingLogSnapshot_20260422() {
  const sheet = _manualGetTechSheet_();
  const C = _manualCols_();
  const props = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const logSheet = _manualGetTradingLogSheet_();
  const openPositions = _manualLoadOpenPositionsFromTradingLog_(logSheet);
  const lastRow = sheet.getLastRow();
  const lastCol = Math.max(sheet.getLastColumn(), C.entryStrategy + 1);
  const rows = lastRow >= 3 ? sheet.getRange(3, 1, lastRow - 2, lastCol).getValues() : [];
  const changedLines = [];
  const rowsByStock = {};
  const syncedStocks = {};
  const missingInTechSheet = [];

  rows.forEach(function(row, index) {
    const stockName = String(row[C.stockName] || "").trim();
    if (!stockName) return;
    const stockLabel = String(row[C.stockLabel] || "").trim();
    rowsByStock[stockName] = {
      rowNumber: index + 3,
      stockName: stockName,
      stockLabel: stockLabel,
      displayName: _manualDisplayName_(stockName, stockLabel),
      values: row
    };
  });

  Object.keys(rowsByStock).forEach(function(stockName) {
    const rowInfo = rowsByStock[stockName];
    const open = openPositions[stockName];
    const currentOpinion = String(rowInfo.values[C.opinion] || "").trim();

    if (open) {
      sheet.getRange(rowInfo.rowNumber, C.entryPrice + 1).setValue(open.entryPrice);
      sheet.getRange(rowInfo.rowNumber, C.entryDate + 1).setValue(_manualDateObject_(open.entryDate));
      sheet.getRange(rowInfo.rowNumber, C.entryStrategy + 1).setValue(open.strategyLabel);

      if (!currentOpinion || currentOpinion === "매도") {
        sheet.getRange(rowInfo.rowNumber, C.opinion + 1).setValue("관망");
      }

      _manualClearPrimaryTradingState_(stockName, props, allProps);
      _manualSaveEntry_(stockName, open.entryPrice, open.entryDate, open.strategyType, props);
      props.deleteProperty("SELL_" + stockName);
      props.deleteProperty("EXIT_REASON_" + stockName);
      props.deleteProperty("SOLD_FLAG_" + stockName);
      props.setProperty("CYCLE_ENTRY_" + stockName, String(open.entryPrice));
      props.setProperty("REENTRY_COUNT_" + stockName, "0");

      syncedStocks[stockName] = true;
      changedLines.push(`${rowInfo.displayName}: 트레이딩로그 미청산 행 반영 / ENTRY_${stockName}=${open.entryPrice}|${open.entryDate}|${open.strategyType}`);
      return;
    }

    if (!_manualHasPositionTrace_(rowInfo, allProps, C)) return;

    if (currentOpinion === "매수") {
      sheet.getRange(rowInfo.rowNumber, C.opinion + 1).setValue("관망");
    }
    sheet.getRange(rowInfo.rowNumber, C.entryPrice + 1).clearContent();
    sheet.getRange(rowInfo.rowNumber, C.entryDate + 1).clearContent();
    sheet.getRange(rowInfo.rowNumber, C.entryStrategy + 1).clearContent();
    _manualClearPrimaryTradingState_(stockName, props, allProps);
    changedLines.push(`${rowInfo.displayName}: 트레이딩로그 미청산 없음 / 시트·ENTRY_ 정리`);
  });

  Object.keys(openPositions).forEach(function(stockName) {
    if (syncedStocks[stockName]) return;
    missingInTechSheet.push(`${stockName}: 기술분석 시트에 없음`);
  });

  _manualClearAllSlotState_(props, props.getProperties());
  SpreadsheetApp.flush();

  console.log("========== 시스템 정합화 시작 ==========");
  changedLines.forEach(function(line) { console.log(" - " + line); });
  if (missingInTechSheet.length > 0) {
    console.log("[경고] 트레이딩로그 미청산 종목 중 기술분석 시트에서 못 찾은 항목");
    missingInTechSheet.forEach(function(line) { console.log(" - " + line); });
  }
  console.log("[후속] 진단 시트 재생성");
  console.log("========== 시스템 정합화 종료 ==========");

  diagnoseCurrentSystemHoldings_20260422();
}

function _manualTradingLogSnapshot_20260422_() {
  return [
    { matchers: ["SK하이닉스"], opinion: "관망" },
    { matchers: ["삼성전자"], opinion: "관망" },
    { matchers: ["레인보우로보틱스"], opinion: "관망" },
    { matchers: ["두산에너빌리티"], opinion: "관망" },
    { matchers: ["한국전력"], opinion: "관망" },
    { matchers: ["현대차", "현대자동차"], opinion: "관망" },
    { matchers: ["한화에어로스페이스"], opinion: "관망" },
    { matchers: ["한화오션"], opinion: "관망" },
    { matchers: ["한미반도체"], opinion: "관망" },
    { matchers: ["SK이노베이션"], opinion: "관망" },
    { matchers: ["삼성전기"], opinion: "관망" },
    { matchers: ["기아차", "기아"], opinion: "관망" },
    // 최신 이미지 기준: 에코프로비엠은 C전략 보유 중이며, ONDS는 스냅샷에서 제외
    { matchers: ["에코프로비엠", "247540"], opinion: "관망", entryPrice: 197800, entryDate: "2026-04-10", strategyType: "C" },
    { matchers: ["로킷헬스케어"], opinion: "관망" },
    { matchers: ["현대제철"], opinion: "관망" },
    { matchers: ["HD현대중공업"], opinion: "관망" },
    { matchers: ["DL이앤씨"], opinion: "관망" },
    { matchers: ["현대글로비스"], opinion: "관망" },
    { matchers: ["현대건설", "000720"], opinion: "관망", entryPrice: 177000, entryDate: "2026-04-19", strategyType: "D" },
    { matchers: ["대덕전자"], opinion: "관망" },
    { matchers: ["LG이노텍"], opinion: "관망" },
    { matchers: ["에이피알"], opinion: "관망" },
    { matchers: ["LIG디펜스앤에어로스페이스", "LIG디펜스", "LIG넥스원", "079550"], opinion: "매수", entryPrice: 1005000, entryDate: "2026-04-22", strategyType: "A" },
    { matchers: ["HOOD"], opinion: "관망" },
    { matchers: ["AVGO"], opinion: "관망" },
    { matchers: ["AMD"], opinion: "관망" },
    { matchers: ["MSFT"], opinion: "관망" },
    { matchers: ["GOOGL"], opinion: "관망" },
    { matchers: ["NVDA"], opinion: "관망" },
    { matchers: ["TSLA"], opinion: "관망" },
    { matchers: ["MU"], opinion: "관망" },
    { matchers: ["LRCX"], opinion: "관망" },
    { matchers: ["ON"], opinion: "관망" },
    { matchers: ["SNDK"], opinion: "관망" },
    { matchers: ["ASTS"], opinion: "관망", entryPrice: 78.44, entryDate: "2026-04-20", strategyType: "D" },
    { matchers: ["AVAV"], opinion: "관망" },
    { matchers: ["IONQ"], opinion: "관망" },
    { matchers: ["RKLB"], opinion: "관망" },
    { matchers: ["PLTR"], opinion: "관망" },
    { matchers: ["APP"], opinion: "관망" },
    { matchers: ["SOXL"], opinion: "관망" },
    { matchers: ["TSLL"], opinion: "관망" },
    { matchers: ["TE"], opinion: "관망" },
    { matchers: ["BE"], opinion: "관망" },
    { matchers: ["PL"], opinion: "관망" },
    { matchers: ["VRT"], opinion: "관망" },
    { matchers: ["LITE"], opinion: "관망", entryPrice: 871.86, entryDate: "2026-04-21", strategyType: "D" },
    { matchers: ["TER"], opinion: "관망" },
    { matchers: ["ANET"], opinion: "관망" },
    { matchers: ["IREN"], opinion: "관망" },
    { matchers: ["HOOG"], opinion: "관망" },
    { matchers: ["SOLT"], opinion: "관망" },
    { matchers: ["ETHU"], opinion: "관망" },
    { matchers: ["NBIS"], opinion: "관망" },
    { matchers: ["LPTH"], opinion: "관망" },
    { matchers: ["CONL"], opinion: "관망" },
    { matchers: ["GLW"], opinion: "관망" },
    { matchers: ["FLNC"], opinion: "관망" },
    { matchers: ["VST"], opinion: "관망" },
    { matchers: ["ASX"], opinion: "관망" },
    { matchers: ["CRCL"], opinion: "관망" },
    { matchers: ["SGML"], opinion: "관망" },
    { matchers: ["AEHR"], opinion: "관망" },
    { matchers: ["MP"], opinion: "관망" },
    { matchers: ["PLAB"], opinion: "관망" },
    { matchers: ["SKYT"], opinion: "관망" },
    { matchers: ["SMTC"], opinion: "관망" },
    { matchers: ["COHR"], opinion: "관망" },
    { matchers: ["MPWR"], opinion: "관망" },
    { matchers: ["CIEN"], opinion: "관망" },
    { matchers: ["KLAC"], opinion: "관망" },
    { matchers: ["FORM"], opinion: "관망" },
    { matchers: ["CRDO"], opinion: "관망" },
    { matchers: ["ACLS"], opinion: "관망" },
    { matchers: ["ONTO"], opinion: "관망" }
  ];
}

function _manualGetTechSheet_() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("기술분석");
  if (!sheet) throw new Error("기술분석 시트를 찾지 못했습니다.");
  return sheet;
}

function _manualCols_() {
  if (typeof CONSTANTS !== "undefined" && CONSTANTS.COL_INDICES) return CONSTANTS.COL_INDICES;
  return {
    stockName: 0,
    stockLabel: 1,
    currentPrice: 44,
    opinion: 3,
    entryPrice: 52,
    entryDate: 53,
    entryStrategy: 54
  };
}

function _manualToNum_(value) {
  if (value === "" || value === null || value === undefined) return 0;
  const n = Number(value);
  return isNaN(n) ? 0 : n;
}

function _manualHasPositionTrace_(rowInfo, allProps, C) {
  const row = rowInfo && rowInfo.values ? rowInfo.values : [];
  const opinion = String(row[C.opinion] || "").trim();
  const hasSheetEntryPrice = _manualToNum_(row[C.entryPrice]) > 0;
  const hasSheetEntryDate = !!_manualDateString_(row[C.entryDate]);
  const hasSheetEntryStrategy = !!String(row[C.entryStrategy] || "").trim();
  const hasSheetHoldingTrace = opinion === "매수" || hasSheetEntryPrice || hasSheetEntryDate || hasSheetEntryStrategy;
  if (hasSheetHoldingTrace) return true;

  const stockName = rowInfo.stockName;
  return !!(
    allProps["ENTRY_" + stockName] ||
    allProps["SELL_" + stockName] ||
    allProps["EXIT_REASON_" + stockName] ||
    allProps["SOLD_FLAG_" + stockName] ||
    allProps["CYCLE_ENTRY_" + stockName] ||
    allProps["REENTRY_COUNT_" + stockName]
  );
}

function _manualGetTradingLogSheet_() {
  if (typeof Utils !== "undefined" && Utils.getTradingLogSheet) {
    return Utils.getTradingLogSheet();
  }

  const sheet = _manualGetTechSheet_();
  const spreadsheetId = sheet.getRange("I1").getValue();
  if (!spreadsheetId) throw new Error("기술분석 시트 I1에 트레이딩로그 스프레드시트 ID가 없습니다.");

  const targetSpreadsheet = SpreadsheetApp.openById(spreadsheetId);
  const logSheet = targetSpreadsheet.getSheetByName("트레이딩로그");
  if (!logSheet) throw new Error("트레이딩로그 시트를 찾지 못했습니다.");
  return logSheet;
}

function _manualLoadOpenPositionsFromTradingLog_(logSheet) {
  if (typeof Utils !== "undefined" && Utils.getOpenTradingLogEntries && Utils.getLatestOpenTradingLogEntryMap) {
    const latestByStock = Utils.getLatestOpenTradingLogEntryMap(Utils.getOpenTradingLogEntries(logSheet));
    const openPositions = {};
    Object.keys(latestByStock).forEach(function(stockName) {
      const entry = latestByStock[stockName];
      openPositions[stockName] = {
        stockName: stockName,
        entryDate: entry.buyDateString || _manualDateString_(entry.buyDate),
        entryPrice: entry.buyPrice,
        strategyType: entry.strategyType,
        strategyLabel: entry.strategyLabel || _manualStrategyLabel_(entry.strategyType),
        sourceRow: entry.rowNumber
      };
    });
    return openPositions;
  }

  const lastRow = logSheet.getLastRow();
  const openPositions = {};
  if (lastRow < 3) return openPositions;

  const data = logSheet.getRange(3, 1, lastRow - 2, 6).getValues();
  data.forEach(function(row, index) {
    const stockName = String(row[0] || "").trim();
    const buyDate = row[1];
    const buyPrice = _manualParsePriceNumber_(row[2]);
    const sellDate = row[3];
    const strategyLabel = String(row[5] || "").trim();
    const strategyType = _manualParseStrategyCodeFromLabel_(strategyLabel);

    if (!stockName || !buyDate || !buyPrice || sellDate) return;
    if (!strategyType) return;

    openPositions[stockName] = {
      stockName: stockName,
      entryDate: _manualDateString_(buyDate),
      entryPrice: buyPrice,
      strategyType: strategyType,
      strategyLabel: strategyLabel || _manualStrategyLabel_(strategyType),
      sourceRow: index + 3
    };
  });

  return openPositions;
}

function _manualParsePriceNumber_(value) {
  if (typeof value === "number") return value;
  const normalized = String(value || "").replace(/[^0-9.\-]/g, "");
  if (!normalized) return 0;
  const n = Number(normalized);
  return isNaN(n) ? 0 : n;
}

function _manualParseStrategyCodeFromLabel_(label) {
  const text = String(label || "").trim();
  const match = text.match(/^([A-F])\s*\./i) || text.match(/^([A-F])$/i);
  return match ? match[1].toUpperCase() : "";
}

function _manualNormalize_(value) {
  return String(value || "")
    .toUpperCase()
    .replace(/\s+/g, "")
    .replace(/[().-]/g, "")
    .trim();
}

function _manualDisplayName_(stockName, stockLabel) {
  if (!stockLabel) return stockName;
  return /^[0-9]{6}$/.test(stockName) ? `${stockLabel}(${stockName})` : stockName;
}

function _manualFindRowInfo_(rowsByStock, matchers) {
  const normalizedMatchers = (matchers || []).map(_manualNormalize_);
  const rowInfos = Object.keys(rowsByStock).map(function(stockName) { return rowsByStock[stockName]; });

  for (let i = 0; i < normalizedMatchers.length; i++) {
    const matcher = normalizedMatchers[i];
    for (let j = 0; j < rowInfos.length; j++) {
      const rowInfo = rowInfos[j];
      const candidates = [
        rowInfo.stockName,
        rowInfo.stockLabel,
        rowInfo.displayName
      ].map(_manualNormalize_);
      if (candidates.some(function(candidate) { return candidate && candidate === matcher; })) {
        return rowInfo;
      }
    }
  }

  for (let i = 0; i < normalizedMatchers.length; i++) {
    const matcher = normalizedMatchers[i];
    if (!matcher || matcher.length < 3) continue;
    for (let j = 0; j < rowInfos.length; j++) {
      const rowInfo = rowInfos[j];
      const candidates = [
        rowInfo.stockName,
        rowInfo.stockLabel,
        rowInfo.displayName
      ].map(_manualNormalize_);
      if (candidates.some(function(candidate) { return candidate && candidate.indexOf(matcher) !== -1; })) {
        return rowInfo;
      }
    }
  }

  return null;
}

function _manualDateObject_(yyyyMmDd) {
  return new Date(`${yyyyMmDd}T00:00:00+09:00`);
}

function _manualDateString_(value) {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(value);
  if (isNaN(date.getTime())) return "";
  return Utilities.formatDate(date, "Asia/Seoul", "yyyy-MM-dd");
}

function _manualStrategyLabel_(type) {
  const map = {
    A: "A. 200일선 상방 & 모멘텀 재가속",
    B: "B. 200일선 하방 & 공황 저점",
    C: "C. 200일선 상방 & 스퀴즈 거래량 돌파",
    D: "D. 200일선 상방 & 상승 흐름 강화",
    E: "E. 200일선 상방 & 스퀴즈 저점",
    F: "F. 200일선 상방 & BB 극단 저점"
  };
  return map[type] || type;
}

function _manualLoadEntry_(stockName, allProps) {
  const raw = allProps["ENTRY_" + stockName];
  if (!raw) return { price: 0, dateString: "", strategyType: "" };
  const parts = String(raw).split("|");
  const price = Number(parts[0]) || 0;
  const dateString = _manualDateString_(parts[1]);
  const strategyType = String(parts[2] || "").trim();
  return {
    price: price,
    dateString: dateString,
    strategyType: strategyType
  };
}

function _manualLoadActiveSlots_(stockName, allProps) {
  return Object.keys(allProps)
    .filter(function(key) {
      return key.indexOf(`SLOT_${stockName}_`) === 0 && key.indexOf(`SLOT_SELL_${stockName}_`) !== 0 && key.indexOf(`SLOT_UPPER_EXIT_ARM_${stockName}_`) !== 0;
    })
    .map(function(key) {
      return key.substring((`SLOT_${stockName}_`).length);
    })
    .sort();
}

function _manualFmtPrice_(price, stockName) {
  if (!price) return "";
  if (/^[0-9]{6}$/.test(String(stockName || ""))) {
    return "₩" + Number(price).toLocaleString("ko-KR");
  }
  return "$" + Number(price).toLocaleString("en-US", {
    minimumFractionDigits: Number(price) % 1 === 0 ? 0 : 2,
    maximumFractionDigits: 2
  });
}

function _manualSaveEntry_(stockName, price, yyyyMmDd, strategyType, props) {
  const value = `${price}|${yyyyMmDd}T00:00:00+09:00|${strategyType}`;
  props.setProperty("ENTRY_" + stockName, value);
}

function _manualSaveSell_(stockName, dateObj, sellPrice, props) {
  const value = Utilities.formatDate(dateObj, "Asia/Seoul", "yyyy-MM-dd'T'HH:mm:ss") + "+09:00";
  props.setProperty("SELL_" + stockName, sellPrice ? `${value}|${sellPrice}` : value);
}

function _manualClearPrimaryTradingState_(stockName, props, allProps) {
  const exactKeys = [
    "ENTRY_" + stockName,
    "SELL_" + stockName,
    "EXIT_REASON_" + stockName,
    "HOLD_ANCHOR_" + stockName,
    "HOLD_WATCH_" + stockName,
    "A_HOLD_ANCHOR_" + stockName,
    "A_HOLD_WATCH_" + stockName,
    "UPPER_EXIT_ARM_" + stockName,
    "CYCLE_ENTRY_" + stockName,
    "REENTRY_COUNT_" + stockName,
    "SOLD_FLAG_" + stockName
  ];

  exactKeys.forEach(function(key) {
    props.deleteProperty(key);
  });

  Object.keys(allProps || {})
    .filter(function(key) {
      return key.indexOf(`SLOT_${stockName}_`) === 0
        || key.indexOf(`SLOT_SELL_${stockName}_`) === 0
        || key.indexOf(`SLOT_UPPER_EXIT_ARM_${stockName}_`) === 0;
    })
    .forEach(function(key) {
      props.deleteProperty(key);
    });
}

function _manualClearAllSlotState_(props, allProps) {
  Object.keys(allProps || {})
    .filter(function(key) {
      return key.indexOf("SLOT_") === 0
        || key.indexOf("SLOT_SELL_") === 0
        || key.indexOf("SLOT_UPPER_EXIT_ARM_") === 0;
    })
    .forEach(function(key) {
      props.deleteProperty(key);
    });
}
