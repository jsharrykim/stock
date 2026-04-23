/**
 * 시스템 상태 점검 / 트레이딩로그 정합화 보조 스크립트
 *
 * 실행 함수:
 * - inspectCurrentSystemHoldings()
 * - inspectCurrentHoldingsOverview()
 * - previewTradingLogSyncFromCurrentSystem()
 * - applyTradingLogSyncFromCurrentSystem()
 */

function inspectCurrentSystemHoldings() {
  const { targetSheet } = Utils.getSheets("기술분석");
  if (!targetSheet) {
    console.log("[점검 실패] 기술분석 시트를 찾지 못했습니다.");
    return;
  }

  const allProperties = PropertiesService.getScriptProperties().getProperties();
  const rows = _sstLoadTechRows_(targetSheet);
  const holdings = [];
  const mismatches = [];

  rows.forEach(({ rowNumber, row }) => {
    const stockName = String(row[Utils.COL_INDICES.stockName] || "").trim();
    if (!stockName) return;

    const displayName = Utils.getDisplayName(stockName, row);
    const opinion = String(row[Utils.COL_INDICES.opinion] || "").trim();
    const saved = Utils.loadEntryInfoFrom(stockName, allProperties);
    const slots = _sstLoadSlotEntries_(stockName, allProperties);
    const sellTime = Utils.loadSellTimeFrom(stockName, allProperties);
    const sellPrice = Utils.loadSellPriceFrom(stockName, allProperties);
    const isHolding = saved.price > 0 || slots.length > 0;

    if (isHolding) {
      const parts = [];
      if (saved.price > 0) {
        const entryDate = saved.date ? Utilities.formatDate(saved.date, "Asia/Seoul", "yyyy-MM-dd") : "-";
        parts.push(`PRIMARY ${saved.strategyType} ${Utils.fmtPrice(saved.price, stockName)} ${entryDate}`);
      }
      slots.forEach(slot => {
        const slotDate = slot.date ? Utilities.formatDate(slot.date, "Asia/Seoul", "yyyy-MM-dd") : "-";
        parts.push(`SLOT ${slot.strategy} ${Utils.fmtPrice(slot.price, stockName)} ${slotDate}`);
      });
      holdings.push(`${displayName} | 의견:${opinion || "-"} | ${parts.join(" | ")}`);
    }

    if (opinion === "매수" && saved.price <= 0) {
      mismatches.push(`${displayName}: 시트 의견은 매수인데 ENTRY_ 없음`);
    }
    if (opinion === "매도" && isHolding) {
      mismatches.push(`${displayName}: 시트 의견은 매도인데 시스템상 보유 상태`);
    }
    if ((opinion === "관망" || opinion === "매수") && sellTime && saved.price <= 0) {
      const sellDate = Utilities.formatDate(sellTime, "Asia/Seoul", "yyyy-MM-dd");
      mismatches.push(`${displayName}: SELL_ 잔존 (${sellDate}, ${sellPrice ? Utils.fmtPrice(sellPrice, stockName) : "-"})`);
    }
  });

  console.log("========== 실제 시스템 보유 점검 시작 ==========");
  console.log("[보유 종목]");
  if (holdings.length === 0) console.log("없음");
  holdings.forEach(line => console.log(" - " + line));

  console.log("[불일치]");
  if (mismatches.length === 0) console.log("없음");
  mismatches.forEach(line => console.log(" - " + line));
  console.log("========== 실제 시스템 보유 점검 종료 ==========");
}

function inspectCurrentHoldingsOverview() {
  const { targetSheet } = Utils.getSheets("기술분석");
  if (!targetSheet) {
    console.log("[점검 실패] 기술분석 시트를 찾지 못했습니다.");
    return;
  }

  const logSheet = Utils.getTradingLogSheet();
  const allProperties = PropertiesService.getScriptProperties().getProperties();
  const rows = _sstLoadTechRows_(targetSheet);
  const openEntries = logSheet ? Utils.getOpenTradingLogEntries(logSheet) : [];
  const openByStock = _sstGroupOpenEntriesByStock_(openEntries);
  const lines = [];
  const mismatches = [];
  let systemHoldingCount = 0;
  let logHoldingCount = 0;
  let sheetHoldingCount = 0;

  rows.forEach(({ row }) => {
    const stockName = String(row[Utils.COL_INDICES.stockName] || "").trim();
    if (!stockName) return;

    const displayName = Utils.getDisplayName(stockName, row);
    const opinion = String(row[Utils.COL_INDICES.opinion] || "").trim();
    const saved = Utils.loadEntryInfoFrom(stockName, allProperties);
    const slots = _sstLoadSlotEntries_(stockName, allProperties);
    const openForStock = openByStock[stockName] || [];

    const rawSheetDate = row[Utils.COL_INDICES.entryDate];
    const sheetEntryDate = rawSheetDate instanceof Date
      ? rawSheetDate
      : (rawSheetDate ? Utils.parseDateKST(rawSheetDate) : null);
    const sheetEntryPrice = Utils.toNum(row[Utils.COL_INDICES.entryPrice]) || 0;
    const sheetStrategyLabel = String(row[Utils.COL_INDICES.entryStrategy] || "").trim();

    const hasSystemHolding = saved.price > 0 || slots.length > 0;
    const hasLogHolding = openForStock.length > 0;
    const hasSheetHolding = sheetEntryPrice > 0 || !!sheetEntryDate || !!sheetStrategyLabel;

    if (hasSystemHolding) systemHoldingCount++;
    if (hasLogHolding) logHoldingCount++;
    if (hasSheetHolding) sheetHoldingCount++;

    if (!hasSystemHolding && !hasLogHolding && !hasSheetHolding && opinion !== "매수") return;

    const parts = [];

    if (saved.price > 0) {
      const entryDate = saved.date ? Utilities.formatDate(saved.date, "Asia/Seoul", "yyyy-MM-dd") : "-";
      parts.push(`SYSTEM PRIMARY ${saved.strategyType} ${Utils.fmtPrice(saved.price, stockName)} ${entryDate}`);
    }

    slots.forEach(slot => {
      const slotDate = slot.date ? Utilities.formatDate(slot.date, "Asia/Seoul", "yyyy-MM-dd") : "-";
      parts.push(`SYSTEM SLOT ${slot.strategy} ${Utils.fmtPrice(slot.price, stockName)} ${slotDate}`);
    });

    if (hasLogHolding) {
      const latestLog = openForStock
        .slice()
        .sort((a, b) => b.buyDate.getTime() - a.buyDate.getTime() || b.rowNumber - a.rowNumber)[0];
      parts.push(`LOG ${openForStock.length}건 / 최신 ${latestLog.strategyType} ${Utils.fmtPrice(latestLog.buyPrice, stockName)} ${latestLog.buyDateString}`);
    }

    if (hasSheetHolding) {
      const sheetDateString = sheetEntryDate ? Utilities.formatDate(sheetEntryDate, "Asia/Seoul", "yyyy-MM-dd") : "-";
      parts.push(`SHEET ${sheetStrategyLabel || "-"} ${sheetEntryPrice > 0 ? Utils.fmtPrice(sheetEntryPrice, stockName) : "-"} ${sheetDateString}`);
    }

    lines.push(`${displayName} | 의견:${opinion || "-"} | ${parts.join(" | ")}`);

    if (hasLogHolding && !hasSystemHolding) {
      mismatches.push(`${displayName}: 트레이딩로그 미청산은 있는데 시스템 보유 상태가 없음`);
    }
    if (hasSystemHolding && !hasLogHolding) {
      mismatches.push(`${displayName}: 시스템 보유 상태는 있는데 트레이딩로그 미청산이 없음`);
    }
    if (opinion === "매수" && !hasSystemHolding && !hasLogHolding) {
      mismatches.push(`${displayName}: 시트 의견은 매수인데 시스템/트레이딩로그 기준 보유가 아님`);
    }
    if (hasSheetHolding && !hasSystemHolding && !hasLogHolding) {
      mismatches.push(`${displayName}: 시트 진입 정보만 남아 있음`);
    }
    if (!hasSheetHolding && (hasSystemHolding || hasLogHolding) && opinion !== "매도") {
      mismatches.push(`${displayName}: 시스템/로그 보유 흔적은 있는데 시트 진입 정보가 비어 있음`);
    }
  });

  console.log("========== 현재 보유 현황 점검 시작 ==========");
  console.log(`[요약] SYSTEM=${systemHoldingCount} / LOG=${logHoldingCount} / SHEET=${sheetHoldingCount}`);
  console.log("[보유/흔적 종목]");
  if (lines.length === 0) console.log("없음");
  lines.forEach(line => console.log(" - " + line));

  console.log("[불일치]");
  if (mismatches.length === 0) console.log("없음");
  mismatches.forEach(line => console.log(" - " + line));
  console.log("========== 현재 보유 현황 점검 종료 ==========");
}

function previewTradingLogSyncFromCurrentSystem() {
  _runTradingLogSyncFromCurrentSystem_(false);
}

function applyTradingLogSyncFromCurrentSystem() {
  _runTradingLogSyncFromCurrentSystem_(true);
}

function _runTradingLogSyncFromCurrentSystem_(shouldApply) {
  const { targetSheet } = Utils.getSheets("기술분석");
  if (!targetSheet) {
    console.log("[동기화 실패] 기술분석 시트를 찾지 못했습니다.");
    return;
  }

  const logSheet = Utils.getTradingLogSheet();
  if (!logSheet) {
    console.log("[동기화 실패] 트레이딩로그 시트를 찾지 못했습니다.");
    return;
  }

  const allProperties = PropertiesService.getScriptProperties().getProperties();
  const rows = _sstLoadTechRows_(targetSheet);
  const openEntries = Utils.getOpenTradingLogEntries(logSheet);
  const openByStock = _sstGroupOpenEntriesByStock_(openEntries);
  const seenStocks = {};
  const actions = [];

  rows.forEach(({ row, rowNumber }) => {
    const stockName = String(row[Utils.COL_INDICES.stockName] || "").trim();
    if (!stockName) return;
    seenStocks[stockName] = true;

    const displayName = Utils.getDisplayName(stockName, row);
    const opinion = String(row[Utils.COL_INDICES.opinion] || "").trim();
    const saved = Utils.loadEntryInfoFrom(stockName, allProperties);
    const slots = _sstLoadSlotEntries_(stockName, allProperties);
    const openForStock = openByStock[stockName] || [];

    if (opinion === "매도") {
      if (openForStock.length > 0) {
        const sellTime = Utils.loadSellTimeFrom(stockName, allProperties);
        const currentPrice = Utils.toNum(row[Utils.COL_INDICES.currentPrice]) || 0;
        const sellPrice = Utils.loadSellPriceFrom(stockName, allProperties) || currentPrice;
        const sellDate = sellTime
          ? Utilities.formatDate(sellTime, "Asia/Seoul", "yyyy-MM-dd")
          : Utilities.formatDate(new Date(), "Asia/Seoul", "yyyy-MM-dd");

        actions.push({
          type: "close_all",
          stockName,
          displayName,
          rowNumber,
          count: openForStock.length,
          sellDate,
          sellPrice
        });
      }
      return;
    }

    const expectedLegs = _sstBuildExpectedOpenLegs_(stockName, row, saved, slots);
    expectedLegs.forEach(expected => {
      const existing = openForStock.find(entry => _sstIsSameOpenLeg_(entry, expected));
      if (!existing) {
        actions.push({
          type: "add_open",
          stockName,
          displayName,
          rowNumber,
          buyDate: expected.buyDateString,
          buyPrice: expected.buyPrice,
          strategyLabel: expected.strategyLabel,
          source: expected.source
        });
      }
    });

    if (expectedLegs.length === 0 && openForStock.length > 0) {
      actions.push({
        type: "warn_open_without_system_position",
        stockName,
        displayName,
        rowNumber,
        count: openForStock.length
      });
    }
  });

  Object.keys(openByStock).forEach(stockName => {
    if (seenStocks[stockName]) return;
    actions.push({
      type: "warn_open_without_sheet_row",
      stockName,
      displayName: stockName,
      count: openByStock[stockName].length
    });
  });

  console.log("");
  console.log("========== 트레이딩로그 시스템 기준 동기화 시작 ==========");
  console.log(`[모드] ${shouldApply ? "실적용" : "미리보기"}`);
  if (actions.length === 0) {
    console.log("동기화할 항목 없음");
    console.log("========== 트레이딩로그 시스템 기준 동기화 종료 ==========");
    return;
  }

  actions.forEach(action => {
    if (action.type === "add_open") {
      console.log(`${shouldApply ? "[추가]" : "[예정]"} ${action.displayName}: 열린 매수행 추가 (${action.buyDate}, ${Utils.fmtPrice(action.buyPrice, action.stockName)}, ${action.strategyLabel}, source=${action.source})`);
      if (!shouldApply) return;
      const newRow = Utils.getNextTradingLogRow(logSheet);
      logSheet.getRange(newRow, 1, 1, 6).setValues([[action.stockName, action.buyDate, "", "", "", action.strategyLabel]]);
      Utils.setTradingLogPriceCell(logSheet.getRange(newRow, 3), action.stockName, action.buyPrice);
      Utils._nextTradingLogRowCache = newRow + 1;
      return;
    }

    if (action.type === "close_all") {
      console.log(`${shouldApply ? "[마감]" : "[예정]"} ${action.displayName}: 열린 ${action.count}건 일괄 마감 (${action.sellDate}, ${Utils.fmtPrice(action.sellPrice, action.stockName)})`);
      if (!shouldApply) return;
      Utils.recordAllOpenSellSignals(action.stockName, action.sellDate, action.sellPrice);
      return;
    }

    if (action.type === "warn_open_without_system_position") {
      console.log(`[경고] ${action.displayName}: 시스템상 비보유인데 트레이딩로그 열린 행 ${action.count}건 존재`);
      return;
    }

    if (action.type === "warn_open_without_sheet_row") {
      console.log(`[경고] ${action.displayName}: 기술분석 시트 행이 없는데 트레이딩로그 열린 행 ${action.count}건 존재`);
    }
  });

  if (shouldApply) SpreadsheetApp.flush();
  console.log("========== 트레이딩로그 시스템 기준 동기화 종료 ==========");
}

function _sstLoadTechRows_(targetSheet) {
  const lastRow = targetSheet.getLastRow();
  if (lastRow < 3) return [];
  const data = targetSheet.getRange(3, 1, lastRow - 2, targetSheet.getLastColumn()).getValues();
  return data.map((row, index) => ({ rowNumber: index + 3, row }));
}

function _sstLoadSlotEntries_(stockName, allProperties) {
  return Object.keys(allProperties)
    .filter(key =>
      key.indexOf(`SLOT_${stockName}_`) === 0 &&
      key.indexOf(`SLOT_SELL_${stockName}_`) !== 0 &&
      key.indexOf(`SLOT_UPPER_EXIT_ARM_${stockName}_`) !== 0
    )
    .sort()
    .map(key => {
      const strategy = key.substring((`SLOT_${stockName}_`).length);
      return Utils.loadSlot(stockName, strategy, allProperties);
    })
    .filter(Boolean);
}

function _sstBuildExpectedOpenLegs_(stockName, row, saved, slots) {
  const expected = [];
  const seen = {};

  const pushLeg = (price, dateObj, strategyType, source) => {
    if (!price || !dateObj || !strategyType) return;
    const buyDateString = Utilities.formatDate(dateObj, "Asia/Seoul", "yyyy-MM-dd");
    const signature = `${buyDateString}|${Number(price)}|${strategyType}`;
    if (seen[signature]) return;
    seen[signature] = true;
    expected.push({
      stockName,
      buyDateString,
      buyPrice: Number(price),
      strategyType,
      strategyLabel: strategyDisplayName(strategyType),
      source
    });
  };

  if (saved && saved.price > 0 && saved.date && saved.strategyType) {
    pushLeg(saved.price, saved.date, saved.strategyType, "ENTRY");
  } else {
    const sheetEntryPrice = Utils.toNum(row[Utils.COL_INDICES.entryPrice]) || 0;
    const sheetEntryDate = row[Utils.COL_INDICES.entryDate] instanceof Date ? row[Utils.COL_INDICES.entryDate] : null;
    const sheetStrategyLabel = String(row[Utils.COL_INDICES.entryStrategy] || "").trim();
    const sheetStrategyType = Utils.parseTradingLogStrategyCode(sheetStrategyLabel);
    pushLeg(sheetEntryPrice, sheetEntryDate, sheetStrategyType, "SHEET");
  }

  (slots || []).forEach(slot => {
    pushLeg(slot.price, slot.date, slot.strategy, "SLOT");
  });

  return expected;
}

function _sstIsSameOpenLeg_(logEntry, expectedLeg) {
  return logEntry.stockName === expectedLeg.stockName
    && logEntry.buyDateString === expectedLeg.buyDateString
    && logEntry.strategyType === expectedLeg.strategyType
    && Math.abs(Number(logEntry.buyPrice) - Number(expectedLeg.buyPrice)) < 0.0001;
}

function _sstGroupOpenEntriesByStock_(openEntries) {
  const grouped = {};
  (openEntries || []).forEach(entry => {
    if (!grouped[entry.stockName]) grouped[entry.stockName] = [];
    grouped[entry.stockName].push(entry);
  });
  return grouped;
}
