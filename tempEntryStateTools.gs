/**
 * 매수 의견이 없는 종목의 사이클 히스토리 키(SOLD_FLAG_, CYCLE_ENTRY_, REENTRY_COUNT_, SELL_)를
 * 정리합니다. ENTRY_가 없는데 이 키들이 남아있으면 다음 신규 진입 시 "재진입"으로 오분류됩니다.
 *
 * 실행 결과: 현재 ENTRY_가 있는(보유 중인) 종목은 건드리지 않고,
 * ENTRY_가 없는 종목의 잔여 사이클 히스토리만 삭제합니다.
 */
function clearStaleCycleHistory() {
  const props = PropertiesService.getScriptProperties();
  const allProperties = props.getProperties();

  // 현재 ENTRY_가 살아있는 종목 목록
  const heldTickers = new Set(
    Object.keys(allProperties)
      .filter(function(k) { return k.indexOf("ENTRY_") === 0; })
      .map(function(k) { return k.replace("ENTRY_", ""); })
      .filter(function(ticker) {
        const raw = allProperties["ENTRY_" + ticker] || "";
        const price = Number(raw.split("|")[0]);
        return price > 0;
      })
  );

  const cycleKeyPrefixes = ["SOLD_FLAG_", "CYCLE_ENTRY_", "REENTRY_COUNT_", "SELL_"];
  const toDelete = [];

  Object.keys(allProperties).forEach(function(key) {
    const matchedPrefix = cycleKeyPrefixes.find(function(prefix) { return key.indexOf(prefix) === 0; });
    if (!matchedPrefix) return;
    const ticker = key.replace(matchedPrefix, "");
    if (!heldTickers.has(ticker)) {
      toDelete.push(key);
    }
  });

  console.log("========== 사이클 히스토리 정리 ==========");
  if (toDelete.length === 0) {
    console.log("[정상] 정리할 잔여 사이클 히스토리 없음");
  } else {
    toDelete.forEach(function(key) {
      console.log(" - 삭제: " + key + " = " + allProperties[key]);
      props.deleteProperty(key);
    });
    console.log("[완료] " + toDelete.length + "개 키 삭제");
  }
  console.log("보유 종목 (ENTRY_ 유효): " + Array.from(heldTickers).join(", "));
  console.log("========== 사이클 히스토리 정리 종료 ==========");
}

function backfillLrcxSndkBuyLogs() {
  const entries = [
    { stockName: "LRCX", buyDate: "2026-04-24", buyPrice: 258.56, strategyType: "D" },
    { stockName: "SNDK", buyDate: "2026-04-24", buyPrice: 932.43, strategyType: "D" },
  ];

  const logSheet = Utils.getTradingLogSheet();
  if (!logSheet) {
    console.log("[중단] 트레이딩로그 시트 접근 실패");
    return;
  }

  const normalizeDate = function(value) {
    if (!value) return "";
    if (value instanceof Date) return Utilities.formatDate(value, "Asia/Seoul", "yyyy-MM-dd");
    const text = String(value).trim();
    return text ? text.replace(/\./g, "-") : "";
  };

  const normalizePrice = function(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "number") return value;
    const text = String(value).replace(/[^0-9.-]/g, "");
    const parsed = Number(text);
    return isNaN(parsed) ? null : parsed;
  };

  const normalizeStrategy = function(value) {
    return String(value || "").replace(/^[A-F]\.\s*/, "").trim();
  };

  const lastRow = logSheet.getLastRow();
  const rows = lastRow >= 3 ? logSheet.getRange(3, 1, lastRow - 2, 6).getValues() : [];

  console.log("========== LRCX/SNDK 트레이딩로그 백필 시작 ==========");

  entries.forEach(function(entry) {
    const strategyLabel = strategyDisplayName(entry.strategyType);
    const exists = rows.some(function(row) {
      return String(row[0] || "").trim() === entry.stockName &&
        normalizeDate(row[1]) === entry.buyDate &&
        !String(row[3] || "").trim() &&
        normalizeStrategy(row[5]) === normalizeStrategy(strategyLabel) &&
        Math.abs((normalizePrice(row[2]) || 0) - entry.buyPrice) < 0.0001;
    });

    if (exists) {
      console.log("[건너뜀] 이미 존재: " + entry.stockName + " / " + entry.buyDate + " / " + Utils.fmtPrice(entry.buyPrice, entry.stockName));
      return;
    }

    Utils.recordBuySignal(entry.stockName, entry.buyDate, entry.buyPrice, strategyLabel);
  });

  console.log("========== LRCX/SNDK 트레이딩로그 백필 종료 ==========");
}

function sendLrcxSndkCorrectionEmail() {
  const sheetInfo = Utils.getSheets("기술분석");
  const targetSheet = sheetInfo && sheetInfo.targetSheet;
  if (!targetSheet) {
    console.log("[중단] 기술분석 시트 접근 실패");
    return;
  }

  const recipientEmail = String(targetSheet.getRange("F1").getValue() || "").trim();
  if (!recipientEmail) {
    console.log("[중단] F1 이메일 주소 없음");
    return;
  }

  const subject = "[정정] 투자의견 변경 알림";
  const htmlBody =
    '<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.6;max-width:600px;">' +
    '<p style="font-size:16px;font-weight:bold;color:#333;">투자의견 변경 알림 정정</p>' +
    '<p>직전 메일은 추적 스냅샷 기준이 꼬여 실제 변경 종목과 다르게 발송되었습니다.</p>' +
    '<p><strong>실제 매수 전환 종목</strong></p>' +
    '<ul>' +
    '<li><strong>LRCX</strong>: 관망 → 매수 (D. 200일선 상방 & 상승 흐름 강화, $258.56)</li>' +
    '<li><strong>SNDK</strong>: 관망 → 매수 (D. 200일선 상방 & 상승 흐름 강화, $932.43)</li>' +
    '</ul>' +
    '<p>참고: ASTS, LITE 는 보유 유지였고, 잘못 메일에 들어간 종목은 실제 변경 종목이 아닙니다.</p>' +
    '</div>';

  MailApp.sendEmail({
    to: recipientEmail,
    subject: subject,
    htmlBody: htmlBody
  });

  console.log("[완료] 정정 이메일 발송: " + recipientEmail);
}

function runTrackingSnapshotNow() {
  Utils.snapshotOpinionsForTracking();
}

function checkBuyOpinionsMissingEntryKey() {
  const sheetInfo = Utils.getSheets("기술분석");
  const targetSheet = sheetInfo && sheetInfo.targetSheet;
  if (!targetSheet) {
    console.log("[중단] 기술분석 시트 접근 실패");
    return;
  }

  const lastRow = targetSheet.getLastRow();
  if (lastRow < 3) {
    console.log("[중단] 기술분석 시트 데이터 없음");
    return;
  }

  const props = PropertiesService.getScriptProperties().getProperties();
  const width = Math.max(
    Utils.COL_INDICES.stockLabel,
    Utils.COL_INDICES.opinion,
    Utils.COL_INDICES.entryPrice,
    Utils.COL_INDICES.entryDate,
    Utils.COL_INDICES.entryStrategy
  ) + 1;
  const rows = targetSheet.getRange(3, 1, lastRow - 2, width).getValues();

  const issues = [];
  const healthy = [];

  rows.forEach(function(row, index) {
    const ticker = String(row[Utils.COL_INDICES.stockName] || "").trim();
    const displayName = String(row[Utils.COL_INDICES.stockLabel] || "").trim();
    const opinion = String(row[Utils.COL_INDICES.opinion] || "").trim();
    if (!ticker || opinion !== "매수") return;

    const entryKey = "ENTRY_" + ticker;
    const rawEntry = String(props[entryKey] || "").trim();
    const parts = rawEntry ? rawEntry.split("|") : [];
    const entryPrice = parts[0] ? Number(parts[0]) : 0;
    const entryDate = parts[1] || "";
    const entryStrategy = parts[2] || "";

    const slotKeys = Object.keys(props)
      .filter(function(key) { return key.indexOf("SLOT_" + ticker + "_") === 0; })
      .sort();

    const sheetEntryPrice = String(row[Utils.COL_INDICES.entryPrice] || "").trim();
    const sheetEntryDateRaw = row[Utils.COL_INDICES.entryDate];
    const sheetEntryDate = sheetEntryDateRaw instanceof Date
      ? Utilities.formatDate(sheetEntryDateRaw, "Asia/Seoul", "yyyy-MM-dd")
      : String(sheetEntryDateRaw || "").trim();
    const sheetEntryStrategy = String(row[Utils.COL_INDICES.entryStrategy] || "").trim();

    const summary =
      ticker +
      (displayName ? " (" + displayName + ")" : "") +
      " | row " + (index + 3) +
      " | 시트 진입가=" + (sheetEntryPrice || "(빈칸)") +
      " | 시트 진입일=" + (sheetEntryDate || "(빈칸)") +
      " | 시트 진입전략=" + (sheetEntryStrategy || "(빈칸)") +
      " | ENTRY_=" + (rawEntry || "(없음)") +
      " | SLOT 수=" + slotKeys.length;

    if (!rawEntry || !(entryPrice > 0) || !entryDate || !entryStrategy) {
      issues.push(summary);
    } else {
      healthy.push(summary);
    }
  });

  console.log("========== 매수 의견 종목 ENTRY_ 점검 시작 ==========");
  console.log("[정상]");
  if (healthy.length === 0) console.log("없음");
  else healthy.forEach(function(line) { console.log(" - " + line); });

  console.log("[이상: 매수 의견인데 ENTRY_ 비어있음/불완전]");
  if (issues.length === 0) console.log("없음");
  else issues.forEach(function(line) { console.log(" - " + line); });

  console.log("[판정] " + (issues.length === 0 ? "정상" : ("이상 " + issues.length + "건")));
  console.log("========== 매수 의견 종목 ENTRY_ 점검 종료 ==========");
}
