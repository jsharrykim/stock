/**
 * 2026-04-23 임시 선별 리셋
 *
 * 목적:
 * - 아래 지정한 종목들만 시스템/트레이딩로그/기술분석 entry 상태를 한 번에 정리한다.
 * - ASTS, LITE 는 유지 대상이라 건드리지 않는다.
 *
 * 실행:
 * - 이 함수 하나만 수동 실행하면 된다.
 *
 * 정리 범위:
 * 1) Script Properties
 *    - ENTRY_/SELL_/EXIT_REASON_/HOLD_/CYCLE_/REENTRY_/SLOT_* 등 관련 상태 삭제
 * 2) 트레이딩로그
 *    - 해당 종목의 A:F 값 전체 삭제
 * 3) 기술분석 시트
 *    - 투자의견 -> 관망
 *    - 진입가 / 진입일 / 진입전략 -> 빈칸
 * 4) lastValues
 *    - 다음 trackChanges 에서 관망 -> 매수 재감지가 가능하도록 대상 종목만 관망으로 스냅샷 보정
 */
function resetSelectedTradingState20260423() {
  const TARGETS = [
    ["000720", "현대건설"],
    ["079550", "LIG디펜스앤에어로스페이스"],
    ["247540", "에코프로비엠"],
    ["TE", "T1 Energy Inc"],
    ["GLW", "Corning, Inc"],
    ["NBIS", "Nebius Group NV"],
    ["042700", "한미반도체"],
    ["000270", "기아차"],
    ["004020", "현대제철"],
    ["086280", "현대글로비스"],
    ["005380", "현대차"],
    ["SNDK", "Sandisk Corp"],
    ["PL", "Planet Labs PBC"],
    ["LPTH", "Lightpath Technologies, Inc"],
    ["LRCX", "Lam Research Corp"],
  ];
  const targetSet = new Set(TARGETS.map(function(item) { return item[0]; }));

  const OPINION_COL = 4;        // D
  const ENTRY_PRICE_COL = 53;   // BA
  const ENTRY_DATE_COL = 54;    // BB
  const ENTRY_STRATEGY_COL = 55; // BC

  const activeSpreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const localTechSheet = activeSpreadsheet.getSheetByName("기술분석");
  if (!localTechSheet) throw new Error("활성 스프레드시트에 기술분석 시트가 없습니다.");

  const spreadsheetId = String(localTechSheet.getRange("I1").getValue()).trim();
  if (!spreadsheetId) throw new Error("기술분석 시트 I1에 트레이딩로그 대상 스프레드시트 ID가 없습니다.");

  const targetSpreadsheet = SpreadsheetApp.openById(spreadsheetId);
  const targetTechSheet = targetSpreadsheet.getSheetByName("기술분석");
  const logSheet = targetSpreadsheet.getSheetByName("트레이딩로그");
  if (!targetTechSheet) throw new Error("대상 스프레드시트에 기술분석 시트가 없습니다.");

  const props = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();

  console.log("========== 선별 트레이딩 상태 리셋 시작 ==========");
  console.log("[대상] " + TARGETS.map(function(item) { return item[0] + "(" + item[1] + ")"; }).join(", "));
  console.log("[유지] ASTS, LITE");

  // 1) 시스템 상태 삭제
  let deletedPropCount = 0;
  TARGETS.forEach(function(item) {
    const stockName = item[0];
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
      "SOLD_FLAG_" + stockName,
    ];

    exactKeys.forEach(function(key) {
      if (Object.prototype.hasOwnProperty.call(allProps, key)) {
        props.deleteProperty(key);
        deletedPropCount++;
      }
    });

    Object.keys(allProps)
      .filter(function(key) {
        return key.indexOf("SLOT_" + stockName + "_") === 0
          || key.indexOf("SLOT_SELL_" + stockName + "_") === 0
          || key.indexOf("SLOT_UPPER_EXIT_ARM_" + stockName + "_") === 0;
      })
      .forEach(function(key) {
        props.deleteProperty(key);
        deletedPropCount++;
      });
  });
  console.log("[1단계] Script Properties 삭제: " + deletedPropCount + "건");

  // 2) 기술분석 시트 리셋
  const techLastRow = targetTechSheet.getLastRow();
  const techRowCount = Math.max(techLastRow - 2, 0);
  const techRows = techRowCount > 0
    ? targetTechSheet.getRange(3, 1, techRowCount, ENTRY_STRATEGY_COL).getValues()
    : [];
  let techResetCount = 0;

  techRows.forEach(function(row, index) {
    const stockName = String(row[0]).trim();
    if (!targetSet.has(stockName)) return;
    const rowNumber = index + 3;
    targetTechSheet.getRange(rowNumber, OPINION_COL).setValue("관망");
    targetTechSheet.getRange(rowNumber, ENTRY_PRICE_COL).setValue("");
    targetTechSheet.getRange(rowNumber, ENTRY_DATE_COL).setValue("");
    targetTechSheet.getRange(rowNumber, ENTRY_STRATEGY_COL).setValue("");
    techResetCount++;
    console.log("[2단계] 기술분석 리셋: " + stockName + " / row " + rowNumber + " / 의견=관망 / 진입가·진입일·진입전략 삭제");
  });
  console.log("[2단계] 기술분석 리셋 완료: " + techResetCount + "행");

  // 3) 트레이딩로그 A:F 삭제
  let logResetCount = 0;
  if (logSheet && logSheet.getLastRow() >= 3) {
    const logLastRow = logSheet.getLastRow();
    const logRows = logSheet.getRange(3, 1, logLastRow - 2, 6).getValues();
    logRows.forEach(function(row, index) {
      const stockName = String(row[0]).trim();
      if (!targetSet.has(stockName)) return;
      const rowNumber = index + 3;
      logSheet.getRange(rowNumber, 1, 1, 6).clearContent();
      logResetCount++;
      console.log("[3단계] 트레이딩로그 삭제: " + stockName + " / row " + rowNumber + " / A:F");
    });
  }
  console.log("[3단계] 트레이딩로그 리셋 완료: " + logResetCount + "행");

  // 4) lastValues 보정
  const rawLastValues = props.getProperty("lastValues");
  let lastValues;
  try {
    lastValues = rawLastValues
      ? JSON.parse(rawLastValues)
      : { initialized: true, vixToday: 0, event: "당분간 없음", lastValidOpinions: {} };
  } catch (e) {
    lastValues = { initialized: true, vixToday: 0, event: "당분간 없음", lastValidOpinions: {} };
  }

  if (!lastValues || typeof lastValues !== "object") {
    lastValues = { initialized: true, vixToday: 0, event: "당분간 없음", lastValidOpinions: {} };
  }
  if (!lastValues.lastValidOpinions || typeof lastValues.lastValidOpinions !== "object") {
    lastValues.lastValidOpinions = {};
  }

  Object.keys(lastValues.lastValidOpinions).forEach(function(stockName) {
    const value = lastValues.lastValidOpinions[stockName];
    if (typeof value === "string") {
      lastValues.lastValidOpinions[stockName] = { opinion: value, reason: value + " 조건 충족" };
    }
  });

  TARGETS.forEach(function(item) {
    lastValues.lastValidOpinions[item[0]] = { opinion: "관망", reason: "선별 리셋" };
  });
  lastValues.initialized = true;
  if (!Object.prototype.hasOwnProperty.call(lastValues, "vixToday")) lastValues.vixToday = 0;
  if (!Object.prototype.hasOwnProperty.call(lastValues, "event")) lastValues.event = "당분간 없음";
  props.setProperty("lastValues", JSON.stringify(lastValues));
  console.log("[4단계] lastValues 보정 완료: " + TARGETS.length + "종목 -> 관망");

  SpreadsheetApp.flush();

  console.log("[완료] 선별 리셋 종료");
  console.log("[참고] ASTS, LITE 는 유지됨");
  console.log("========== 선별 트레이딩 상태 리셋 종료 ==========");
}
