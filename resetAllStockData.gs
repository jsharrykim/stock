function resetAllStockData() {

  // ── 1. 대상 스프레드시트 접근 및 종목 목록 시트에서 읽기 ──────────────
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const localSheet  = spreadsheet.getSheetByName("기술분석");
  if (!localSheet) { console.log("[오류] 기술분석 시트 없음"); return; }

  const spreadsheetId = String(localSheet.getRange("I1").getValue()).trim();
  let targetSheet, logSheet;
  try {
    const targetSS = SpreadsheetApp.openById(spreadsheetId);
    targetSheet    = targetSS.getSheetByName("기술분석");
    logSheet       = targetSS.getSheetByName("트레이딩로그");
  } catch (e) {
    console.log(`[오류] 대상 스프레드시트 접근 실패: ${e}`);
    return;
  }

  if (!targetSheet) { console.log("[오류] 대상 기술분석 시트 없음"); return; }

  const lastRow  = targetSheet.getLastRow();
  const rowCount = lastRow - 2;
  if (rowCount < 1) { console.log("[스킵] 데이터 없음"); return; }

  // 시트 A열에서 종목 코드 수집
  const data    = targetSheet.getRange(3, 1, rowCount, 1).getValues();
  const TICKERS = data.map(r => String(r[0]).trim()).filter(Boolean);
  console.log(`[종목 파악] 시트에서 ${TICKERS.length}개 종목 확인`);

  // ── 2. PropertiesService 전체 스캔 삭제 ───────────────────────────────
  const props = PropertiesService.getScriptProperties();
  const cache = CacheService.getScriptCache();

  const PROP_PREFIXES = ["ENTRY_","SELL_","EXIT_REASON_","SOLD_FLAG_","REENTRY_COUNT_","CYCLE_ENTRY_","LRT_","HOLD_ANCHOR_","HOLD_WATCH_","A_HOLD_ANCHOR_","A_HOLD_WATCH_"];
  const allProps = props.getProperties();
  let propDeleted = 0;
  Object.keys(allProps).forEach(k => {
    if (k === "lastValues" || PROP_PREFIXES.some(p => k.startsWith(p))) {
      props.deleteProperty(k);
      propDeleted++;
    }
  });

  // 캐시도 시트 종목 기준으로 제거
  const CACHE_PREFIXES = ["PRICE_US_","PRICE_KR_","US_","KR_","LR_KR_","LR_US_"];
  const CACHE_SUFFIXES = ["","_14","_155"];
  const cacheKeys = [];
  TICKERS.forEach(sym => {
    CACHE_PREFIXES.forEach(p => CACHE_SUFFIXES.forEach(s => cacheKeys.push(p + sym + s)));
  });
  for (let i = 0; i < cacheKeys.length; i += 100) cache.removeAll(cacheKeys.slice(i, i + 100));

  console.log(`[1단계] Props 삭제: ${propDeleted}건 / Cache 키 제거: ${cacheKeys.length}개`);

  // ── 3. 투자의견(D열) → 관망, 진입가/진입일 → 빈칸 ────────────────────
  const COL_OPINION     = 4;   // D열
  const COL_ENTRY_PRICE = 53;  // BA열
  const COL_ENTRY_DATE  = 54;  // BB열

  const rows = [];
  data.forEach((row, i) => {
    if (String(row[0]).trim()) rows.push(i + 3);
  });

  rows.forEach(r => {
    targetSheet.getRange(r, COL_OPINION).setValue("관망");
    targetSheet.getRange(r, COL_ENTRY_PRICE).setValue("");
    targetSheet.getRange(r, COL_ENTRY_DATE).setValue("");
  });

  console.log(`[2단계] 시트 초기화 — ${rows.length}개 행: 투자의견→관망, 진입가/진입일→빈칸`);

  // ── 4. 트레이딩로그 A~F열만 값 초기화 (서식·G열 이후 수식 유지) ──────
  if (logSheet && logSheet.getLastRow() > 1) {
    const logLastRow = logSheet.getLastRow();
    logSheet.getRange(2, 1, logLastRow - 1, 6).clearContent();
    console.log(`[3단계] 트레이딩로그 A~F열 값 초기화 완료 (${logLastRow - 1}행, 서식 유지)`);
  } else {
    console.log(`[3단계] 트레이딩로그 — 삭제할 데이터 없음`);
  }

  // ── 5. lastValues 초기화 — 전 종목 "관망" 기준으로 세팅 ──────────────
  // lastValues가 없으면 trackChanges가 "초기 실행"으로 판단해 매수 시그널을 무시함.
  // 리셋 후 다음 실행에서 관망→매수 변경이 정상 감지되도록 관망 기준값을 명시 저장.
  const initialOpinions = {};
  TICKERS.forEach(sym => {
    initialOpinions[sym] = { opinion: "관망", reason: "리셋 후 초기값" };
  });
  props.setProperty("lastValues", JSON.stringify({
    data: [["reset"]],
    vixToday: 0,
    event: "당분간 없음",
    lastValidOpinions: initialOpinions
  }));
  console.log(`[4단계] lastValues 초기화 — ${TICKERS.length}개 종목 관망 기준으로 세팅`);

  console.log(`[리셋 완료] NasdaqPeakSellState는 유지됨`);
}
