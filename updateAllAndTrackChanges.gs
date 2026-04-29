var PIPELINE_STATE_KEY = "UPDATE_ALL_PIPELINE_STATE";
var PIPELINE_LOCK_WAIT_MS = 5000;

function updateAllAndTrackChanges() {
  const startMs = Date.now();
  const elapsed = () => `${((Date.now() - startMs) / 1000).toFixed(1)}s`;
  const lock = LockService.getScriptLock();

  if (!lock.tryLock(PIPELINE_LOCK_WAIT_MS)) {
    console.log(`[트리거 건너뜀] 이전 실행이 아직 진행 중입니다. (${new Date().toLocaleString("ko-KR", { timeZone: "Asia/Seoul" })})`);
    return;
  }

  try {
    console.log(`[트리거 시작] updateAllAndTrackChanges (${new Date().toLocaleString("ko-KR", { timeZone: "Asia/Seoul" })})`);
    setPipelineState_("running", "updateAllAndTrackChanges", null);

    console.log(`[시작] batchPrefetchPrices`);
    batchPrefetchPrices();
    console.log(`[완료] RSI/CCI 캐시 적재 (${elapsed()})`);

    console.log(`[시작] updateLRTrendlineAll`);
    updateLRTrendlineAll();
    console.log(`[완료] LR 추세선 캐시 적재 (${elapsed()})`);

    SpreadsheetApp.flush();
    console.log(`[flush] 시트 쓰기 커밋 완료 (${elapsed()})`);

    console.log(`[대기] 데이터 반영 대기 30초`);
    Utilities.sleep(30000);
    console.log(`[대기 완료] 나스닥 고점 상태 점검 시작`);

    // 고점 경고 상태는 같은 실행 안에서 바로 투자의견/매도 판단에 반영되어야 한다.
    // 기존에는 checkNasdaqMASignals()가 trackChanges() 뒤에 있어, 경고 메일이 와도
    // 실제 보유 종목 청산이 다음 트리거로 밀리는 순서 버그가 발생할 수 있었다.
    console.log(`[시작] checkNasdaqMASignals`);
    checkNasdaqMASignals();
    console.log(`[완료] checkNasdaqMASignals (${elapsed()})`);

    console.log(`[시작] 투자의견 업데이트 준비`);

    // updateInvestmentOpinion 실행 전 현재 시트 상태를 trackChanges 비교 기준으로 저장.
    // batchPrefetchPrices / updateLRTrendlineAll 실행 도중 Properties가 초기화되는 경우에도
    // lastValues 가 항상 올바른 "변경 전" 상태를 갖도록 보장한다.
    snapshotOpinionsBeforeUpdate_();
    console.log(`[스냅샷] 투자의견 비교 기준 저장 완료 (${elapsed()})`);

    console.log(`[시작] updateInvestmentOpinion`);
    updateInvestmentOpinion();
    console.log(`[완료] 투자의견 업데이트 (${elapsed()})`);

    console.log(`[대기] 투자의견 반영 대기 20초`);
    Utilities.sleep(20000);
    console.log(`[대기 완료] 투자의견 변경 감지 및 이메일 알림 시작`);

    console.log(`[시작] trackChanges`);
    trackChanges();
    console.log(`[완료] trackChanges (${elapsed()})`);

    setPipelineState_("completed", "done", null);
    console.log(`[트리거 종료] 모든 작업 완료 (총 ${elapsed()})`);
  } catch (e) {
    setPipelineState_("failed", "updateAllAndTrackChanges", String(e && e.message ? e.message : e));
    console.log(`[FATAL] updateAllAndTrackChanges 실패: ${e && e.stack ? e.stack : e}`);
    throw e;
  } finally {
    lock.releaseLock();
  }
}

/**
 * updateInvestmentOpinion 실행 직전에 시트의 현재 투자의견을 lastValues로 저장.
 * trackChanges가 항상 "변경 전" 기준을 올바르게 갖도록 보장한다.
 */
function snapshotOpinionsBeforeUpdate_() {
  const ss        = SpreadsheetApp.getActiveSpreadsheet();
  const techSheet = ss.getSheetByName("기술분석");
  if (!techSheet) { console.log("[스냅샷] 기술분석 시트 없음 — 스킵"); return; }

  const lastRow = techSheet.getLastRow();
  if (lastRow < 3) { console.log("[스냅샷] 데이터 없음 — 스킵"); return; }

  const numRows  = lastRow - 2;
  const names    = techSheet.getRange(3, 1, numRows, 1).getValues();  // A열: 종목명 (COL_INDICES.stockName = 0)
  const opinions = techSheet.getRange(3, 4, numRows, 1).getValues();  // D열: 투자의견 (COL_INDICES.opinion = 3)

  const VALID = new Set(["매수", "매도", "관망"]);
  const lastValidOpinions = {};
  names.forEach(([name], i) => {
    const n  = String(name).trim();
    const op = String(opinions[i][0]).trim();
    if (!n || !VALID.has(op)) return;
    lastValidOpinions[n] = { opinion: op, reason: "트리거 실행 전 기준값" };
  });

  PropertiesService.getScriptProperties().setProperty("lastValues", JSON.stringify({
    initialized:       true,
    vixToday:          0,
    event:             "당분간 없음",
    lastValidOpinions,
  }));
  console.log(`[스냅샷] ${Object.keys(lastValidOpinions).length}종목 기준값 저장 완료`);
}

function setPipelineState_(status, stage, error) {
  const props = PropertiesService.getScriptProperties();
  const nowIso = new Date().toISOString();
  const next = {
    status: status,
    stage: stage || null,
    updatedAt: nowIso,
    error: error || null
  };
  props.setProperty(PIPELINE_STATE_KEY, JSON.stringify(next));
}
