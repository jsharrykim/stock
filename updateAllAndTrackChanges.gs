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
    console.log(`[대기 완료] 투자의견 업데이트 시작`);

    console.log(`[시작] pre-update opinion snapshot`);
    Utils.snapshotOpinionsForTracking();
    console.log(`[완료] pre-update opinion snapshot (${elapsed()})`);

    console.log(`[시작] updateInvestmentOpinion`);
    updateInvestmentOpinion();
    console.log(`[완료] 투자의견 업데이트 (${elapsed()})`);

    console.log(`[대기] 투자의견 반영 대기 20초`);
    Utilities.sleep(20000);
    console.log(`[대기 완료] 투자의견 변경 감지 및 이메일 알림 시작`);

    console.log(`[시작] trackChanges`);
    trackChanges();
    console.log(`[완료] trackChanges (${elapsed()})`);

    console.log(`[시작] checkNasdaqMASignals`);
    checkNasdaqMASignals();
    console.log(`[완료] checkNasdaqMASignals (${elapsed()})`);

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
