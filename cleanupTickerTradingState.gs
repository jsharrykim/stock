/**
 * cleanupTickerTradingState.gs
 *
 * 현대건설(000720) 트레이딩 상태 잔존 키 정리용 1회성 스크립트.
 *
 * 실행 순서:
 * 1. preview_000720_trading_state_cleanup() 실행
 * 2. apply_000720_trading_state_cleanup() 실행
 *
 * 삭제 대상:
 * - ENTRY_/SELL_/EXIT_REASON_
 * - HOLD_*/A_HOLD_*
 * - UPPER_EXIT_ARM_/CYCLE_ENTRY_/REENTRY_COUNT_/SOLD_FLAG_
 * - SLOT_/SLOT_SELL_/SLOT_UPPER_EXIT_ARM_
 *
 * 삭제 제외:
 * - LRT_, RCCI_ 같은 계산 캐시
 */

function preview_000720_trading_state_cleanup() {
  _run_000720_trading_state_cleanup_(false);
}

function apply_000720_trading_state_cleanup() {
  _run_000720_trading_state_cleanup_(true);
}

function _run_000720_trading_state_cleanup_(shouldDelete) {
  const props = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const ticker = "000720";
  const keys = _collect_000720_trading_state_keys_(allProps);

  console.log("");
  console.log("========== 트레이딩 상태 정리 시작 ==========");
  console.log("[대상] 현대건설(000720)");
  console.log(`[모드] ${shouldDelete ? "실삭제" : "미리보기"}`);
  console.log("");

  if (keys.length === 0) {
    console.log("삭제 대상 키 없음");
    console.log("========== 트레이딩 상태 정리 종료 ==========");
    return;
  }

  keys.forEach(key => {
    console.log(`${shouldDelete ? "[삭제]" : "[대상]"} ${key} = ${allProps[key]}`);
    if (shouldDelete) props.deleteProperty(key);
  });

  console.log("");
  console.log(`[완료] ${shouldDelete ? "삭제" : "확인"} 대상 ${keys.length}건`);
  console.log("========== 트레이딩 상태 정리 종료 ==========");
}

function _collect_000720_trading_state_keys_(allProps) {
  const ticker = "000720";
  const exactKeys = [
    `ENTRY_${ticker}`,
    `SELL_${ticker}`,
    `EXIT_REASON_${ticker}`,
    `HOLD_ANCHOR_${ticker}`,
    `HOLD_WATCH_${ticker}`,
    `A_HOLD_ANCHOR_${ticker}`,
    `A_HOLD_WATCH_${ticker}`,
    `UPPER_EXIT_ARM_${ticker}`,
    `CYCLE_ENTRY_${ticker}`,
    `REENTRY_COUNT_${ticker}`,
    `SOLD_FLAG_${ticker}`,
  ];

  const dynamicPrefixes = [
    `SLOT_${ticker}_`,
    `SLOT_SELL_${ticker}_`,
    `SLOT_UPPER_EXIT_ARM_${ticker}_`,
  ];

  const keys = [];

  exactKeys.forEach(key => {
    if (Object.prototype.hasOwnProperty.call(allProps, key)) keys.push(key);
  });

  Object.keys(allProps)
    .filter(key => dynamicPrefixes.some(prefix => key.indexOf(prefix) === 0))
    .sort()
    .forEach(key => keys.push(key));

  return keys.sort();
}
