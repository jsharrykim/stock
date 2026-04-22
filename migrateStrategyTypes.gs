/**
 * migrateStrategyTypes.gs
 *
 * 전략 그룹 재편(2026.04) 후 PropertiesService에 저장된
 * 구 전략 타입을 신규 타입으로 일괄 변환하는 1회성 마이그레이션 스크립트.
 *
 * 구 → 신 매핑:
 *   A (스퀴즈 저점)   → E
 *   B (BB 극단 저점)  → F
 *   C (골든크로스)    → A
 *   D (공황 저점)     → B
 *   squeeze           → E (구 문자열 키)
 *   ma200u            → F
 *   ma200d            → B
 *
 * 영향 범위:
 *   ENTRY_<ticker>            값의 3번째 파이프 필드 (strategyType)
 *   SLOT_<ticker>_<strategy>  키 이름의 strategy 접미사
 *   SLOT_SELL_<ticker>_<s>    키 이름의 strategy 접미사
 *
 * 실행 방법:
 *   1. Apps Script 에디터에서 `migrateStrategyTypes` 함수를 직접 실행
 *   2. 로그에서 변경 내역 확인 후 필요 시 `rollbackStrategyMigration` 으로 복구
 *
 * ⚠️  실행 전 스프레드시트 데이터를 백업해두세요.
 */

// ── 구 → 신 매핑 ──────────────────────────────────────────────────────────────
const LEGACY_MAP = {
  "squeeze": "E",
  "ma200u":  "F",
  "ma200d":  "B",
  "A":       "E",
  "B":       "F",
  "C":       "A",
  "D":       "B",
};

// 이미 신규 타입인 경우 변경 불필요
const NEW_TYPES = new Set(["E", "F", "A", "B", "C", "D"]);

function migrateStrategyTypes() {
  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const keys     = Object.keys(allProps);

  const toSet    = {};   // 새로 저장할 키-값
  const toDel    = [];   // 삭제할 키 (슬롯 키 이름 변경 시)
  const log      = [];   // 변경 로그

  // ── 1. ENTRY_ 처리: 값의 strategyType 필드 변환 ────────────────────────────
  keys.filter(k => k.startsWith("ENTRY_")).forEach(key => {
    const val   = allProps[key];
    const parts = val.split("|");
    if (parts.length < 3) return;

    const oldType = parts[2];
    const newType = LEGACY_MAP[oldType];
    if (!newType) return; // 이미 신규 타입이거나 알 수 없는 값 → 스킵

    parts[2] = newType;
    toSet[key] = parts.join("|");
    log.push(`ENTRY  | ${key.padEnd(25)} | "${oldType}" → "${newType}" | 값: ${val} → ${toSet[key]}`);
  });

  // ── 2. SLOT_ / SLOT_SELL_ 처리: 키 이름의 strategy 부분 변환 ───────────────
  keys.filter(k => k.startsWith("SLOT_")).forEach(key => {
    // 키 형식: SLOT_<ticker>_<strategy>  또는  SLOT_SELL_<ticker>_<strategy>
    const parts    = key.split("_");
    const oldType  = parts[parts.length - 1];   // 마지막 부분이 strategy
    const newType  = LEGACY_MAP[oldType];
    if (!newType) return;

    parts[parts.length - 1] = newType;
    const newKey = parts.join("_");

    toSet[newKey] = allProps[key];
    toDel.push(key);
    log.push(`SLOT   | ${key.padEnd(30)} → ${newKey} | 값: ${allProps[key]}`);
  });

  // ── 3. 실제 적용 ────────────────────────────────────────────────────────────
  if (Object.keys(toSet).length === 0 && toDel.length === 0) {
    console.log("[마이그레이션] 변경 대상 없음 — 이미 모두 신규 타입이거나 보유 종목 없음");
    return;
  }

  // 새 값/키 저장 (한 번에 최대 500개; 73종목 기준 훨씬 적으므로 단순 처리)
  Object.entries(toSet).forEach(([k, v]) => props.setProperty(k, v));
  // 구 슬롯 키 삭제 (새 키 저장 후 삭제)
  toDel.forEach(k => props.deleteProperty(k));

  // ── 4. 변경 내역 로그 출력 ──────────────────────────────────────────────────
  console.log(`\n════ 전략 타입 마이그레이션 완료 ════`);
  console.log(`변경 항목: ${log.length}건 (ENTRY ${keys.filter(k=>k.startsWith("ENTRY_")).filter(k=>LEGACY_MAP[(allProps[k]||"").split("|")[2]]).length}건, SLOT ${toDel.length}건)`);
  log.forEach(l => console.log(l));

  // ── 5. 마이그레이션 후 현재 보유 현황 출력 ──────────────────────────────────
  console.log(`\n════ 마이그레이션 후 보유 현황 ════`);
  const updatedProps = props.getProperties();
  const holdings = Object.entries(updatedProps)
    .filter(([k]) => k.startsWith("ENTRY_"))
    .map(([k, v]) => {
      const parts = v.split("|");
      return { ticker: k.replace("ENTRY_", ""), price: parts[0], date: parts[1], strategy: parts[2] || "-" };
    })
    .filter(h => Number(h.price) > 0);

  if (holdings.length === 0) {
    console.log("  현재 보유 종목 없음");
  } else {
    holdings.forEach(h =>
      console.log(`  ${h.ticker.padEnd(12)} | 진입가: ${h.price} | 진입일: ${h.date} | 전략: ${h.strategy}그룹`)
    );
  }
}


/**
 * 마이그레이션 롤백 (실행 직후에만 유효 — 구 키 값을 다시 쓰는 방식)
 * 실행 전 콘솔 로그를 저장해두어야 복구 가능.
 * 긴급 시에는 resetAllStockData() 로 전체 초기화하는 방법도 있음.
 */
function rollbackStrategyMigration() {
  const REVERSE_MAP = { "E": "A", "F": "B", "A": "C", "B": "D" };
  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const keys     = Object.keys(allProps);
  let count = 0;

  keys.filter(k => k.startsWith("ENTRY_")).forEach(key => {
    const parts = allProps[key].split("|");
    if (parts.length < 3) return;
    const rev = REVERSE_MAP[parts[2]];
    if (!rev) return;
    parts[2] = rev;
    props.setProperty(key, parts.join("|"));
    count++;
  });

  keys.filter(k => k.startsWith("SLOT_")).forEach(key => {
    const parts   = key.split("_");
    const oldType = parts[parts.length - 1];
    const rev     = REVERSE_MAP[oldType];
    if (!rev) return;
    parts[parts.length - 1] = rev;
    const newKey = parts.join("_");
    props.setProperty(newKey, allProps[key]);
    props.deleteProperty(key);
    count++;
  });

  console.log(`[롤백 완료] ${count}건 복구됨`);
}


/**
 * 마이그레이션 없이 현재 저장된 전략 타입 현황만 출력 (dry-run / 확인용)
 */
function previewStrategyMigration() {
  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const keys     = Object.keys(allProps);

  console.log(`\n════ 현재 보유 현황 (마이그레이션 전) ════`);
  const entries = keys.filter(k => k.startsWith("ENTRY_")).map(k => {
    const parts = allProps[k].split("|");
    return { key: k, ticker: k.replace("ENTRY_", ""), price: parts[0], date: parts[1], type: parts[2] || "-" };
  }).filter(e => Number(e.price) > 0);

  if (entries.length === 0) {
    console.log("  보유 종목 없음");
  } else {
    entries.forEach(e => {
      const newType = LEGACY_MAP[e.type];
      const arrow   = newType ? ` → ${newType}그룹 (변경 예정)` : ` (신규 타입, 변경 불필요)`;
      console.log(`  ${e.ticker.padEnd(12)} | 진입가: ${e.price} | 진입일: ${e.date} | 현재: ${e.type}그룹${arrow}`);
    });
  }

  console.log(`\n════ 슬롯 현황 ════`);
  const slots = keys.filter(k => k.startsWith("SLOT_") && !k.startsWith("SLOT_SELL_"));
  if (slots.length === 0) {
    console.log("  활성 슬롯 없음");
  } else {
    slots.forEach(k => {
      const parts   = k.split("_");
      const oldType = parts[parts.length - 1];
      const newType = LEGACY_MAP[oldType];
      const arrow   = newType ? ` → 키: ${k.replace(`_${oldType}`, `_${newType}`)} (변경 예정)` : ` (변경 불필요)`;
      console.log(`  ${k}${arrow} | 값: ${allProps[k]}`);
    });
  }
}

// ── 개별 종목 수동 정리 ────────────────────────────────────────────────────────

/**
 * 특정 종목의 잔존 ENTRY_ 키 및 SLOT_ 키 현황을 출력합니다.
 */
function inspectTicker(ticker) {
  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();

  console.log(`\n════ ${ticker} PropertiesService 현황 ════`);
  const relevant = Object.keys(allProps).filter(k => k.includes(ticker));
  if (relevant.length === 0) {
    console.log("  관련 키 없음");
  } else {
    relevant.forEach(k => console.log(`  ${k} = ${allProps[k]}`));
  }
}

/** 현대건설(000720) 현황 확인 — 파라미터 없이 바로 실행 */
function inspect_000720() {
  inspectTicker("000720");
}

/**
 * 현대건설(000720) stale ENTRY_ 정리 + SLOT_D → ENTRY_D로 승격.
 *
 * 실행 결과:
 *   - 기존 stale ENTRY_000720 (B그룹 잔존 데이터) 제거
 *   - SLOT_000720_D 의 진입 데이터로 ENTRY_000720 (D그룹) 재설정
 *   - SLOT_000720_D 삭제 (중복 방지)
 *
 * ※ 현대건설이 실제로 D그룹으로 추적돼야 할 경우에만 실행하세요.
 *   추적 자체를 원하지 않으면 fixHyundaiConstruction_reset() 을 대신 실행하세요.
 */
function fixHyundaiConstruction_promoteSlot() {
  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  const ticker   = "000720";

  const slotKey  = `SLOT_${ticker}_D`;
  const entryKey = `ENTRY_${ticker}`;

  const slotVal  = allProps[slotKey];
  if (!slotVal) {
    console.log(`[오류] ${slotKey} 없음 — 실행 취소`);
    return;
  }

  // SLOT_ 값 파싱: price|date|...
  const parts = slotVal.split("|");
  const price  = parts[0];
  const date   = parts[1] || "";

  const newEntryVal = `${price}|${date}|D`;
  console.log(`[기존 ENTRY_] ${allProps[entryKey] || "(없음)"}`);
  console.log(`[삭제] ${slotKey}`);
  console.log(`[설정] ${entryKey} = ${newEntryVal}`);

  props.setProperty(entryKey, newEntryVal);
  props.deleteProperty(slotKey);

  console.log("✅ 현대건설(000720) ENTRY_ D그룹 정상화 완료");
}

/**
 * 현대건설(000720) 관련 ENTRY_ 및 SLOT_ 전체 삭제 (추적 중단용).
 * 시트에서도 수동으로 의견을 "관망"으로 변경 필요.
 */
function fixHyundaiConstruction_reset() {
  const props  = PropertiesService.getScriptProperties();
  const ticker = "000720";

  const keys = Object.keys(PropertiesService.getScriptProperties().getProperties())
    .filter(k => k.includes(ticker));

  if (keys.length === 0) { console.log("관련 키 없음"); return; }

  keys.forEach(k => {
    props.deleteProperty(k);
    console.log(`[삭제] ${k}`);
  });
  console.log("✅ 현대건설(000720) 전체 정리 완료 — 시트에서 의견도 수동 변경 필요");
}
