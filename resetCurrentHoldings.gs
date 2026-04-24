/**
 * resetCurrentHoldings.gs
 *
 * 현재 보유 중인 모든 종목의 상태를 초기화하는 일회성 스크립트.
 *
 * 수행 내용:
 *  1. Script Properties에서 ENTRY_, SLOT_, SELL_, HOLD_ANCHOR_, HOLD_WATCH_,
 *     A_HOLD_ANCHOR_, A_HOLD_WATCH_, UPPER_EXIT_ARM_ 키 전부 삭제
 *  2. "기술분석" 시트의 진입가(BA열), 진입일(BB열), 진입전략(BC열) 컬럼 클리어
 *  3. "트레이딩로그" 시트에서 매도 기록이 없는 열린 매수 행 삭제
 *
 * ⚠️ 이 함수는 직접 실행 전용입니다. 트리거로 등록하지 마십시오.
 */

/**
 * 슬롯 시스템 마이그레이션 — 배포 후 최초 1회만 실행
 *
 * 수행 내용:
 *  1. 기존 ENTRY_* 키를 읽어 SLOTS_ 배열에 "migrated" 슬롯으로 등록
 *     → processMultiSlots가 동일 전략 슬롯을 중복 생성하는 것을 방지
 *  2. 레거시 SLOT_{종목}_{전략} 키 삭제 (구 형식 잔재 정리)
 *
 * ⚠️ 직접 실행 전용. 트리거 등록 금지.
 */
function migrateToSlotSystem() {
  console.log("========== 슬롯 시스템 마이그레이션 시작 ==========");

  const props    = PropertiesService.getScriptProperties();
  const allProps = props.getProperties();
  let migratedCount = 0;
  let legacyDeleted = 0;

  // ── 1. ENTRY_* → SLOTS_ 배열에 migrated 슬롯 등록 ─────────────────────────
  Object.entries(allProps).forEach(([key, val]) => {
    if (!key.startsWith("ENTRY_")) return;
    const stockName = key.substring("ENTRY_".length);
    if (!stockName) return;

    // 이미 SLOTS_ 있으면 건너뜀
    if (allProps[`SLOTS_${stockName}`]) return;

    // ENTRY_ 파싱: "price|dateStr|strategy"
    const parts = (val || "").split("|");
    const price = Number(parts[0]) || 0;
    if (!price) return;

    const strategy = /^[A-F]$/.test(parts[2]) ? parts[2] : "A";
    const dateStr  = parts[1] || Utilities.formatDate(new Date(), "Asia/Seoul", "yyyy-MM-dd");

    const slot = { id: `${strategy}_migrated`, strategy, price, date: dateStr };
    props.setProperty(`SLOTS_${stockName}`, JSON.stringify([slot]));
    migratedCount++;
    console.log(`[마이그레이션] ${stockName}: ${strategy}그룹 진입가 ${price} → SLOTS_ 등록`);
  });

  // ── 2. 레거시 SLOT_{종목}_{전략} 키 삭제 ──────────────────────────────────
  Object.keys(allProps).forEach(key => {
    // SLOT_{종목}_{전략} 형식 (SELL, UPPER_EXIT_ARM, SLOTS_ 는 제외)
    if (!key.startsWith("SLOT_")) return;
    if (key.startsWith("SLOT_SELL_")) return;
    if (key.startsWith("SLOT_UPPER_EXIT_ARM_")) return;
    if (key.startsWith("SLOTS_")) return;
    // 나머지는 레거시 단일 키
    props.deleteProperty(key);
    legacyDeleted++;
    console.log(`[레거시 정리] 삭제: ${key}`);
  });

  console.log(`========== 마이그레이션 완료 — ENTRY_ ${migratedCount}건 등록, 레거시 ${legacyDeleted}건 삭제 ==========`);
}
function resetCurrentHoldings() {
  console.log("========== 보유 종목 상태 초기화 시작 ==========");

  // ── 1. Script Properties 전체 삭제 (GROQ_API_KEY 등 설정 키만 보존) ─────────
  const props      = PropertiesService.getScriptProperties();
  const allProps   = props.getProperties();
  const PRESERVE   = new Set(["GROQ_API_KEY"]);
  let deletedPropCount = 0;
  Object.keys(allProps).forEach(key => {
    if (PRESERVE.has(key)) return;
    props.deleteProperty(key);
    deletedPropCount++;
  });
  console.log(`[Script Properties] 전체 ${deletedPropCount}개 키 삭제 완료 (보존: ${[...PRESERVE].join(", ")})`);


  // ── 2. 기술분석 시트 진입 정보 컬럼 클리어 ───────────────────────────────
  const ss           = SpreadsheetApp.getActiveSpreadsheet();
  const techSheet    = ss.getSheetByName("기술분석");
  if (techSheet) {
    const lastRow = techSheet.getLastRow();
    if (lastRow >= 3) {
      const dataRows = lastRow - 2;
      const C = Utils.COL_INDICES;
      // 진입가 / 진입일 / 진입전략 — 인덱스는 0-based, 시트 열은 1-based
      const entryPriceCol    = C.entryPrice    + 1;
      const entryDateCol     = C.entryDate     + 1;
      const entryStrategyCol = C.entryStrategy + 1;

      techSheet.getRange(3, entryPriceCol,    dataRows, 1).clearContent();
      techSheet.getRange(3, entryDateCol,     dataRows, 1).clearContent();
      techSheet.getRange(3, entryStrategyCol, dataRows, 1).clearContent();
      SpreadsheetApp.flush();
      console.log(`[기술분석 시트] 진입가/진입일/진입전략 ${dataRows}행 클리어 완료`);
    }
  } else {
    console.log("[기술분석 시트] 시트를 찾을 수 없습니다. 건너뜁니다.");
  }

  // ── 3. 트레이딩로그 열린 매수 행 삭제 ────────────────────────────────────
  const logSheet = ss.getSheetByName("트레이딩로그");
  if (logSheet) {
    const lastRow = logSheet.getLastRow();
    if (lastRow >= 3) {
      const data = logSheet.getRange(3, 1, lastRow - 2, 6).getValues();
      const rowsToDelete = [];
      for (let i = data.length - 1; i >= 0; i--) {
        const buyPrice   = data[i][2];
        const sellDate   = data[i][3];
        const hasBuy     = buyPrice !== "" && buyPrice !== null && buyPrice !== 0;
        const hasSell    = sellDate !== "" && sellDate !== null;
        // 매수는 있고 매도 날짜가 없으면 열린 포지션
        if (hasBuy && !hasSell) {
          rowsToDelete.push(i + 3); // 시트 행 번호 (헤더 2행 + 1-based)
        }
      }
      rowsToDelete.forEach(rowNum => logSheet.deleteRow(rowNum));
      console.log(`[트레이딩로그] 열린 매수 ${rowsToDelete.length}행 삭제 완료`);
    }
  } else {
    console.log("[트레이딩로그 시트] 시트를 찾을 수 없습니다. 건너뜁니다.");
  }

  // ── 4. 투자의견 '관망' 일괄 변경 + 관망 상태를 lastValues 스냅샷으로 저장 ──
  if (techSheet) {
    const lastRow = techSheet.getLastRow();
    if (lastRow >= 3) {
      const C          = Utils.COL_INDICES;
      const opinionCol = C.opinion + 1;
      const nameCol    = C.stockName + 1;
      const dataRows   = lastRow - 2;

      // 투자의견 '매수' → '관망' 일괄 변경
      const opinions   = techSheet.getRange(3, opinionCol, dataRows, 1).getValues();
      const updates    = opinions.map(([v]) => [v === "매수" ? "관망" : v]);
      techSheet.getRange(3, opinionCol, dataRows, 1).setValues(updates);
      SpreadsheetApp.flush();
      const changedCount = opinions.filter(([v]) => v === "매수").length;
      console.log(`[기술분석 시트] 투자의견 '매수' → '관망' ${changedCount}종목 변경 완료`);

      // 변경 후 전체 종목명을 읽어서 "전체 관망" 스냅샷을 lastValues로 저장
      // → 다음 trackChanges 실행 시 updateInvestmentOpinion이 바꾼 매수 종목이 변경으로 감지됨
      const names = techSheet.getRange(3, nameCol, dataRows, 1).getValues();
      const lastValidOpinions = {};
      names.forEach(([name], i) => {
        const n = String(name).trim();
        if (!n) return;
        const op = updates[i][0];
        if (Utils.isValidOpinion(op)) {
          lastValidOpinions[n] = { opinion: op, reason: "리셋 기준값" };
        }
      });
      props.setProperty("lastValues", JSON.stringify({
        initialized:       true,
        vixToday:          0,
        event:             "당분간 없음",
        lastValidOpinions,
      }));
      console.log(`[lastValues] 전체 관망 스냅샷 저장 완료 (${Object.keys(lastValidOpinions).length}종목)`);
    }
  }

  console.log("========== 보유 종목 상태 초기화 완료 ==========");
}

/**
 * 시스템상 현재 보유 상태 전체 리셋
 *
 * 용도:
 * - 잘못 꼬인 ENTRY_/SLOTS_/SELL_/복원 상태 전부 초기화
 * - 기술분석 시트의 진입 정보(진입가/진입일/진입전략) 초기화
 * - 트레이딩로그의 "열린 매수" 행 삭제
 * - lastValues 를 전체 관망 기준으로 다시 저장
 *
 * 실행 후:
 * - 다음 `updateAllAndTrackChanges()` 실행부터
 *   현재 시트 조건 기준으로 신규 진입/변경을 다시 정상 감지함
 *
 * ⚠️ 직접 실행 전용
 */
function resetAllPortfolioState() {
  console.log("========== 전체 포트폴리오 상태 리셋 시작 ==========");
  resetCurrentHoldings();
  console.log("========== 전체 포트폴리오 상태 리셋 종료 ==========");
}

/** 임시 확인용 — SLOTS_ / ENTRY_ 키 상태 출력 */
function checkSlotKeys() {
  const allProps = PropertiesService.getScriptProperties().getProperties();
  const slots  = Object.entries(allProps).filter(([k]) => k.startsWith("SLOTS_"));
  const entries = Object.entries(allProps).filter(([k]) => k.startsWith("ENTRY_"));

  console.log(`===== ENTRY_ 키 (${entries.length}건) =====`);
  entries.forEach(([k, v]) => console.log(`  ${k} = ${v}`));

  console.log(`===== SLOTS_ 키 (${slots.length}건) =====`);
  if (slots.length === 0) {
    console.log("  (없음)");
  } else {
    slots.forEach(([k, v]) => {
      try {
        const arr = JSON.parse(v);
        console.log(`  ${k} → ${arr.length}개 슬롯`);
        arr.forEach((s, i) => console.log(`    [${i}] id=${s.id}, strategy=${s.strategy}, price=${s.price}, date=${s.date}`));
      } catch (e) {
        console.log(`  ${k} = (파싱 오류) ${v}`);
      }
    });
  }
}

/**
 * 2026-04-24 오탐 재진입 정리
 *
 * 대상:
 * - 000270(E)
 * - SNDK(D)
 *
 * 수행 내용:
 *  1. migrated 슬롯이 있으면 그것만 남기고 같은 전략의 당일 중복 슬롯 제거
 *  2. 잘못 추가된 열린 매수 로그(당일, 동일 전략) 삭제
 *  3. REENTRY_COUNT_ 초기화
 *
 * ⚠️ 이번 오탐 건 전용. 직접 실행 전용.
 */
function cleanupFalseReentry_20260424() {
  const props = PropertiesService.getScriptProperties();
  const targets = [
    { stockName: "000270", strategy: "E", buyDate: "2026-04-24" },
    { stockName: "SNDK",   strategy: "D", buyDate: "2026-04-24" }
  ];

  console.log("========== 오탐 재진입 정리 시작 ==========");

  targets.forEach(target => {
    const slotKey = `SLOTS_${target.stockName}`;
    const raw = props.getProperty(slotKey);
    if (!raw) {
      console.log(`[정리 스킵] ${target.stockName}: SLOTS_ 없음`);
      return;
    }

    let slots;
    try {
      slots = JSON.parse(raw);
    } catch (e) {
      console.log(`[정리 실패] ${target.stockName}: SLOTS_ 파싱 오류 ${e}`);
      return;
    }

    const sameStrategySlots = slots.filter(s => s && s.strategy === target.strategy);
    if (sameStrategySlots.length <= 1) {
      console.log(`[정리 스킵] ${target.stockName}: ${target.strategy}그룹 중복 슬롯 없음`);
    } else {
      const migratedSlot = sameStrategySlots.find(s => String(s.id || "").endsWith("_migrated"));
      const keepId = migratedSlot
        ? migratedSlot.id
        : sameStrategySlots
            .slice()
            .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")))[0].id;

      const filtered = slots.filter(s => !(s && s.strategy === target.strategy && s.id !== keepId));
      props.setProperty(slotKey, JSON.stringify(filtered));
      console.log(`[슬롯 정리] ${target.stockName}: ${target.strategy}그룹 ${sameStrategySlots.length}개 → 1개 유지 (${keepId})`);
    }

    props.deleteProperty(`REENTRY_COUNT_${target.stockName}`);

    const logSheet = (typeof Utils !== "undefined" && typeof Utils.getTradingLogSheet === "function")
      ? Utils.getTradingLogSheet()
      : null;
    if (!logSheet) {
      console.log(`[로그 정리 스킵] ${target.stockName}: 트레이딩로그 시트 접근 불가`);
      return;
    }

    const strategyLabel = typeof strategyDisplayName === "function"
      ? strategyDisplayName(target.strategy)
      : target.strategy;
    const normalizeLabel = l => String(l || "").replace(/^[A-F]\.\s*/, "").trim();
    const lastRow = logSheet.getLastRow();
    if (lastRow < 3) {
      console.log(`[로그 정리 스킵] ${target.stockName}: 트레이딩로그 데이터 없음`);
      return;
    }

    const data = logSheet.getRange(3, 1, lastRow - 2, 6).getValues();
    const toDateKey = value => {
      if (!value) return "";
      if (Object.prototype.toString.call(value) === "[object Date]" && !isNaN(value.getTime())) {
        return Utilities.formatDate(value, "Asia/Seoul", "yyyy-MM-dd");
      }
      const str = String(value).trim();
      const m = str.match(/^(\d{4})[-.]\s*(\d{2})[-.]\s*(\d{2})/);
      return m ? `${m[1]}-${m[2]}-${m[3]}` : str;
    };
    let rowToDelete = null;
    for (let i = data.length - 1; i >= 0; i--) {
      const stock = String(data[i][0] || "").trim();
      const buyDate = toDateKey(data[i][1]);
      const sellDate = String(data[i][3] || "").trim();
      const label = normalizeLabel(data[i][5]);
      if (
        stock === target.stockName &&
        !sellDate &&
        label === normalizeLabel(strategyLabel) &&
        (!target.buyDate || buyDate === target.buyDate)
      ) {
        rowToDelete = i + 3;
        break;
      }
    }

    if (!rowToDelete) {
      for (let i = data.length - 1; i >= 0; i--) {
        const stock = String(data[i][0] || "").trim();
        const sellDate = String(data[i][3] || "").trim();
        const label = normalizeLabel(data[i][5]);
        if (
          stock === target.stockName &&
          !sellDate &&
          label === normalizeLabel(strategyLabel)
        ) {
          rowToDelete = i + 3;
          console.log(`[로그 정리 보완] ${target.stockName}: 날짜 일치 행 없음 → 마지막 열린 ${target.strategy}그룹 매수 로그로 대체`);
          break;
        }
      }
    }

    if (rowToDelete) {
      logSheet.deleteRow(rowToDelete);
      console.log(`[로그 정리] ${target.stockName}: 마지막 열린 매수 로그 1행 삭제 (${rowToDelete}행)`);
    } else {
      console.log(`[로그 정리 스킵] ${target.stockName}: 삭제할 열린 매수 로그 없음`);
    }
  });

  console.log("========== 오탐 재진입 정리 완료 ==========");
}

/**
 * 현재 Script Properties 전체 상태 확인용
 *
 * - 시스템 관련 키를 전부 출력
 * - 민감 키(API KEY)는 값 마스킹
 * - prefix별 건수 요약 + 전체 key/value 출력
 *
 * ⚠️ 직접 실행 전용
 */
function checkAllSystemKeys() {
  const allProps = PropertiesService.getScriptProperties().getProperties();
  const entries = Object.entries(allProps).sort((a, b) => a[0].localeCompare(b[0]));

  const groups = [
    "ENTRY_",
    "SLOTS_",
    "SELL_",
    "SLOT_SELL_",
    "REENTRY_COUNT_",
    "CYCLE_ENTRY_",
    "HOLD_ANCHOR_",
    "HOLD_WATCH_",
    "A_HOLD_ANCHOR_",
    "A_HOLD_WATCH_",
    "UPPER_EXIT_ARM_",
    "SLOT_UPPER_EXIT_ARM_",
    "lastValues",
    "Nasdaq",
    "UPDATE_ALL_PIPELINE_STATE"
  ];

  const summarizeGroup = prefix =>
    entries.filter(([key]) => key === prefix || key.startsWith(prefix)).length;

  console.log("========== Script Properties 전체 점검 시작 ==========");
  console.log(`[전체 키 수] ${entries.length}개`);
  groups.forEach(prefix => {
    const count = summarizeGroup(prefix);
    if (count > 0) {
      console.log(`[그룹 요약] ${prefix} → ${count}개`);
    }
  });

  if (entries.length === 0) {
    console.log("(비어 있음)");
    console.log("========== Script Properties 전체 점검 종료 ==========");
    return;
  }

  console.log("===== 전체 키 목록 =====");
  entries.forEach(([key, value]) => {
    const maskedValue = /KEY|TOKEN|SECRET/i.test(key)
      ? "[MASKED]"
      : String(value);
    console.log(`${key} = ${maskedValue}`);
  });
  console.log("========== Script Properties 전체 점검 종료 ==========");
}
