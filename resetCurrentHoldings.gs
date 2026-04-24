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
