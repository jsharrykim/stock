/**
 * earningsAlert.gs
 * 실적발표 전날 알림 메일 — 매일 밤 9시 KST 트리거
 *
 * 실적발표 D-1 종목은 기술분석 시트 AZ열의 "(D-1)" 값으로 판단
 */

function sendEarningsAlert() {
  console.log("========== 실적발표 알림 시작 ==========");

  const { targetSheet } = Utils.getSheets("기술분석");
  if (!targetSheet) return;

  const recipientEmail = String(targetSheet.getRange("F1").getValue() || "").trim();
  if (!recipientEmail) { console.log("[중단] F1 이메일 주소 없음"); return; }

  const earningsStocks = _earningsGetFromSheet(targetSheet);
  if (earningsStocks.length === 0) {
    console.log("[완료] 내일 실적발표 종목 없음 — 메일 미발송");
    return;
  }
  console.log(`[발표] ${earningsStocks.length}개 종목: ${earningsStocks.map(e => e.ticker).join(", ")}`);

  const trendData   = Utils.getLatestTrendData(targetSheet);
  const industryMap = Utils.buildIndustryMap(targetSheet);
  const globalData  = _earningsGetGlobalData(targetSheet);
  const opinionMap  = _earningsGetOpinionMap(targetSheet);

  const stocks = earningsStocks.map(e => {
    const industry   = industryMap[e.ticker] || null;
    const trendBadge = industry && trendData ? Utils.buildTrendBadge(industry, trendData) : null;
    const opinion    = opinionMap[e.ticker]  || "관망";
    return { ...e, industry, trendBadge, opinion };
  });

  const now     = new Date();
  const kstDate = Utilities.formatDate(now, "Asia/Seoul",       "yyyy.MM.dd HH:mm");
  const estDate = Utilities.formatDate(now, "America/New_York", "MM/dd HH:mm 'ET'");

  _earningsSendEmail(recipientEmail, stocks, trendData, globalData, kstDate, estDate);
  console.log("========== 실적발표 알림 종료 ==========");
}

// 기술분석 시트 AZ열에서 D-1 종목 추출
function _earningsGetFromSheet(targetSheet) {
  const lastRow = targetSheet.getLastRow();
  if (lastRow < 3) return [];

  const EARNINGS_COL = 51; // AZ열 (0-indexed)
  const data = targetSheet.getRange(3, 1, lastRow - 2, 52).getValues();
  const results = [];

  data.forEach(row => {
    const ticker      = String(row[Utils.COL_INDICES.stockName] || "").trim();
    const earningsRaw = String(row[EARNINGS_COL] || "").trim();
    if (!ticker || !earningsRaw || earningsRaw === "-") return;

    if (earningsRaw.includes("(D-1)")) {
      const date = earningsRaw.split(" ")[0]; // "2026-04-29 (D-1)" → "2026-04-29"
      results.push({ ticker, date });
    }
  });

  return results;
}

// 글로벌 데이터 (VIX, IXIC 이격도)
function _earningsGetGlobalData(targetSheet) {
  const vix       = Number(targetSheet.getRange("O1").getValue()) || 0;
  const ixicPrice = Number(targetSheet.getRange("W1").getValue()) || 0;
  const ixicMa200 = Number(targetSheet.getRange("AE1").getValue()) || 0;
  const ixicDist  = (ixicPrice > 0 && ixicMa200 > 0) ? ((ixicPrice / ixicMa200 - 1) * 100) : 0;
  return { vix, ixicDist };
}

// 투자의견 맵  { ticker → opinion }
function _earningsGetOpinionMap(targetSheet) {
  const C       = Utils.COL_INDICES;
  const lastRow = targetSheet.getLastRow();
  if (lastRow < 3) return {};
  const data = targetSheet.getRange(3, 1, lastRow - 2, C.opinion + 1).getValues();
  const map  = {};
  data.forEach(row => {
    const ticker  = String(row[C.stockName] || "").trim();
    const opinion = String(row[C.opinion]   || "").trim();
    if (ticker) map[ticker] = opinion;
  });
  return map;
}

// 우호 / 주의 요인 자동 도출
function _earningsBuildProsCons(stock, globalData) {
  const pros = [];
  const cons = [];

  if (stock.trendBadge) {
    if (stock.trendBadge.includes("[주도]") || stock.trendBadge.includes("[강세]")) {
      pros.push("업종 주도권 유지 중");
    } else {
      cons.push("업종 트렌드 비주도");
    }
  } else {
    cons.push("업종 트렌드 매칭 없음");
  }

  if (stock.opinion === "매수") {
    pros.push("현재 시스템 매수 신호 활성");
  } else if (stock.opinion === "관망") {
    cons.push("현재 매수 신호 미충족");
  } else if (stock.opinion.includes("매도")) {
    cons.push("현재 매도/쿨다운 상태");
  }

  if (globalData.vix > 25) {
    cons.push(`시장 공포 구간 (VIX ${globalData.vix.toFixed(1)})`);
  } else if (globalData.vix < 18) {
    pros.push(`시장 안정 구간 (VIX ${globalData.vix.toFixed(1)})`);
  }

  if (globalData.ixicDist > 5) {
    cons.push("나스닥 과열 — 기대치 선반영 가능성");
  } else if (globalData.ixicDist < -5) {
    pros.push("나스닥 저점 — 실적 서프라이즈 반응 강할 수 있음");
  }

  if (pros.length >= 2) {
    cons.push("기대치 과다 선반영 가능 — 가이던스가 더 중요");
  }

  return {
    pros: pros.length > 0 ? pros : ["특이 우호 요인 없음"],
    cons: cons.length > 0 ? cons : ["특이 주의 요인 없음"],
  };
}

// 발표 직후 볼 것 — 산업별 자동 도출
function _earningsBuildFocusPoints(industry) {
  if (!industry) return "매출 · EPS · 가이던스 · 컨콜 톤";
  const i = industry.toLowerCase();
  if (i.includes("반도체") || i.includes("ai")   || i.includes("gpu"))
    return "매출 · EPS · 데이터센터/AI 가이던스 · 컨콜 톤";
  if (i.includes("클라우드") || i.includes("saas") || i.includes("소프트웨어"))
    return "ARR · NRR · 영업이익률 · 차기 분기 가이던스";
  if (i.includes("플랫폼") || i.includes("모빌리티") || i.includes("리테일"))
    return "Gross Bookings/GMV · 영업이익률 · FY 가이던스";
  if (i.includes("바이오") || i.includes("헬스케어") || i.includes("제약"))
    return "임상 데이터 · 신약 승인 현황 · 파이프라인 가이던스";
  if (i.includes("금융") || i.includes("은행"))
    return "NIM · 대출 성장률 · 충당금 · 배당 가이던스";
  if (i.includes("에너지") || i.includes("원유"))
    return "생산량 · 자본지출 · 유가 헤징 · 배당";
  return "매출 · EPS · 가이던스 · 컨콜 톤";
}

// 이메일 HTML 빌드
function _earningsBuildEmailBody(stocks, trendData, globalData, kstDate, estDate) {
  const topSectors = trendData && trendData.ranks
    ? trendData.ranks.slice(0, 3).map(r => `<strong>${r.sector} (${r.rank}위)</strong>`).join(", ")
    : "트렌드 데이터 없음";

  const ixicDistStr = globalData.ixicDist >= 0
    ? `+${globalData.ixicDist.toFixed(1)}%`
    : `${globalData.ixicDist.toFixed(1)}%`;

  const contextBar =
    `<div style="margin-bottom:16px;padding:8px 12px;background:#f0f7ff;border-left:3px solid #3498db;font-size:13px;color:#333;">` +
    `이번 주 주도 섹터: ${topSectors}` +
    `&nbsp;|&nbsp; VIX <strong>${globalData.vix.toFixed(1)}</strong>` +
    `&nbsp;|&nbsp; IXIC 이격도 <strong>${ixicDistStr}</strong>` +
    `</div>`;

  const cardsHtml = stocks.map(s => {
    const isHolding   = s.opinion === "매수";
    const isSell      = s.opinion.includes("매도");
    const borderColor = isHolding ? "#2ecc71" : isSell ? "#e74c3c" : "#95a5a6";
    const badgeBg     = isHolding ? "#27ae60" : isSell ? "#c0392b" : "#7f8c8d";

    const { pros, cons } = _earningsBuildProsCons(s, globalData);
    const focusPoints    = _earningsBuildFocusPoints(s.industry);

    return (
      `<div style="margin-bottom:10px;padding:10px;background:#f9f9f9;border-left:3px solid ${borderColor};">` +
      `<strong style="font-size:15px;">${s.ticker}</strong>` +
      `&nbsp;<span style="font-size:12px;color:#fff;background:${badgeBg};padding:1px 6px;border-radius:3px;">${s.opinion}</span>` +
      `<br>` +
      `<span style="font-size:13px;">발표일: <strong>${s.date.replace(/-/g, ".")} (한국시간)</strong></span><br>` +
      (s.industry   ? `<span style="font-size:12px;color:#666;">산업: ${s.industry}</span><br>` : "") +
      (s.trendBadge ? `<span style="font-size:12px;color:#e67e22;">${s.trendBadge}</span><br>` : "") +
      `<br>` +
      `<span style="font-size:13px;color:#27ae60;">▲ 우호 요인</span>` +
      `<span style="font-size:13px;color:#444;">&nbsp;${pros.join(" · ")}</span><br>` +
      `<span style="font-size:13px;color:#c0392b;">▼ 주의 요인</span>` +
      `<span style="font-size:13px;color:#444;">&nbsp;${cons.join(" · ")}</span><br>` +
      `<br>` +
      `<span style="font-size:13px;font-weight:bold;color:#333;">발표 직후 볼 것: ${focusPoints}</span>` +
      `</div>`
    );
  }).join("");

  const footerNote =
    `<div style="padding:10px 14px;background:#fffbe6;border-left:3px solid #f39c12;font-size:13px;color:#555;">` +
    `<strong>발표 후 공통 체크</strong><br>` +
    `숫자(매출·EPS)보다 <strong>가이던스·컨콜 톤</strong>이 시초가를 결정하는 경우가 많습니다.<br>` +
    `시간외 급등락이 "실적 호조" 때문인지 "가이던스 실망" 때문인지 구분 후 다음날 시초가 판단하세요.` +
    `</div>`;

  return (
    `<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.6;max-width:600px;">` +
    `<p style="font-size:16px;font-weight:bold;color:#333;border-bottom:2px solid #eee;padding-bottom:8px;">` +
    `내일 실적발표 종목 알림` +
    `</p>` +
    contextBar +
    cardsHtml +
    `<br>` +
    footerNote +
    `<br>` +
    `<p style="color:#888;font-size:12px;margin:0;">` +
    `발송 시각 (한국): ${kstDate}<br>` +
    `발송 시각 (미 동부): ${estDate}` +
    `</p>` +
    `</div>`
  );
}

// 이메일 발송
function _earningsSendEmail(recipientEmail, stocks, trendData, globalData, kstDate, estDate) {
  const tickerList = stocks.map(s => s.ticker).join(", ");
  const subject    = `[실적발표] ${tickerList} — 내일 발표`;
  const htmlBody   = _earningsBuildEmailBody(stocks, trendData, globalData, kstDate, estDate);

  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      GmailApp.sendEmail(recipientEmail, subject, "", { htmlBody });
      console.log(`[이메일 발송] ${subject} → ${recipientEmail}`);
      return;
    } catch (e) {
      console.log(`[이메일 실패] 시도 ${attempt}/${maxAttempts}: ${e}`);
      if (attempt < maxAttempts) Utilities.sleep(1500 * attempt);
    }
  }
  console.log("[이메일 FATAL] 재시도 후에도 발송 실패");
}
