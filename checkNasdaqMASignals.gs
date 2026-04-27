function checkNasdaqMASignals() {
  var RECIPIENT_EMAIL_CELL = "F1";
  var NASDAQ_CURRENT_PRICE_CELL_REF = "W1";
  var VIX_TODAY_CELL = "O1";
  var VIX_YESTERDAY_CELL = "Q1";
  var NASDAQ_MA200_CELL = "AE1";

  var PEAK_CONFIG = {
    multiplier: 1.12,
    vixThreshold: 18,
    vixSurgePercent: 10,
    stateKey: "NasdaqPeakSellState",
    dailyReachedKey: "NasdaqPeakReachedToday",
    lastResetDateKey: "NasdaqPeakLastResetDate"
  };

  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var targetSheet = ss.getSheetByName("기술분석");
  if (!targetSheet) return;

  var properties = PropertiesService.getScriptProperties();
  var now = new Date();

  var kstDate = Utilities.formatDate(now, "Asia/Seoul", "yyyy. MM. dd, HH:mm:ss");
  var kstDateOnly = Utilities.formatDate(now, "Asia/Seoul", "yyyy-MM-dd");
  var estString = Utilities.formatDate(now, "America/New_York", "M/d/yyyy, h:mm:ss a");

  var recipientEmail = targetSheet.getRange(RECIPIENT_EMAIL_CELL).getValue();
  if (!recipientEmail || recipientEmail.length === 0) return;

  try {
    var currentPrice = Number(targetSheet.getRange(NASDAQ_CURRENT_PRICE_CELL_REF).getValue());
    var vixToday = Number(targetSheet.getRange(VIX_TODAY_CELL).getValue());
    var vixYesterday = Number(targetSheet.getRange(VIX_YESTERDAY_CELL).getValue());
    var nasdaqMA200 = Number(targetSheet.getRange(NASDAQ_MA200_CELL).getValue());

    if (!currentPrice || !nasdaqMA200) {
      Logger.log("[고점 경고] 데이터 오류 — 현재가: " + currentPrice + ", MA200: " + nasdaqMA200);
      return;
    }

    var multiplierLabel = "×" + PEAK_CONFIG.multiplier.toFixed(2);

    var lastResetDate = properties.getProperty(PEAK_CONFIG.lastResetDateKey);
    if (lastResetDate !== kstDateOnly) {
      properties.setProperty(PEAK_CONFIG.dailyReachedKey, "FALSE");
      properties.setProperty(PEAK_CONFIG.lastResetDateKey, kstDateOnly);
      Logger.log("[고점 경고] 날짜 변경 (" + lastResetDate + " → " + kstDateOnly + "). 당일 돌파 플래그 초기화.");
    }

    var nasdaqThreshold = nasdaqMA200 * PEAK_CONFIG.multiplier;
    var nasdaqPremiumPercent = ((currentPrice / nasdaqMA200 - 1) * 100).toFixed(2);

    var vixChangePercent = vixYesterday !== 0
      ? ((vixToday / vixYesterday - 1) * 100).toFixed(2)
      : 0;
    var isVixSurge = Number(vixChangePercent) >= PEAK_CONFIG.vixSurgePercent;
    var isVixHigh = vixToday > PEAK_CONFIG.vixThreshold;
    var isVixConditionMet = isVixHigh || isVixSurge;

    var isNasdaqHigh = currentPrice > nasdaqThreshold;
    var lastPeakState = properties.getProperty(PEAK_CONFIG.stateKey) === "TRUE";

    var peakReachedToday = properties.getProperty(PEAK_CONFIG.dailyReachedKey) === "TRUE";
    if (isNasdaqHigh && !peakReachedToday) {
      properties.setProperty(PEAK_CONFIG.dailyReachedKey, "TRUE");
      Logger.log("[고점 경고] 당일 최초 " + multiplierLabel + " 돌파 감지. 플래그 기록. (현재가: " + currentPrice.toFixed(2) + " > 기준선: " + nasdaqThreshold.toFixed(2) + ")");
    }

    var isPeakReachedTodayNow = properties.getProperty(PEAK_CONFIG.dailyReachedKey) === "TRUE";
    var isPeakTriggered = isPeakReachedTodayNow && isVixConditionMet;

    Logger.log("[고점 경고] 나스닥 현재가: " + currentPrice.toFixed(2));
    Logger.log("[고점 경고] MA200: " + nasdaqMA200.toFixed(2) + ", 기준선 (" + multiplierLabel + "): " + nasdaqThreshold.toFixed(2));
    Logger.log("[고점 경고] MA200 대비: +" + nasdaqPremiumPercent + "%");
    Logger.log("[고점 경고] 현재 " + multiplierLabel + " 초과: " + (isNasdaqHigh ? "YES" : "NO"));
    Logger.log("[고점 경고] 당일 1회 이상 돌파: " + (isPeakReachedTodayNow ? "YES" : "NO"));
    Logger.log("[고점 경고] VIX 현재: " + vixToday.toFixed(2) + ", 전일: " + vixYesterday.toFixed(2) + ", 변화: " + (Number(vixChangePercent) >= 0 ? "+" : "") + vixChangePercent + "%");
    Logger.log("[고점 경고] VIX 조건 충족: " + (isVixConditionMet ? "YES" : "NO"));
    Logger.log("[고점 경고] 최종 시그널: " + (isPeakTriggered ? "TRIGGERED" : "NOT TRIGGERED"));
    Logger.log("[고점 경고] 이전 알림 상태: " + (lastPeakState ? "SENT" : "NOT SENT"));

    if (isPeakTriggered && !lastPeakState) {
      var vixDescription = "";
      if (isVixHigh && isVixSurge) {
        vixDescription = "VIX가 " + vixToday.toFixed(2) + "로 " + PEAK_CONFIG.vixThreshold + "을 초과했으며, 전일 대비 +" + vixChangePercent + "% 급등했습니다.";
      } else if (isVixHigh) {
        vixDescription = "VIX가 " + vixToday.toFixed(2) + "로 " + PEAK_CONFIG.vixThreshold + "을 초과했습니다.";
      } else if (isVixSurge) {
        vixDescription = "VIX가 전일 대비 +" + vixChangePercent + "% 급등했습니다 (" + vixYesterday.toFixed(2) + " → " + vixToday.toFixed(2) + ").";
      }

      var emailSubject = "나스닥 고점 구간 알림 (매도 시그널)";
      var emailBody =
        '<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.6;color:#222;padding-top:8px;">' +
          '<p style="font-size:16px;font-weight:bold;color:#333;margin:0 0 12px 0;">' +
            "나스닥 종합지수가 고점 구간에 진입했으며, 시장 불안이 감지되었습니다." +
          "</p>" +
          '<div style="margin:0 0 14px 0;">' +
            '<div style="margin:4px 0;"><strong>나스닥 현재가:</strong> ' + currentPrice.toFixed(2) + "</div>" +
            '<div style="margin:4px 0;"><strong>나스닥 200일 이평선:</strong> ' + nasdaqMA200.toFixed(2) + "</div>" +
            '<div style="margin:4px 0;"><strong>200일선 대비:</strong> +' + nasdaqPremiumPercent + "%</div>" +
            '<div style="margin:4px 0;"><strong>기준선 (MA200 ' + multiplierLabel + "):</strong> " + nasdaqThreshold.toFixed(2) + "</div>" +
          "</div>" +
          '<div style="margin:0 0 14px 0;">' +
            '<div style="margin:4px 0;"><strong>VIX 현재:</strong> ' + vixToday.toFixed(2) + "</div>" +
            '<div style="margin:4px 0;"><strong>VIX 전일:</strong> ' + vixYesterday.toFixed(2) + "</div>" +
            '<div style="margin:4px 0;"><strong>VIX 변화:</strong> ' + (Number(vixChangePercent) >= 0 ? "+" : "") + vixChangePercent + "%</div>" +
          "</div>" +
          "<p style=\"margin:0 0 12px 0;\">" + vixDescription + "</p>" +
          '<p style="margin:0 0 12px 0;">' +
            "나스닥 종합지수가 금일 중 200일 이동평균선 대비 " + multiplierLabel + "를 돌파했으며," +
            " 변동성 지수(VIX)가 상승하여 시장 불안감이 커지고 있습니다." +
            " 역사적으로 이러한 조건에서 대규모 조정이 발생했습니다." +
            " 따라서 보유 중인 종목의 부분 매도 또는 일괄 매도를 신중히 고려하시기 바랍니다." +
          "</p>" +
          '<p style="margin:0 0 12px 0;">' +
            "단, 시장 전체의 방향성과 별개로 산업군·종목에 따라 개별 상승 모멘텀이 유효한 경우도 있으며," +
            " 금리·인플레이션 등 거시 지표나 주요 기업 실적발표 일정에 따라 국면이 달라질 수 있습니다." +
            " 매수 조건을 충족한 종목은 이후에도 시그널 메일이 발송될 수 있으나," +
            " 가급적 당분간은 신규 진입을 자제하고 개별 종목 단위의 진입 여부는 시황을 직접 확인한 후 스스로 판단하시기 바랍니다." +
          "</p>" +
          '<p style="margin:0 0 12px 0;">' +
            "※ 알림은 조건 충족 시 <strong>한 번만</strong> 발송되며, 나스닥이 기준선 아래로 하락 시 재알림이 가능합니다." +
          "</p>" +
          '<p style="margin:0;">' +
            "발송 시각 (한국 날짜): " + kstDate + "<br>" +
            "발송 시각 (미 동부 시간): " + estString +
          "</p>" +
        "</div>";

      try {
        GmailApp.sendEmail(recipientEmail, emailSubject, "", { htmlBody: emailBody });
        properties.setProperty(PEAK_CONFIG.stateKey, "TRUE");
        Logger.log("[고점 경고 SUCCESS] 알림 발송 완료. 상태 TRUE 저장. (현재가: " + currentPrice.toFixed(2) + ")");
      } catch (e) {
        Logger.log("[고점 경고 FATAL] 이메일 발송 실패: " + e.toString());
      }
    } else {
      Logger.log("[고점 경고] 시그널 미감지 또는 이미 발송됨 — 스킵");
    }

    if (currentPrice <= nasdaqThreshold && lastPeakState) {
      properties.setProperty(PEAK_CONFIG.stateKey, "FALSE");
      Logger.log("[고점 경고] 나스닥이 기준선(" + nasdaqThreshold.toFixed(2) + ") 아래로 하락. 상태 FALSE 초기화 (재알림 가능).");
    }
  } catch (e) {
    Logger.log("[고점 경고 FATAL] 처리 중 오류: " + e.toString());
  }
}
