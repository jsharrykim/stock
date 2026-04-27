function SHEETID() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName("기술분석");
  var range = sheet.getRange("H1");
  var existingValue = range.getValue();

  if (existingValue && existingValue.length > 0) {
    return existingValue;
  }

  var spreadsheetId = spreadsheet.getId();
  range.setValue(spreadsheetId);
  return spreadsheetId;
}

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("시트 도구")
    .addItem("시트 ID 가져오기", "SHEETID")
    .addToUi();
}
