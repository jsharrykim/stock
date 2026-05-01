import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'

type Market = 'KR' | 'US'
type Valuation = '저평가' | '보통' | '고평가'
type Opinion = '매수' | '관망' | '매도'
type TradeStatus = '익절' | '손절' | '실패 익절' | '보유 중'

type Stock = {
  ticker: string
  name: string
  market: Market
  fairPrice: string
  currentPrice: string
  valuation: Valuation
  opinion: Opinion
  strategies: string[]
  updatedAt: string
}

type TradeLog = {
  ticker: string
  strategy: string
  buyDate: string
  buyPrice: string
  sellDate: string
  sellPrice: string
  returnPct: number
  holdingDays: number | '-'
  status: TradeStatus
}

type TooltipState = {
  text: string
  x: number
  y: number
}

type ValuationMetric = {
  marketCap: string
  sales: string
  salesQoq: string
  salesYoyTtm: string
  salesPastYears: string
  currentRatio: string
  priceToFreeCashFlow: string
  priceToSales: string
  per: string
  pbr: string
  roe: string
  peg: string
  sharesOutstanding: string
  grossMargin: string
  operatingMargin: string
  epsTtm: string
  epsNextYear: string
  epsQoq: string
  ruleOf40: string
  earningsDate: string
}

const MAX_WATCHLIST_ITEMS = 50

const demoLimitStocks: Stock[] = Array.from({ length: 38 }, (_, index) => {
  const sequence = index + 1

  return {
    ticker: `DEMO${String(sequence).padStart(2, '0')}`,
    name: `데모종목 ${sequence}`,
    market: sequence % 3 === 0 ? 'KR' : 'US',
    fairPrice: sequence % 3 === 0 ? `₩${(42_000 + sequence * 1_150).toLocaleString()}` : `$${(36 + sequence * 1.7).toFixed(2)}`,
    currentPrice: sequence % 3 === 0 ? `₩${(39_500 + sequence * 1_030).toLocaleString()}` : `$${(34 + sequence * 1.45).toFixed(2)}`,
    valuation: sequence % 4 === 0 ? '고평가' : sequence % 2 === 0 ? '보통' : '저평가',
    opinion: sequence % 4 === 0 ? '매도' : sequence % 2 === 0 ? '관망' : '매수',
    strategies: ['D. 200일선 상방 & 상승 흐름 강화'],
    updatedAt: '데모',
  }
})

const searchUniverse: Stock[] = [
  {
    ticker: '005930',
    name: '삼성전자',
    market: 'KR',
    fairPrice: '₩82,000',
    currentPrice: '₩84,200',
    valuation: '보통',
    opinion: '관망',
    strategies: ['D. 200일선 상방 & 상승 흐름 강화'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'NVDA',
    name: 'NVIDIA',
    market: 'US',
    fairPrice: '$98.00',
    currentPrice: '$109.88',
    valuation: '고평가',
    opinion: '관망',
    strategies: ['A. 200일선 상방 & 모멘텀 재가속'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'AAPL',
    name: 'Apple',
    market: 'US',
    fairPrice: '$203.00',
    currentPrice: '$195.42',
    valuation: '보통',
    opinion: '매수',
    strategies: ['C. 200일선 상방 & 스퀴즈 거래량 돌파', 'D. 200일선 상방 & 상승 흐름 강화'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'TSLA',
    name: 'Tesla',
    market: 'US',
    fairPrice: '$230.00',
    currentPrice: '$265.30',
    valuation: '고평가',
    opinion: '매도',
    strategies: ['F. 200일선 상방 & BB 극단 저점'],
    updatedAt: '2시간 전',
  },
  {
    ticker: '035420',
    name: 'NAVER',
    market: 'KR',
    fairPrice: '₩245,000',
    currentPrice: '₩209,500',
    valuation: '저평가',
    opinion: '매수',
    strategies: ['E. 200일선 상방 & 스퀴즈 저점'],
    updatedAt: '2시간 전',
  },
  {
    ticker: '042700',
    name: '한미반도체',
    market: 'KR',
    fairPrice: '₩178,000',
    currentPrice: '₩169,400',
    valuation: '보통',
    opinion: '관망',
    strategies: ['D. 200일선 상방 & 상승 흐름 강화'],
    updatedAt: '2시간 전',
  },
  {
    ticker: '247540',
    name: '에코프로비엠',
    market: 'KR',
    fairPrice: '₩132,000',
    currentPrice: '₩151,800',
    valuation: '고평가',
    opinion: '매도',
    strategies: ['E. 200일선 상방 & 스퀴즈 저점'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'ONON',
    name: 'On Holding',
    market: 'US',
    fairPrice: '$46.00',
    currentPrice: '$38.30',
    valuation: '보통',
    opinion: '매수',
    strategies: ['B. 200일선 하방 & 공황 저점'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'BE',
    name: 'Bloom Energy',
    market: 'US',
    fairPrice: '$27.00',
    currentPrice: '$20.70',
    valuation: '저평가',
    opinion: '관망',
    strategies: ['F. 200일선 상방 & BB 극단 저점'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'LRCX',
    name: 'Lam Research',
    market: 'US',
    fairPrice: '$104.00',
    currentPrice: '$95.20',
    valuation: '보통',
    opinion: '매수',
    strategies: ['D. 200일선 상방 & 상승 흐름 강화'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'SNDK',
    name: 'Sandisk',
    market: 'US',
    fairPrice: '$68.00',
    currentPrice: '$57.40',
    valuation: '저평가',
    opinion: '관망',
    strategies: ['A. 200일선 상방 & 모멘텀 재가속'],
    updatedAt: '2시간 전',
  },
  {
    ticker: 'MSFT',
    name: 'Microsoft',
    market: 'US',
    fairPrice: '$520.00',
    currentPrice: '$485.90',
    valuation: '보통',
    opinion: '매수',
    strategies: ['C. 200일선 상방 & 스퀴즈 거래량 돌파'],
    updatedAt: '2시간 전',
  },
  ...demoLimitStocks,
]

const initialWatchlist = [
  'AAPL',
  '035420',
  'NVDA',
  '005930',
  '042700',
  '247540',
  'ONON',
  'BE',
  'LRCX',
  'SNDK',
  'TSLA',
  'MSFT',
  ...demoLimitStocks.map((stock) => stock.ticker),
]

const trades: TradeLog[] = [
  {
    ticker: 'AAPL',
    strategy: 'C. 200일선 상방 & 스퀴즈 거래량 돌파',
    buyDate: '2026.03.18',
    buyPrice: '$195.42',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 3.4,
    holdingDays: 44,
    status: '보유 중',
  },
  {
    ticker: 'NVDA',
    strategy: 'A. 200일선 상방 & 모멘텀 재가속',
    buyDate: '2026.02.04',
    buyPrice: '$109.88',
    sellDate: '2026.03.26',
    sellPrice: '$121.20',
    returnPct: 10.3,
    holdingDays: 50,
    status: '익절',
  },
  {
    ticker: 'TSLA',
    strategy: 'F. 200일선 상방 & BB 극단 저점',
    buyDate: '2026.01.12',
    buyPrice: '$265.30',
    sellDate: '2026.02.19',
    sellPrice: '$248.90',
    returnPct: -6.2,
    holdingDays: 38,
    status: '손절',
  },
  {
    ticker: '035420',
    strategy: 'E. 200일선 상방 & 스퀴즈 저점',
    buyDate: '2026.04.12',
    buyPrice: '₩209,500',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 1.9,
    holdingDays: 19,
    status: '보유 중',
  },
  {
    ticker: '005930',
    strategy: 'D. 200일선 상방 & 상승 흐름 강화',
    buyDate: '2026.04.05',
    buyPrice: '₩84,200',
    sellDate: '2026.04.18',
    sellPrice: '₩88,500',
    returnPct: 5.1,
    holdingDays: 13,
    status: '익절',
  },
  {
    ticker: 'ONON',
    strategy: 'B. 200일선 하방 & 공황 저점',
    buyDate: '2026.03.02',
    buyPrice: '$38.30',
    sellDate: '2026.03.20',
    sellPrice: '$44.55',
    returnPct: 16.3,
    holdingDays: 18,
    status: '익절',
  },
  {
    ticker: 'BE',
    strategy: 'F. 200일선 상방 & BB 극단 저점',
    buyDate: '2026.02.21',
    buyPrice: '$20.70',
    sellDate: '2026.03.04',
    sellPrice: '$19.30',
    returnPct: -6.8,
    holdingDays: 11,
    status: '손절',
  },
  {
    ticker: 'LRCX',
    strategy: 'D. 200일선 상방 & 상승 흐름 강화',
    buyDate: '2026.01.28',
    buyPrice: '$95.20',
    sellDate: '2026.02.27',
    sellPrice: '$105.41',
    returnPct: 10.7,
    holdingDays: 30,
    status: '실패 익절',
  },
  {
    ticker: '042700',
    strategy: 'D. 200일선 상방 & 상승 흐름 강화',
    buyDate: '2026.04.24',
    buyPrice: '₩169,400',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: '247540',
    strategy: 'E. 200일선 상방 & 스퀴즈 저점',
    buyDate: '2026.04.22',
    buyPrice: '₩151,800',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'ONON',
    strategy: 'B. 200일선 하방 & 공황 저점',
    buyDate: '2026.04.19',
    buyPrice: '$38.30',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'BE',
    strategy: 'F. 200일선 상방 & BB 극단 저점',
    buyDate: '2026.04.18',
    buyPrice: '$20.70',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'LRCX',
    strategy: 'D. 200일선 상방 & 상승 흐름 강화',
    buyDate: '2026.04.16',
    buyPrice: '$95.20',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'SNDK',
    strategy: 'A. 200일선 상방 & 모멘텀 재가속',
    buyDate: '2026.04.15',
    buyPrice: '$57.40',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'TSLA',
    strategy: 'F. 200일선 상방 & BB 극단 저점',
    buyDate: '2026.04.13',
    buyPrice: '$265.30',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: '005930',
    strategy: 'D. 200일선 상방 & 상승 흐름 강화',
    buyDate: '2026.04.11',
    buyPrice: '₩84,200',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'MSFT',
    strategy: 'C. 200일선 상방 & 스퀴즈 거래량 돌파',
    buyDate: '2026.04.09',
    buyPrice: '$485.90',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
  {
    ticker: 'NVDA',
    strategy: 'A. 200일선 상방 & 모멘텀 재가속',
    buyDate: '2026.04.07',
    buyPrice: '$109.88',
    sellDate: '보유 중',
    sellPrice: '-',
    returnPct: 0,
    holdingDays: '-',
    status: '보유 중',
  },
]

const operatorTickers = ['005930', '042700', '247540', 'ONON', 'BE', 'LRCX', 'SNDK', 'AAPL', '035420', 'NVDA', 'TSLA', 'MSFT']
const strategyFilters = ['A', 'B', 'C', 'D', 'E', 'F']
const personalTradeTickers = new Set(['AAPL', 'NVDA', '035420'])
const personalTrades = trades.filter((trade) => personalTradeTickers.has(trade.ticker))
const operatorTrades = trades
const valuationMetrics: Record<string, ValuationMetric> = {
  '005930': {
    marketCap: '106조 7,264억',
    sales: '186조 2,545억',
    salesQoq: '+3.2%',
    salesYoyTtm: '+11.8%',
    salesPastYears: '+4.9% / +6.2%',
    currentRatio: '2.61',
    priceToFreeCashFlow: '18.4',
    priceToSales: '1.02',
    per: '15.03',
    pbr: '1.21',
    roe: '8.41%',
    peg: '1.8',
    sharesOutstanding: '59억 6,978만',
    grossMargin: '38.7%',
    operatingMargin: '14.2%',
    epsTtm: '₩3,240',
    epsNextYear: '₩4,110',
    epsQoq: '+18.2%',
    ruleOf40: '26.0%',
    earningsDate: '2026.07.30',
  },
  NVDA: {
    marketCap: '1,292조 1,040억',
    sales: '333조 6,059억',
    salesQoq: '+12.1%',
    salesYoyTtm: '+68.4%',
    salesPastYears: '+37.4% / +52.8%',
    currentRatio: '4.77',
    priceToFreeCashFlow: '59.2',
    priceToSales: '30.46',
    per: '38.57',
    pbr: '20.18',
    roe: '52.3%',
    peg: '1.2',
    sharesOutstanding: '244억',
    grossMargin: '74.6%',
    operatingMargin: '61.9%',
    epsTtm: '$2.84',
    epsNextYear: '$4.12',
    epsQoq: '+64.1%',
    ruleOf40: '130.3%',
    earningsDate: '2026.05.27',
  },
  AAPL: {
    marketCap: '973조 1,467억',
    sales: '341억',
    salesQoq: '+0.4%',
    salesYoyTtm: '+2.1%',
    salesPastYears: '+1.1% / +5.4%',
    currentRatio: '0.87',
    priceToFreeCashFlow: '28.6',
    priceToSales: '9.43',
    per: '21.81',
    pbr: '7.37',
    roe: '44.0%',
    peg: '2.7',
    sharesOutstanding: '151억',
    grossMargin: '46.2%',
    operatingMargin: '31.5%',
    epsTtm: '$6.43',
    epsNextYear: '$7.12',
    epsQoq: '+3.7%',
    ruleOf40: '33.6%',
    earningsDate: '2026.07.23',
  },
  TSLA: {
    marketCap: '108조 7,264억',
    sales: '186조 2,545억',
    salesQoq: '-8.7%',
    salesYoyTtm: '-3.4%',
    salesPastYears: '+16.2% / +24.1%',
    currentRatio: '2.03',
    priceToFreeCashFlow: '96.3',
    priceToSales: '8.29',
    per: '148.2',
    pbr: '10.46',
    roe: '5.0%',
    peg: '5.8',
    sharesOutstanding: '32억 1,000만',
    grossMargin: '17.8%',
    operatingMargin: '6.3%',
    epsTtm: '$1.79',
    epsNextYear: '$2.18',
    epsQoq: '-20.4%',
    ruleOf40: '2.9%',
    earningsDate: '2026.07.16',
  },
  '035420': {
    marketCap: '27조 9,575억',
    sales: '97조 4,293억',
    salesQoq: '+4.8%',
    salesYoyTtm: '+10.9%',
    salesPastYears: '+9.1% / +13.3%',
    currentRatio: '1.55',
    priceToFreeCashFlow: '15.8',
    priceToSales: '0.29',
    per: '3.27',
    pbr: '0.58',
    roe: '19.0%',
    peg: '0.6',
    sharesOutstanding: '1억 6,400만',
    grossMargin: '39.4%',
    operatingMargin: '15.1%',
    epsTtm: '₩64,120',
    epsNextYear: '₩69,800',
    epsQoq: '+9.8%',
    ruleOf40: '26.0%',
    earningsDate: '2026.08.06',
  },
  '042700': {
    marketCap: '35조 7,499억',
    sales: '5,766억',
    salesQoq: '+16.7%',
    salesYoyTtm: '+44.6%',
    salesPastYears: '+28.2% / +35.1%',
    currentRatio: '3.21',
    priceToFreeCashFlow: '42.6',
    priceToSales: '60.83',
    per: '164.8',
    pbr: '50.56',
    roe: '34.8%',
    peg: '3.9',
    sharesOutstanding: '9,771만',
    grossMargin: '57.1%',
    operatingMargin: '36.8%',
    epsTtm: '₩1,028',
    epsNextYear: '₩1,790',
    epsQoq: '+52.6%',
    ruleOf40: '81.4%',
    earningsDate: '2026.08.12',
  },
  '247540': {
    marketCap: '20조 1,531억',
    sales: '2조 5,316억',
    salesQoq: '-4.9%',
    salesYoyTtm: '-21.6%',
    salesPastYears: '+18.4% / +41.7%',
    currentRatio: '1.12',
    priceToFreeCashFlow: '-',
    priceToSales: '7.96',
    per: '511.17',
    pbr: '11.65',
    roe: '2.29%',
    peg: '-',
    sharesOutstanding: '9,782만',
    grossMargin: '12.8%',
    operatingMargin: '-1.6%',
    epsTtm: '₩297',
    epsNextYear: '₩1,240',
    epsQoq: '-36.5%',
    ruleOf40: '-23.2%',
    earningsDate: '2026.08.07',
  },
  ONON: {
    marketCap: '62조 1,452억',
    sales: '11조 3,143억',
    salesQoq: '+28.9%',
    salesYoyTtm: '+32.2%',
    salesPastYears: '+43.6% / +49.8%',
    currentRatio: '2.77',
    priceToFreeCashFlow: '48.5',
    priceToSales: '5.49',
    per: '91.44',
    pbr: '6.59',
    roe: '7.70%',
    peg: '2.1',
    sharesOutstanding: '6억 3,000만',
    grossMargin: '59.8%',
    operatingMargin: '12.6%',
    epsTtm: '$0.42',
    epsNextYear: '$0.75',
    epsQoq: '+41.0%',
    ruleOf40: '44.8%',
    earningsDate: '2026.05.12',
  },
  BE: {
    marketCap: '5조 7,499억',
    sales: '1조 2,202억',
    salesQoq: '-3.6%',
    salesYoyTtm: '+8.1%',
    salesPastYears: '+17.7% / +19.4%',
    currentRatio: '2.01',
    priceToFreeCashFlow: '-',
    priceToSales: '0.68',
    per: '37.21',
    pbr: '50.56',
    roe: '4.67%',
    peg: '1.4',
    sharesOutstanding: '2억 2,800만',
    grossMargin: '23.8%',
    operatingMargin: '-4.7%',
    epsTtm: '$0.56',
    epsNextYear: '$0.92',
    epsQoq: '+12.3%',
    ruleOf40: '3.4%',
    earningsDate: '2026.05.08',
  },
  LRCX: {
    marketCap: '53조 5,742억',
    sales: '13조 6,549억',
    salesQoq: '+7.5%',
    salesYoyTtm: '+18.0%',
    salesPastYears: '+8.8% / +15.7%',
    currentRatio: '2.44',
    priceToFreeCashFlow: '24.2',
    priceToSales: '4.81',
    per: '81.71',
    pbr: '14.37',
    roe: '19.20%',
    peg: '1.9',
    sharesOutstanding: '12억 8,000만',
    grossMargin: '47.5%',
    operatingMargin: '29.6%',
    epsTtm: '$1.16',
    epsNextYear: '$1.42',
    epsQoq: '+20.6%',
    ruleOf40: '47.6%',
    earningsDate: '2026.07.29',
  },
  SNDK: {
    marketCap: '15조 8,925억',
    sales: '15조 2,726억',
    salesQoq: '+5.2%',
    salesYoyTtm: '+14.6%',
    salesPastYears: '+2.3% / +7.2%',
    currentRatio: '1.62',
    priceToFreeCashFlow: '31.4',
    priceToSales: '1.04',
    per: '55.1',
    pbr: '35.64',
    roe: '75.30%',
    peg: '2.4',
    sharesOutstanding: '3억 4,200만',
    grossMargin: '38.1%',
    operatingMargin: '18.3%',
    epsTtm: '$1.04',
    epsNextYear: '$1.68',
    epsQoq: '+39.1%',
    ruleOf40: '32.9%',
    earningsDate: '2026.08.14',
  },
  MSFT: {
    marketCap: '3,840조 1,000억',
    sales: '328조 8,000억',
    salesQoq: '+6.4%',
    salesYoyTtm: '+15.2%',
    salesPastYears: '+14.0% / +16.8%',
    currentRatio: '1.35',
    priceToFreeCashFlow: '45.2',
    priceToSales: '11.68',
    per: '36.4',
    pbr: '11.90',
    roe: '33.1%',
    peg: '2.3',
    sharesOutstanding: '74억 3,000만',
    grossMargin: '69.8%',
    operatingMargin: '44.7%',
    epsTtm: '$13.32',
    epsNextYear: '$15.10',
    epsQoq: '+11.2%',
    ruleOf40: '59.9%',
    earningsDate: '2026.07.28',
  },
}

function normalizeQuery(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, '')
}

function statusClass(value: Valuation | Opinion | TradeStatus) {
  if (value === '저평가' || value === '매수' || value === '익절') return 'positive'
  if (value === '고평가' || value === '매도' || value === '손절' || value === '실패 익절') return 'negative'
  return 'neutral'
}

function returnClass(value: number) {
  if (value > 0) return 'return-positive'
  if (value < 0) return 'return-negative'
  return ''
}

function strategyCode(strategy: string) {
  return strategy.slice(0, 1)
}

function strategyInfo(strategy: string) {
  const descriptions: Record<string, string> = {
    A: '강세 구조에서 눌림 이후 모멘텀이 다시 붙는 구간입니다. 추세가 살아 있을 때 재가속 신호를 우선 봅니다.',
    B: '200일선 아래에서 과매도와 공포가 겹친 구간입니다. 반등 가능성은 보지만 실패 시 손절 기준이 중요합니다.',
    C: '변동성이 압축된 뒤 거래량과 함께 방향성이 나오는 구간입니다. 돌파 이후 추세 지속 여부를 확인합니다.',
    D: '200일선 위에서 상승 흐름이 더 강해지는 구간입니다. 이미 강한 종목의 추세 추종 성격이 큽니다.',
    E: '상승 구조는 유지되지만 가격과 변동성이 눌린 구간입니다. 추세 안의 저점 재진입 후보로 봅니다.',
    F: '강세 구조 안에서 볼린저밴드 하단까지 과하게 밀린 구간입니다. 극단 저점 반등을 노리지만 변동성이 큽니다.',
  }
  return descriptions[strategyCode(strategy)] ?? '전략 요약 정보가 준비 중입니다. 세부 수식보다 신호의 성격만 제공합니다.'
}

function tradeResultLabel(status: TradeStatus) {
  if (status === '익절') return '성공(익절)'
  if (status === '손절') return '실패(손절)'
  if (status === '실패 익절') return '실패(익절)'
  return '보유중'
}

function tradeCriteriaInfo(strategy: string) {
  const code = strategyCode(strategy)

  if (['A', 'B', 'C'].includes(code)) {
    return `${code} 전략 기준: 성공은 매수가 대비 +20% 도달 시 즉시 익절입니다. -30%에 닿으면 손절 실패이고, 60거래일 경과 후 수익 중이거나 120거래일 최대 보유 기간에 걸려 청산되면 수익이어도 목표 미달 실패(익절)로 볼 수 있습니다.`
  }

  if (code === 'D') {
    return 'D 전략 기준: 성공은 매수가 대비 +12% 도달 시 즉시 익절입니다. -25%에 닿으면 손절 실패이고, 30거래일 최대 보유 기간 안에 목표를 채우지 못해 청산되면 수익이어도 목표 미달 실패(익절)로 볼 수 있습니다.'
  }

  if (['E', 'F'].includes(code)) {
    return `${code} 전략 기준: 성공은 +20% 도달 후 MACD 둔화 신호가 나오거나 목표 도달 후 5거래일 대기 만료 시 청산입니다. -30%에 닿으면 손절 실패이고, 60거래일 수익 중 청산이나 120거래일 최대 보유 기간 청산은 조건 충족 여부에 따라 수익이어도 실패(익절)로 볼 수 있습니다.`
  }

  return '전략별 성공/실패 기준 정보가 준비 중입니다.'
}

function tradeResultInfo(trade: TradeLog) {
  if (trade.status !== '보유 중') return tradeCriteriaInfo(trade.strategy)
  return '아직 매도 신호가 없어 성공/실패를 확정하지 않은 보유 중 거래입니다.'
}

function isSystemHolding(ticker: string, targetTrades: TradeLog[]) {
  return targetTrades.some((trade) => trade.ticker === ticker && trade.status === '보유 중')
}

function stockName(ticker: string) {
  return searchUniverse.find((stock) => stock.ticker === ticker)?.name ?? ticker
}

function stockMarket(ticker: string) {
  return searchUniverse.find((stock) => stock.ticker === ticker)?.market ?? 'US'
}

function marketFlag(market: Market) {
  return market === 'KR' ? '🇰🇷' : '🇺🇸'
}

function StrategyTag({
  strategy,
  onTooltipOpen,
  onTooltipClose,
}: {
  strategy: string
  onTooltipOpen: (tooltip: TooltipState) => void
  onTooltipClose: () => void
}) {
  const openTooltip = (element: HTMLElement) => {
    const rect = element.getBoundingClientRect()
    const minX = 280
    const maxX = window.innerWidth - 280
    const centeredX = rect.left + rect.width / 2

    onTooltipOpen({
      text: strategyInfo(strategy),
      x: Math.min(Math.max(centeredX, minX), maxX),
      y: rect.top - 8,
    })
  }

  return (
    <span
      className="strategy-item"
      onBlur={onTooltipClose}
      onFocus={(event) => openTooltip(event.currentTarget)}
      onMouseEnter={(event) => openTooltip(event.currentTarget)}
      onMouseLeave={onTooltipClose}
      tabIndex={0}
    >
      <span className={`strategy-pill strategy-${strategyCode(strategy).toLowerCase()}`}>
        {strategy}
      </span>
    </span>
  )
}

function ResultBadge({
  trade,
  onTooltipOpen,
  onTooltipClose,
}: {
  trade: TradeLog
  onTooltipOpen: (tooltip: TooltipState) => void
  onTooltipClose: () => void
}) {
  const openTooltip = (element: HTMLElement) => {
    const rect = element.getBoundingClientRect()
    const minX = 280
    const maxX = window.innerWidth - 280
    const centeredX = rect.left + rect.width / 2

    onTooltipOpen({
      text: tradeResultInfo(trade),
      x: Math.min(Math.max(centeredX, minX), maxX),
      y: rect.top - 8,
    })
  }

  return (
    <span
      className={`status-badge result-badge ${statusClass(trade.status)}`}
      onBlur={onTooltipClose}
      onFocus={(event) => openTooltip(event.currentTarget)}
      onMouseEnter={(event) => openTooltip(event.currentTarget)}
      onMouseLeave={onTooltipClose}
      tabIndex={0}
    >
      {tradeResultLabel(trade.status)}
    </span>
  )
}

function formatWinRate(label: string, targetTrades: TradeLog[]) {
  const finished = targetTrades.filter((trade) => trade.status !== '보유 중')
  if (finished.length === 0) return `${label} -`
  const wins = finished.filter((trade) => trade.status === '익절').length
  return `${label} ${Math.round((wins / finished.length) * 100)}%`
}

function daysFromFirstTrade(targetTrades: TradeLog[]) {
  if (targetTrades.length === 0) return 0
  const timestamps = targetTrades.map((trade) => new Date(trade.buyDate.replaceAll('.', '-')).getTime())
  const first = Math.min(...timestamps)
  const latest = Math.max(...timestamps)
  return Math.max(1, Math.ceil((latest - first) / 86_400_000) + 1)
}

function parseTradeDate(value: string) {
  return new Date(value.replaceAll('.', '-')).getTime()
}

function holdingPeriodDays(trade: TradeLog) {
  const endTime = trade.status === '보유 중' ? Date.now() : parseTradeDate(trade.sellDate)
  return Math.max(0, Math.ceil((endTime - parseTradeDate(trade.buyDate)) / 86_400_000))
}

function parsePriceValue(value: string) {
  const parsed = Number(value.replace(/[^0-9.-]/g, ''))
  return Number.isFinite(parsed) ? parsed : null
}

function currentReturnPct(trade: TradeLog) {
  const buyPrice = parsePriceValue(trade.buyPrice)
  const currentPrice = parsePriceValue(searchUniverse.find((stock) => stock.ticker === trade.ticker)?.currentPrice ?? '')

  if (!buyPrice || currentPrice === null) return null
  return ((currentPrice - buyPrice) / buyPrice) * 100
}

function tradeKey(trade: TradeLog) {
  return `${trade.ticker}-${trade.buyDate}`
}

const gnbMenus = ['HOME', '가치 분석', '기술 분석', '시장 주요 이벤트', '시장 트렌드']
const valueMetricColumns: Array<{ label: string; value: (metric: ValuationMetric) => string; tooltip?: string }> = [
  { label: 'Market Cap', value: (metric) => metric.marketCap, tooltip: '기업의 전체 시장 가치입니다. 클수록 안정성은 높지만 성장 여지는 제한될 수 있으며, 동종 업계와 비교해 과도하게 비싼지 함께 봅니다.' },
  { label: 'Sales', value: (metric) => metric.sales, tooltip: '최근 매출 규모입니다. 클수록 사업 기반이 크다는 뜻이며, 성장률 없이 매출만 크면 매력도가 낮을 수 있습니다.' },
  { label: 'Sales Q/Q', value: (metric) => metric.salesQoq, tooltip: '전분기 대비 매출 성장률입니다. 양수이고 높을수록 단기 모멘텀이 좋고, 마이너스가 지속되면 수요 둔화를 의심합니다.' },
  { label: 'Sales Y/Y (TTM)', value: (metric) => metric.salesYoyTtm, tooltip: '최근 12개월 매출의 전년 대비 성장률입니다. 높을수록 구조적 성장 가능성이 크고, 둔화되면 밸류에이션 부담을 낮춰 봅니다.' },
  { label: 'Sales past 3/5Y', value: (metric) => metric.salesPastYears, tooltip: '최근 3년/5년 매출 성장 흐름입니다. 꾸준히 높으면 장기 성장성이 좋고, 변동이 크면 사이클 산업 여부를 확인합니다.' },
  { label: 'Current Ratio', value: (metric) => metric.currentRatio, tooltip: '유동자산을 유동부채로 나눈 단기 지급 능력입니다. 보통 1 이상이면 안정적으로 보고, 너무 낮으면 단기 재무 리스크가 커질 수 있습니다.' },
  { label: 'P/FCF', value: (metric) => metric.priceToFreeCashFlow, tooltip: '시가총액 대비 잉여현금흐름 배수입니다. 낮을수록 현금창출력 대비 저렴하고, 높으면 미래 성장 기대가 이미 반영됐을 수 있습니다.' },
  { label: 'P/S', value: (metric) => metric.priceToSales, tooltip: '시가총액을 매출로 나눈 배수입니다. 낮을수록 매출 대비 가격 부담이 작고, 고성장 기업은 업계 평균과 함께 비교합니다.' },
  { label: 'PER', value: (metric) => metric.per, tooltip: '주가를 주당순이익으로 나눈 배수입니다. 낮을수록 이익 대비 저렴할 수 있고, 성장률이 낮은데 높으면 부담이 큽니다.' },
  { label: 'PBR', value: (metric) => metric.pbr, tooltip: '주가를 주당순자산으로 나눈 배수입니다. 낮을수록 장부가 대비 저렴하고, ROE가 낮은 기업의 높은 PBR은 주의합니다.' },
  { label: 'ROE', value: (metric) => metric.roe, tooltip: '자기자본으로 얼마나 이익을 냈는지 보여줍니다. 높고 꾸준할수록 자본 효율이 좋으며, 부채로 높아진 ROE인지 함께 확인합니다.' },
  { label: 'PEG', value: (metric) => metric.peg, tooltip: 'PER을 이익 성장률로 나눈 값입니다. 1 안팎이면 성장 대비 가격이 합리적이고, 높을수록 성장 대비 비싸다는 신호입니다.' },
  { label: 'Shares Outstanding', value: (metric) => metric.sharesOutstanding, tooltip: '시장에 발행된 총 주식 수입니다. 증가하면 주주 지분이 희석될 수 있고, 자사주 매입으로 감소하면 주당 가치에 우호적입니다.' },
  { label: 'Gross Margin', value: (metric) => metric.grossMargin, tooltip: '매출에서 원가를 뺀 매출총이익률입니다. 높을수록 제품 경쟁력과 가격 결정력이 좋고, 하락 추세면 원가 부담을 의심합니다.' },
  { label: 'Oper. Margin', value: (metric) => metric.operatingMargin, tooltip: '영업이익을 매출로 나눈 수익성 지표입니다. 높고 안정적이면 본업 경쟁력이 좋고, 마이너스면 성장보다 비용 구조를 먼저 봅니다.' },
  { label: 'EPS (TTM)', value: (metric) => metric.epsTtm, tooltip: '최근 12개월 주당순이익입니다. 높고 증가하면 이익 체력이 좋으며, 일회성 이익인지 확인이 필요합니다.' },
  { label: 'EPS Next Y', value: (metric) => metric.epsNextYear, tooltip: '다음 해 예상 주당순이익입니다. 현재 EPS보다 높으면 이익 성장 기대가 있고, 예상치 하향이 반복되면 보수적으로 봅니다.' },
  { label: 'EPS Q/Q (%)', value: (metric) => metric.epsQoq, tooltip: '전분기 대비 주당순이익 변화율입니다. 양수이고 높으면 단기 실적 모멘텀이 좋고, 마이너스가 지속되면 수익성 둔화를 의심합니다.' },
  { label: 'Rule of 40%', value: (metric) => metric.ruleOf40, tooltip: '성장률과 수익성을 함께 보는 지표입니다. 40% 이상이면 성장과 수익의 균형이 좋다고 보고, 낮거나 음수면 성장의 질을 다시 확인합니다.' },
  { label: '실적발표일', value: (metric) => metric.earningsDate },
]

function MetricValue({
  children,
  tooltip,
  onTooltipOpen,
  onTooltipClose,
}: {
  children: string
  tooltip?: string
  onTooltipOpen: (tooltip: TooltipState) => void
  onTooltipClose: () => void
}) {
  if (!tooltip) return <>{children}</>

  const openTooltip = (element: HTMLElement) => {
    const rect = element.getBoundingClientRect()
    const minX = 280
    const maxX = window.innerWidth - 280
    const centeredX = rect.left + rect.width / 2

    onTooltipOpen({
      text: tooltip,
      x: Math.min(Math.max(centeredX, minX), maxX),
      y: rect.top - 8,
    })
  }

  return (
    <button
      className="metric-tooltip-trigger"
      type="button"
      onBlur={onTooltipClose}
      onClick={(event) => openTooltip(event.currentTarget)}
      onFocus={(event) => openTooltip(event.currentTarget)}
      onMouseEnter={(event) => openTooltip(event.currentTarget)}
      onMouseLeave={onTooltipClose}
    >
      {children}
    </button>
  )
}

function ValueAnalysisPage({
  stocks,
  viewMode,
  onAddStock,
  onTooltipOpen,
  onTooltipClose,
}: {
  stocks: Stock[]
  viewMode: 'personal' | 'operator'
  onAddStock: () => void
  onTooltipOpen: (tooltip: TooltipState) => void
  onTooltipClose: () => void
}) {
  const valueBlankRows = Math.max(0, 20 - stocks.length)

  return (
    <section className="panel value-analysis-panel">
      <div className="section-heading value-analysis-heading">
        <div>
          <h2>가치 분석</h2>
          <p>Home 관심 종목 기준으로 적정가, 현재가, 투자 의견과 핵심 재무 지표를 확인합니다.</p>
        </div>
        <span>{stocks.length}개</span>
      </div>

      <div className="sheet-wrap value-analysis-sheet">
        {stocks.length === 0 ? (
          <div className="watchlist-empty-panel">
            <div className="empty-watchlist">
              <strong>관심 종목이 없습니다.</strong>
              <span>Home에서 종목을 추가하면 가치 분석 표에 표시됩니다.</span>
              {viewMode === 'personal' && (
                <button type="button" onClick={onAddStock}>관심 종목 추가</button>
              )}
            </div>
          </div>
        ) : (
          <table className="sheet-table value-analysis-table">
            <thead>
              <tr>
                <th>종목명</th>
                <th>티커</th>
                <th>구분</th>
                <th>산업</th>
                <th>적정 주가 범위</th>
                <th>현재가</th>
                <th>투자 의견</th>
                {valueMetricColumns.map((column) => (
                  <th key={column.label}>
                    <MetricValue
                      tooltip={column.tooltip}
                      onTooltipClose={onTooltipClose}
                      onTooltipOpen={onTooltipOpen}
                    >
                      {column.label}
                    </MetricValue>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stocks.map((stock) => {
                const metric = valuationMetrics[stock.ticker]

                return (
                  <tr key={stock.ticker}>
                    <td className="name-data-cell">
                      <div className="name-cell">
                        <span className="market-flag" aria-hidden="true">{marketFlag(stock.market)}</span>
                        <span>{stock.name}</span>
                      </div>
                    </td>
                    <td className="ticker-cell">{stock.ticker}</td>
                    <td>{stock.market === 'KR' ? '성장주' : '가치주'}</td>
                    <td className="industry-cell">{stock.strategies[0].includes('스퀴즈') ? '반도체, AI, 소프트웨어' : '반도체, 커머스, 클라우드'}</td>
                    <td className="number-cell">{stock.fairPrice}</td>
                    <td className="number-cell">{stock.currentPrice}</td>
                    <td><span className={`status-badge ${statusClass(stock.valuation)}`}>{stock.valuation}</span></td>
                    {valueMetricColumns.map((column) => (
                      <td className="number-cell" key={column.label}>
                        {metric ? column.value(metric) : '-'}
                      </td>
                    ))}
                  </tr>
                )
              })}
              {Array.from({ length: valueBlankRows }).map((_, index) => (
                <tr className="blank-row" key={`value-blank-${index}`}>
                  {Array.from({ length: 7 + valueMetricColumns.length }).map((__, cellIndex) => (
                    <td key={`value-blank-cell-${index}-${cellIndex}`}>&nbsp;</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [watchlist, setWatchlist] = useState(initialWatchlist)
  const [operatorWatchlist, setOperatorWatchlist] = useState(operatorTickers)
  const [personalTradeLogs, setPersonalTradeLogs] = useState(personalTrades)
  const [isAddingStock, setIsAddingStock] = useState(false)
  const [viewMode, setViewMode] = useState<'personal' | 'operator'>('personal')
  const [selectedStrategy, setSelectedStrategy] = useState('전체')
  const [sortDirection, setSortDirection] = useState<'desc' | 'asc'>('desc')
  const [activeTooltip, setActiveTooltip] = useState<TooltipState | null>(null)
  const [selectedTickers, setSelectedTickers] = useState<string[]>([])
  const [selectedHoldingTradeKeys, setSelectedHoldingTradeKeys] = useState<string[]>([])
  const [isResetConfirmOpen, setIsResetConfirmOpen] = useState(false)
  const [isHoldingDeleteConfirmOpen, setIsHoldingDeleteConfirmOpen] = useState(false)
  const [activePage, setActivePage] = useState<'home' | 'value-analysis'>('home')
  const addStockButtonRef = useRef<HTMLButtonElement | null>(null)
  const inlineAddRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!isAddingStock) return

    const closeInlineAddOnOutsideClick = (event: MouseEvent) => {
      const target = event.target as Node

      if (inlineAddRef.current?.contains(target) || addStockButtonRef.current?.contains(target)) {
        return
      }

      setIsAddingStock(false)
    }

    document.addEventListener('mousedown', closeInlineAddOnOutsideClick)
    return () => document.removeEventListener('mousedown', closeInlineAddOnOutsideClick)
  }, [isAddingStock])

  const watchlistStocks = useMemo(
    () => watchlist
      .map((ticker) => searchUniverse.find((stock) => stock.ticker === ticker))
      .filter((stock): stock is Stock => Boolean(stock)),
    [watchlist],
  )

  const operatorStocks = useMemo(
    () => operatorWatchlist
      .map((ticker) => searchUniverse.find((stock) => stock.ticker === ticker))
      .filter((stock): stock is Stock => Boolean(stock)),
    [operatorWatchlist],
  )

  const searchResults = useMemo(() => {
    const normalized = normalizeQuery(query)
    if (!normalized) return []
    return searchUniverse.filter((stock) => {
      const ticker = normalizeQuery(stock.ticker)
      const name = normalizeQuery(stock.name)

      return ticker.startsWith(normalized) || name.startsWith(normalized)
    })
  }, [query])

  const scopedTrades = viewMode === 'personal' ? personalTradeLogs : operatorTrades
  const scopedOpenTrades = scopedTrades.filter((trade) => trade.status === '보유 중')
  const filteredTrades = scopedTrades
    .filter((trade) => selectedStrategy === '전체' || strategyCode(trade.strategy) === selectedStrategy)
    .slice()
    .sort((a, b) => {
      const aTime = new Date(a.buyDate.replaceAll('.', '-')).getTime()
      const bTime = new Date(b.buyDate.replaceAll('.', '-')).getTime()
      return sortDirection === 'desc' ? bTime - aTime : aTime - bTime
    })
  const visibleWinRates = [
    formatWinRate('통합', scopedTrades),
    ...strategyFilters
      .map((code) => formatWinRate(code, scopedTrades.filter((trade) => strategyCode(trade.strategy) === code))),
  ].join(', ')
  const strategyCriteriaLine = 'A/B/C(+20% 즉시, -30%), D(+12%, -25%, 최대 30일), E/F(+20% 후 MACD 둔화·5일 대기, -30%)'
  const investingDays = daysFromFirstTrade(scopedTrades)

  const addToWatchlist = (ticker: string) => {
    if (watchlist.length >= MAX_WATCHLIST_ITEMS) {
      setIsAddingStock(true)
      return
    }

    setWatchlist((current) => current.includes(ticker) ? current : [...current, ticker])
    setQuery('')
    setIsAddingStock(false)
  }

  const removeSelectedStocks = () => {
    if (viewMode === 'personal') {
      setWatchlist((current) => current.filter((ticker) => !selectedTickers.includes(ticker)))
    } else {
      setOperatorWatchlist((current) => current.filter((ticker) => !selectedTickers.includes(ticker)))
    }
    setSelectedTickers([])
  }

  const toggleSelectedTicker = (ticker: string) => {
    setSelectedTickers((current) => (
      current.includes(ticker)
        ? current.filter((item) => item !== ticker)
        : [...current, ticker]
    ))
  }

  const toggleSelectedHoldingTrade = (key: string) => {
    setSelectedHoldingTradeKeys((current) => (
      current.includes(key)
        ? current.filter((item) => item !== key)
        : [...current, key]
    ))
  }

  const removeSelectedHoldingTrades = () => {
    setPersonalTradeLogs((current) => current.filter((trade) => !selectedHoldingTradeKeys.includes(tradeKey(trade))))
    setSelectedHoldingTradeKeys([])
    setIsHoldingDeleteConfirmOpen(false)
  }

  const resetSystemRecords = () => {
    setWatchlist([])
    setPersonalTradeLogs([])
    setSelectedTickers([])
    setSelectedHoldingTradeKeys([])
    setQuery('')
    setIsAddingStock(false)
    setSelectedStrategy('전체')
    setIsResetConfirmOpen(false)
    setIsHoldingDeleteConfirmOpen(false)
  }

  const tableStocks = viewMode === 'personal' ? watchlistStocks : operatorStocks
  const isPersonalWatchlistEmpty = viewMode === 'personal' && tableStocks.length === 0
  const isPersonalWatchlistFull = viewMode === 'personal' && watchlist.length >= MAX_WATCHLIST_ITEMS
  const exampleStock = tableStocks[0]
  const showEmptyTradeExample = viewMode === 'personal' && tableStocks.length > 0 && scopedTrades.length === 0
  const showEmptyHoldingExample = viewMode === 'personal' && tableStocks.length > 0 && scopedOpenTrades.length === 0
  const tradeBlankRows = Math.max(3, 22 - filteredTrades.length - (showEmptyTradeExample ? 1 : 0))
  const watchlistBlankRows = Math.max(0, 10 - tableStocks.length)
  const holdingBlankRows = Math.max(0, 10 - scopedOpenTrades.length - (showEmptyHoldingExample ? 1 : 0))

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <img alt="공수성가 로고" className="brand-logo" src="/gongsu-logo.png" />
          <span>공수성가</span>
        </div>
        <nav className="gnb-menu" aria-label="주요 메뉴">
          {gnbMenus.map((menu) => {
            const isActive = (menu === 'HOME' && activePage === 'home') || (menu === '가치 분석' && activePage === 'value-analysis')

            return (
              <button
                className={isActive ? 'active' : ''}
                key={menu}
                type="button"
                onClick={() => {
                  if (menu === 'HOME') setActivePage('home')
                  if (menu === '가치 분석') setActivePage('value-analysis')
                }}
              >
                {menu}
              </button>
            )
          })}
        </nav>
        <div className="updated-text">
          <span>지표와 판단 결과는 2시간마다 업데이트됩니다.</span>
          <span>공수성가 또한 실제 데이터이며, 참고할 수 있게 제공됩니다.</span>
          <span>단, 모든 투자의 책임은 본인에게 있습니다.</span>
        </div>
        <div className="segmented-tabs global-tabs" aria-label="화면 기준">
          <button className={viewMode === 'personal' ? 'active' : ''} type="button" onClick={() => { setViewMode('personal'); setSelectedTickers([]); setSelectedHoldingTradeKeys([]) }}>
            본인
          </button>
          <button className={viewMode === 'operator' ? 'active' : ''} type="button" onClick={() => { setViewMode('operator'); setSelectedTickers([]); setSelectedHoldingTradeKeys([]) }}>
            공수성가
          </button>
        </div>
        <button className="reset-button" type="button" onClick={() => setIsResetConfirmOpen(true)}>
          초기화
        </button>
        <button className="login-button" type="button">로그인</button>
      </header>

      {activePage === 'home' ? (
      <section className="dashboard-grid">
        <section className={`panel trading-log-panel ${isPersonalWatchlistEmpty ? 'dimmed-panel' : ''}`}>
          <div className="log-header">
            <div className="log-title-row">
              <h2>트레이딩 로그</h2>
              <div className="strategy-filter" aria-label="전략 필터">
                {['전체', ...strategyFilters].map((code) => (
                  <button
                    className={selectedStrategy === code ? 'active' : ''}
                    key={code}
                    type="button"
                    onClick={() => setSelectedStrategy(code)}
                  >
                    {code}
                  </button>
                ))}
              </div>
            </div>
            <div className="log-sub-row">
              <div className="log-meta">
                <p>총 투자 기간 {investingDays}일 <span>|</span> 승률: {visibleWinRates}</p>
                <p>성공/실패 기준: {strategyCriteriaLine}</p>
              </div>
              <button
                className="sort-button"
                type="button"
                onClick={() => setSortDirection((current) => current === 'desc' ? 'asc' : 'desc')}
              >
                정렬
                <span aria-hidden="true">{sortDirection === 'desc' ? '↓' : '↑'}</span>
              </button>
            </div>
          </div>

          <div className="sheet-wrap trading-log-scroll">
            <table className="sheet-table trading-log-table">
              <thead>
                <tr>
                  <th>No</th>
                  <th>종목명</th>
                  <th>티커</th>
                  <th>매수 신호일</th>
                  <th>매수 신호 가격</th>
                  <th>매도 신호일</th>
                  <th>매도 신호 가격</th>
                  <th>기준</th>
                  <th>수익률</th>
                  <th>보유 기간</th>
                  <th>결과</th>
                </tr>
              </thead>
              <tbody>
                {showEmptyTradeExample && exampleStock && (
                  <tr className="example-row">
                    <td className="numbering-cell">예시</td>
                    <td>
                      <div className="name-cell">
                        <span className="market-flag" aria-hidden="true">{marketFlag(exampleStock.market)}</span>
                        <span>{exampleStock.name}</span>
                      </div>
                    </td>
                    <td className="ticker-cell">{exampleStock.ticker}</td>
                    <td>신호 발생 시</td>
                    <td className="number-cell">{exampleStock.currentPrice}</td>
                    <td className="dash-cell">-</td>
                    <td className="dash-cell">-</td>
                    <td><span className="example-note">매수 시그널 충족 시 기록됩니다.</span></td>
                    <td className="dash-cell">-</td>
                    <td className="dash-cell">-</td>
                    <td><span className="status-badge neutral">예시</span></td>
                  </tr>
                )}
                {filteredTrades.map((trade, index) => (
                  <tr key={`${trade.ticker}-${trade.buyDate}`}>
                    <td className="numbering-cell">{index + 1}</td>
                    <td>
                      <div className="name-cell">
                        <span className="market-flag" aria-hidden="true">{marketFlag(stockMarket(trade.ticker))}</span>
                        <span>{stockName(trade.ticker)}</span>
                      </div>
                    </td>
                    <td className="ticker-cell">{trade.ticker}</td>
                    <td>{trade.buyDate}</td>
                    <td className="number-cell">{trade.buyPrice}</td>
                    <td>{trade.sellDate}</td>
                    <td className={trade.sellPrice === '-' ? 'dash-cell' : 'number-cell'}>{trade.sellPrice}</td>
                    <td>
                      <StrategyTag
                        onTooltipClose={() => setActiveTooltip(null)}
                        onTooltipOpen={setActiveTooltip}
                        strategy={trade.strategy}
                      />
                    </td>
                    {trade.status === '보유 중' ? (
                      <td className="dash-cell">-</td>
                    ) : (
                      <td className={`number-cell ${returnClass(trade.returnPct)}`}>
                        {trade.returnPct >= 0 ? '+' : ''}{trade.returnPct.toFixed(1)}%
                      </td>
                    )}
                    <td>{holdingPeriodDays(trade)}</td>
                    <td>
                      <ResultBadge
                        onTooltipClose={() => setActiveTooltip(null)}
                        onTooltipOpen={setActiveTooltip}
                        trade={trade}
                      />
                    </td>
                  </tr>
                ))}
                {Array.from({ length: tradeBlankRows }).map((_, index) => (
                  <tr className="blank-row" key={`trade-blank-${index}`}>
                    <td className="numbering-cell">&nbsp;</td>
                    <td>&nbsp;</td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="right-column">
          <section className="panel watchlist-panel">
            <div className="section-heading">
              <div className="section-title-inline">
                <h2>관심 종목</h2>
                <span>총 {tableStocks.length}개</span>
              </div>
              <div className="heading-actions">
                {viewMode === 'personal' ? (
                  <>
                    {isPersonalWatchlistFull && (
                      <span className="watchlist-limit-copy">
                        관심 종목은 최대 {MAX_WATCHLIST_ITEMS}개까지 등록할 수 있습니다.
                        <br />
                        새 종목을 추가하려면 기존 관심 종목을 제거해 주세요.
                      </span>
                    )}
                    {selectedTickers.length > 0 && (
                      <button className="remove-selected-button" type="button" onClick={removeSelectedStocks}>
                        제거
                      </button>
                    )}
                    <button
                      className={`add-stock-button ${isPersonalWatchlistFull ? 'watchlist-limit-button' : ''}`}
                      disabled={isPersonalWatchlistFull}
                      ref={addStockButtonRef}
                      type="button"
                      onClick={() => setIsAddingStock((value) => !value)}
                    >
                      + 추가
                    </button>
                  </>
                ) : (
                  <button
                    aria-disabled="true"
                    className="add-stock-button readonly-mode-button"
                    tabIndex={-1}
                    type="button"
                  >
                    공수성가 기준
                  </button>
                )}
              </div>
            </div>

            {isAddingStock && viewMode === 'personal' && !isPersonalWatchlistFull && (
              <div className="inline-add" ref={inlineAddRef}>
                <input
                  autoFocus
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="삼성전자, 005930, AAPL"
                />
                {query && (
                  <div className="inline-results">
                    {searchResults.length > 0 ? searchResults.map((stock) => {
                      const isAlreadyAdded = watchlist.includes(stock.ticker)

                      return (
                        <button
                          disabled={isAlreadyAdded}
                          key={stock.ticker}
                          type="button"
                          onClick={() => addToWatchlist(stock.ticker)}
                        >
                          <span>
                            <strong>{stock.name}</strong>
                            <small>{stock.ticker} · {stock.market}</small>
                          </span>
                          <span>{isAlreadyAdded ? '이미 추가됨' : '추가하기'}</span>
                        </button>
                      )
                    }) : (
                      <div className="empty-result">검색 결과가 없습니다.</div>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="sheet-wrap watchlist-sheet">
              {tableStocks.length === 0 ? (
                <div className="watchlist-empty-panel">
                  <div className="empty-watchlist">
                    <strong>관심 종목이 없습니다.</strong>
                    <span>먼저 종목을 추가해 주세요.</span>
                    {viewMode === 'personal' && (
                      <button type="button" onClick={() => setIsAddingStock(true)}>관심 종목 추가</button>
                    )}
                  </div>
                </div>
              ) : (
                <table className="sheet-table watchlist-table">
                  <thead>
                    <tr>
                      {viewMode === 'personal' && <th>선택</th>}
                      <th>No</th>
                      <th>종목명</th>
                      <th>티커</th>
                      <th>적정 가격</th>
                      <th>현재가</th>
                      <th>가치 분석</th>
                      <th>기술 분석</th>
                      <th>시스템 보유</th>
                      <th>매수 전략</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableStocks.map((stock, index) => (
                      <tr key={stock.ticker}>
                        {viewMode === 'personal' && (
                          <td className="checkbox-cell">
                            <input
                              aria-label={`${stock.name} 선택`}
                              checked={selectedTickers.includes(stock.ticker)}
                              onChange={() => toggleSelectedTicker(stock.ticker)}
                              type="checkbox"
                            />
                          </td>
                        )}
                        <td className="numbering-cell">
                          <span>{index + 1}</span>
                        </td>
                        <td className="name-data-cell">
                          <div className="name-cell">
                            <span className="market-flag" aria-hidden="true">{marketFlag(stock.market)}</span>
                            <span>{stock.name}</span>
                          </div>
                        </td>
                        <td className="ticker-cell">{stock.ticker}</td>
                        <td className="number-cell">{stock.fairPrice}</td>
                        <td className="number-cell">{stock.currentPrice}</td>
                        <td><span className={`status-badge ${statusClass(stock.valuation)}`}>{stock.valuation}</span></td>
                        <td><span className={`status-badge ${statusClass(stock.opinion)}`}>{stock.opinion}</span></td>
                        <td>
                          {isSystemHolding(stock.ticker, scopedTrades) ? '보유 중' : '미보유'}
                        </td>
                        <td className={isSystemHolding(stock.ticker, scopedTrades) ? 'strategy-data-cell' : 'strategy-data-cell dash-cell'}>
                          {isSystemHolding(stock.ticker, scopedTrades) ? stock.strategies.map((strategy) => (
                            <StrategyTag
                              key={strategy}
                              onTooltipClose={() => setActiveTooltip(null)}
                              onTooltipOpen={setActiveTooltip}
                              strategy={strategy}
                            />
                          )) : '-'}
                        </td>
                      </tr>
                    ))}
                    {Array.from({ length: watchlistBlankRows }).map((_, index) => (
                      <tr className="blank-row" key={`watchlist-blank-${index}`}>
                        {viewMode === 'personal' && <td></td>}
                        <td className="numbering-cell">&nbsp;</td>
                        <td>&nbsp;</td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>

          <section className={`panel ${isPersonalWatchlistEmpty ? 'dimmed-panel' : ''}`}>
            <div className="section-heading">
              <div className="section-title-inline">
                <h2>보유중인 종목</h2>
                <span>총 {scopedOpenTrades.length}개</span>
              </div>
              <div className="heading-actions">
                {viewMode === 'personal' && (
                  <button
                    aria-hidden={selectedHoldingTradeKeys.length === 0}
                    className={`remove-selected-button ${selectedHoldingTradeKeys.length === 0 ? 'reserved-action-button' : ''}`}
                    tabIndex={selectedHoldingTradeKeys.length === 0 ? -1 : 0}
                    type="button"
                    onClick={() => {
                      if (selectedHoldingTradeKeys.length > 0) setIsHoldingDeleteConfirmOpen(true)
                    }}
                  >
                    제거
                  </button>
                )}
              </div>
            </div>

            <div className="sheet-wrap holding-sheet">
              <table className="sheet-table holding-table">
                <thead>
                  <tr>
                    {viewMode === 'personal' && <th>선택</th>}
                    <th>No</th>
                    <th>티커</th>
                    <th>종목명</th>
                    <th>신호일</th>
                    <th>매수 전략</th>
                    <th>현재 수익률</th>
                    <th>보유 기간</th>
                  </tr>
                </thead>
                <tbody>
                  {showEmptyHoldingExample && exampleStock && (
                    <tr className="example-row">
                      {viewMode === 'personal' && <td></td>}
                      <td className="numbering-cell">예시</td>
                      <td className="ticker-cell">{exampleStock.ticker}</td>
                      <td>
                        <div className="name-cell">
                          <span className="market-flag" aria-hidden="true">{marketFlag(exampleStock.market)}</span>
                          <span>{exampleStock.name}</span>
                        </div>
                      </td>
                      <td>신호 발생 시</td>
                      <td><span className="example-note">보유 전환 시 표시됩니다.</span></td>
                      <td className="dash-cell">-</td>
                      <td className="dash-cell">-</td>
                    </tr>
                  )}
                  {scopedOpenTrades.map((trade, index) => (
                    <tr key={`open-${tradeKey(trade)}`}>
                      {viewMode === 'personal' && (
                        <td className="checkbox-cell">
                          <input
                            aria-label={`${stockName(trade.ticker)} 보유 항목 선택`}
                            checked={selectedHoldingTradeKeys.includes(tradeKey(trade))}
                            onChange={() => toggleSelectedHoldingTrade(tradeKey(trade))}
                            type="checkbox"
                          />
                        </td>
                      )}
                      <td className="numbering-cell">{index + 1}</td>
                      <td className="ticker-cell">{trade.ticker}</td>
                      <td>
                        <div className="name-cell">
                          <span className="market-flag" aria-hidden="true">{marketFlag(stockMarket(trade.ticker))}</span>
                          <span>{stockName(trade.ticker)}</span>
                        </div>
                      </td>
                      <td>{trade.buyDate}</td>
                      <td>
                        <StrategyTag
                          onTooltipClose={() => setActiveTooltip(null)}
                          onTooltipOpen={setActiveTooltip}
                          strategy={trade.strategy}
                        />
                      </td>
                      {currentReturnPct(trade) === null ? (
                        <td className="dash-cell">-</td>
                      ) : (
                        <td className={`number-cell ${returnClass(currentReturnPct(trade) ?? 0)}`}>
                          {(currentReturnPct(trade) ?? 0) >= 0 ? '+' : ''}{(currentReturnPct(trade) ?? 0).toFixed(1)}%
                        </td>
                      )}
                      <td>{holdingPeriodDays(trade)}</td>
                    </tr>
                  ))}
                  {Array.from({ length: holdingBlankRows }).map((_, index) => (
                    <tr className="blank-row" key={`holding-blank-${index}`}>
                      {viewMode === 'personal' && <td></td>}
                      <td className="numbering-cell">&nbsp;</td>
                      <td>&nbsp;</td>
                      <td></td>
                      <td></td>
                      <td></td>
                      <td></td>
                      <td></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </section>
      ) : (
        <ValueAnalysisPage
          stocks={tableStocks}
          viewMode={viewMode}
          onTooltipClose={() => setActiveTooltip(null)}
          onTooltipOpen={setActiveTooltip}
          onAddStock={() => {
            setActivePage('home')
            setIsAddingStock(true)
          }}
        />
      )}
      {activeTooltip && (
        <div
          className="floating-tooltip"
          style={{
            left: activeTooltip.x,
            top: activeTooltip.y,
          }}
        >
          {activeTooltip.text}
        </div>
      )}
      {isResetConfirmOpen && (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="confirm-modal" role="dialog">
            <h3>본인 기록을 모두 초기화할까요?</h3>
            <p>본인 관심 종목, 보유중인 종목, 트레이딩 로그 등 시스템에 기록된 본인 데이터를 모두 삭제합니다. 단, 공수성가 데이터는 유지됩니다.</p>
            <div className="modal-actions">
              <button className="modal-cancel" type="button" onClick={() => setIsResetConfirmOpen(false)}>
                취소
              </button>
              <button className="modal-confirm" type="button" onClick={resetSystemRecords}>
                초기화
              </button>
            </div>
          </div>
        </div>
      )}
      {isHoldingDeleteConfirmOpen && (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="confirm-modal" role="dialog">
            <h3>선택한 보유 종목을 삭제할까요?</h3>
            <p>선택한 보유중인 종목 기록이 삭제되며, 연결된 트레이딩 로그도 함께 삭제됩니다. 이 작업은 복구할 수 없습니다.</p>
            <div className="modal-actions">
              <button className="modal-cancel" type="button" onClick={() => setIsHoldingDeleteConfirmOpen(false)}>
                취소
              </button>
              <button className="modal-confirm" type="button" onClick={removeSelectedHoldingTrades}>
                삭제
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}

export default App
