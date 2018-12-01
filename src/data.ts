export interface IRange {
  start: number;
  end: number;
}

export const RANGES = [{
  start: 7,
  end: 10,
  extra: 'fuck0',
  title: 'TiDB session.Execute',
}, {
  start: 5,
  end: 20,
  extra: 'fuck1',
  title: 'TiDB session.getTxnFuture',
}, {
  start: 10,
  end: 20,
  extra: 'fuck2',
  title: 'TiDB GetTsAsync',
}, {
  start: 4,
  end: 28,
  extra: 'fuck3',
  title: 'TiDB ParseSQL',
}];

export function createDataSets(ranges: ISpan[]) {
  // const minimum = Math.min(...ranges.map(r => r.startPercent));
  ranges = ranges.sort((a, b) => a.startTime - b.startTime);
  const minimum = Math.min(...ranges.map(r => r.startTime));
  const maximum = Math.max(...ranges.map(r => r.startTime + r.duration));
  return [{
    key: 'pre',
    backgroundColor: 'rgba(0,0,0,0)',
    hoverBackgroundColor: 'rgba(0,0,0,0)',
    hoverBorderColor: 'rgba(0,0,0,0)',
    data: ranges.map(r => r.startTime - minimum),
  }, {
    key: 'current',
    backgroundColor: 'rgba(255,99,132,0.6)',
    hoverBackgroundColor: 'rgba(255,99,132,1)',
    hoverBorderColor: 'rgba(0,0,0,0)',
    data: ranges.map(r => r.duration),
  }, {
    key: 'next',
    backgroundColor: 'rgba(0,0,0,0)',
    hoverBackgroundColor: 'rgba(0,0,0,0)',
    hoverBorderColor: 'rgba(0,0,0,0)',
    data: ranges.map(r => maximum - r.startTime - r.duration),
  }];
}

export function formatTime(ms: number) {
  if (ms < 1500) {
    return `${ms}ms`;
  }
  if (ms < 100 * 1000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  if (ms < 100 * 1000 * 60) {
    return `${(ms / 1000 / 60).toFixed(1)}min`;
  }
  return `${(ms / 1000 / 60 / 60).toFixed(1)}h`;
}

export interface ISpan {
  duration: number;
  flags: number;
  logs: any[];
  operationName: string;
  processID: string;
  references: Array<{
    refType: string;
    traceID: string;
    spanID: string;
  }>;
  spanID: string;
  startTime: number;
  tags: Array<{
    key: string;
    type: string;
    value: string;
  }>;
  traceID: string;
  warnings: null;
}

export interface ITrace {
  traceID: string;
  spans: ISpan[];
}
